# ADR 014 — Track 1：基於步驟描述的 Embedding 初始化

**狀態：** 已接受  
**日期：** 2026-05-30  
**相關：** ADR-013（模型選擇與訓練資料格式）、ADR-011（參數 EDA）、ADR-012（EDA 統整）

---

## 背景

GPT-2 的 `transformer.wte`（shape `[205, 256]`）預設為隨機初始化（Gaussian，σ≈0.02）。
模型必須純粹從序列 co-occurrence 學習步驟間的物理語意關係，例如：
- "THERMAL OXIDATION" 和 "ANNEAL" 都是熱製程，應在 embedding 空間中接近
- "ALIGN MASK LEVEL 1" 到 "LEVEL 6" 是同一類型操作，應形成有序的 cluster
- IGBT 專屬步驟（6 個 litho level）應在 embedding 空間中靠近彼此

競賽提供了兩份額外資料可作為先驗知識：
1. `*_Longdescr.csv`：每個步驟的文字描述（設備型號、化學式等）
2. `*_longdescription_parameters.csv`：已解析的量化參數（temperature_c、time_min 等）
3. `data/processed/parameters_parsed.json`：Claude API 解析後的結構化參數

EDA 分析（`scripts/eda_entropy_params.py`）已確認參數在 training token 層次是冗餘的
（H(next | step, family, param) = H(next | step, family)），但這不代表參數對初始化 embedding
無益——embedding 初始化利用的是**靜態語意知識**，而非動態序列預測信號。

---

## 決策：使用三層 feature vector 初始化 wte

### Feature 架構（27-dim → Linear → 256-dim）

```
[category_onehot(8)] + [numeric_params(3)] + [name_tfidf(16)] = 27-dim
         ↓
    Linear(27, 256, bias=False) + LayerNorm
         ↓
  model.transformer.wte.weight  (初始化，不 freeze)
```

#### Layer 1：製程類別（8-dim one-hot）

基於步驟名稱關鍵字分類：

| 類別 | 關鍵字匹配 |
|---|---|
| `litho`   | ALIGN, EXPOSE, DEVELOP, COAT PHOTORESIST, PHOTO |
| `thermal` | OXIDATION, ANNEAL, RTA, DIFFUSION, CURE |
| `deposit` | DEPOSIT, CVD, PVD, EPITAX, GROW |
| `etch`    | ETCH, STRIP, CMP, POLISH |
| `implant` | IMPLANT, ION |
| `measure` | MEASURE, INSPECT, SCAN, KLA, CHECK |
| `clean`   | CLEAN, RINSE, DRY, HF DIP, RCA |
| `other`   | 以上皆不符合（TEST、SHIP、RECEIVE 等） |

優先匹配較長關鍵字以避免誤判（e.g. "THERMAL OXIDATION" → thermal，非 etch）。

#### Layer 2：量化參數（3-dim continuous）

從 `parameters_parsed.json` 提取，針對每個 step 取各 family 的均值，再 min-max normalize 到 [0, 1]：

| 維度 | 參數 | 無值時 |
|---|---|---|
| 0 | `temperature_c` | 0.0 |
| 1 | `time_min`（time_s 轉換） | 0.0 |
| 2 | litho level number（從名稱提取）| 0.0 |

#### Layer 3：步驟名稱 TF-IDF（16-dim）

- 把 198 個步驟名稱當作文件集合（每個詞是一個 term）
- 計算 TF-IDF 矩陣（shape `[198, vocab_words]`）
- TruncatedSVD 降維到 16 dim
- 結果：同類步驟自然聚集（"ALIGN MASK LEVEL 1/2/3" 幾乎重疊；"MEASURE X" 形成 cluster）

### 無描述步驟的處理（31%，62/198 個步驟）

136/198 個步驟有 `parameters_parsed.json` 記錄。剩餘 62 個步驟：
- Layer 1（category）：從步驟名稱關鍵字匹配，覆蓋率接近 100%
- Layer 2（numeric）：填 0（無參數資訊）
- Layer 3（TF-IDF）：只用步驟名稱，無描述文字補充

### 不 freeze embedding

初始化後讓 `wte` 繼續 fine-tune。理由：
- Freeze 會阻止模型用序列 co-occurrence 信號修正 embedding
- 語意初始化提供更好的梯度方向，不需要靠 freeze 維持

### Special tokens 處理

`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`, `[IC]`, `[IGBT]`, `[MOSFET]` 維持隨機初始化，不套用描述向量。

---

## 實作位置

```
src/models/
  desc_embed.py       ← 新檔：build_description_feature_matrix()
  process_lm.py       ← 修改：build_model() 加 embedding_init 參數
```

`train.py` 讀取 config 中的 `model.embedding_init` 和 `model.desc_path`，傳入 `build_model()`。

Config 範例（`leonardo_B_desc_embed.yaml`）：

```yaml
model:
  type: gpt2_scratch
  embedding_init: description
  desc_path: data/processed/parameters_parsed.json
```

---

## 不採用的方案

| 方案 | 否決原因 |
|---|---|
| 使用 pre-trained GPT-2 weights | Vocabulary mismatch（50K BPE vs 198 custom tokens）；fine-tune 會 overfit |
| Cross-attention dual-path（描述 + 序列） | 36 小時內無法穩定訓練（已在 ADR-013 否決）|
| 把描述加入訓練序列（token 化） | 與 Level 2 相同問題：對 step+family 固定，entropy 不降低 |
| Freeze embedding | 阻止模型從序列學習修正語意方向 |

---

## 預期效益與評估標準

| 指標 | 預期 |
|---|---|
| 收斂速度 | step 300 時 loss 低於 random init 約 0.05–0.15 nats |
| Top-1 Accuracy @step 3000 | 相比 Exp A（random init）+1–5% |
| 實作風險 | 低：只影響 wte 初始值，不改架構或訓練迴圈 |

**評估條件**：在 Leonardo 上與 `leonardo_A_family_token.yaml` 同條件跑，唯一差異為 `embedding_init: description`（Rule 5）。

---

## 後果

- `src/models/desc_embed.py` 為一次性建構工具，僅在 `build_model()` 時呼叫
- `data/processed/parameters_parsed.json` 正式進入訓練管線（但只影響初始化，不影響訓練資料格式）
- Exp B 失敗（Top-1 低於 Exp A）時，直接採用 Exp A checkpoint，無架構變更成本
- 此方向在 Gate 2 通過前為**選配實驗**，不影響 Gate 1 判斷標準
