# ADR 013 — Track 1：從 EDA 發現到模型選擇與訓練資料格式

**狀態：** 已接受
**日期：** 2026-05-30
**更新：** 2026-05-30（以 10k 資料集重算所有統計數字）
**相關：** ADR-008（Bigram 基線）、ADR-009（GPT-2 策略）、ADR-011（參數 EDA）、ADR-012（EDA 統整）

---

## 背景

完成 EDA 與參數解析（ADR-011、ADR-012）後，需要根據發現正式決定：

1. GPT-2 模型應採用「三個獨立 family 模型」還是「單一 family 條件化模型」
2. 訓練序列的格式應僅包含步驟名稱，還是加入 family token 或參數 token

以下決策基於以 10k 資料集（`ic_10k.csv`、`igbt_10k.csv`、`mosfet_10k.csv`）重算的 EDA 數據（括號內為原 1k 估計值，供比較）：

| 發現 | 10k 數值 | (1k 估計) |
|---|---|---|
| Per-family 條件熵 H(next\|current) | IC=0.871, IGBT=0.939, MOSFET=0.764 bits | (0.89 / 0.95 / 0.78) |
| Combined 條件熵 | **1.095 bits** | (1.11) |
| 三 family 共有步驟 | **94 步驟（47%）** | (73 步驟) ※ |
| Family 專屬 bigram 比例 | IC=31.8%, IGBT=27.2%, MOSFET=21.2% | (40–46%) |
| Optional 步驟數（出現率 <100%） | IC=73, IGBT=70, MOSFET=68 | (未量化) |
| 總訓練序列 | **30,000（每 family 10,000）** | (3,000) |
| 詞彙量 | **198（與 1k 完全一致，已收斂）** | (198) |
| 參數 PCA 解釋變異 | PC1=26.5%, PC2=22.4%，合計 49% | — |
| 跨 family 最大溫度差 | ≤50°C（集中於熱處理步驟） | — |
| 參數覆蓋率 | 94–95%（三個 family 均一致） | — |

※ 1k EDA 記錄「73 共有步驟」是以參數檔（374 行）為分母；此處 94 是以序列詞彙（198 個 step name）為分母，兩者計算對象不同。

---

## 決策一：模型架構——單一 Family 條件化 GPT-2

**決定：** 訓練一個 unified GPT-2 模型，以 `[FAMILY]` 作為 BOS conditioning token，而非三個獨立模型。

| 選項 | 決定 | 原因 |
|---|---|---|
| 三個獨立模型（bigram 做法） | 否決 | 放棄 94 個共享步驟的 transfer learning；三份 checkpoint 管理複雜 |
| Unified + family prefix token | **採用** | 共享骨架學習製程物理約束；單一 checkpoint；IGBT 6 個 litho level 需要 family context 才能消歧 |
| 無 family conditioning | 否決 | Combined H=1.095 bits > per-family 0.764–0.939 bits；模型無法區分 ALIGN MASK LEVEL 5 是否合法 |

**關鍵依據：**
IGBT 有 6 個 litho level，IC 與 MOSFET 只有 4 個。沒有 family token，模型看到
`ALIGN MASK LEVEL 5` 時無法判斷這是合法步驟（IGBT）還是異常（MOSFET/IC）。

---

## 決策二：訓練資料格式——三層漸進策略

依競賽時程分三個層級，依序評估效益再推進。

### Level 0（已實作）：純步驟序列

```
RECEIVE WAFER LOT | LOT IDENTIFICATION | INITIAL WAFER INSPECTION | ...
```

- Bigram Top-1 = 68.3%，Token Acc = 6.2%
- GPT-2 zero-shot Top-1 = 13.3%（下界）

### Level 1（Gate 1 後立即實作）：Family Prefix Token

```
[IC]    RECEIVE WAFER LOT | LOT IDENTIFICATION | ...
[IGBT]  RECEIVE WAFER LOT | INITIAL CLEANING   | ...
[MOSFET] RECEIVE WAFER LOT | EPITAXIAL GROWTH   | ...
```

**改動：** 僅新增 3 個 special token（`[IC]`、`[IGBT]`、`[MOSFET]`），無需修改 tokenizer 架構。

**預期效益：** combined H (1.095) − per-family H (0.764–0.939) = 0.16–0.33 bits entropy reduction → 預估 Top-1 提升 +3–8 個百分點。

### Level 2（Gate 2 通過後評估）：參數量化 Token

僅加入 top-discriminating 參數（temperature_c、time_min），量化為 bucket token：

```
THERMAL OXIDATION [T=high][t=med] | DEPOSIT GATE OXIDE [T=mid] | ...
```

Bucket 定義（依製程範圍）：

| Token | temperature_c | Token | time_min |
|---|---|---|---|
| `[T=low]` | 0–400°C | `[t=short]` | 0–5 min |
| `[T=mid]` | 400–800°C | `[t=med]` | 5–30 min |
| `[T=high]` | 800–1200°C | `[t=long]` | >30 min |

**觸發條件：** 僅在 Gate 2 後 Top-1 停滯於 0.40–0.55 時評估。
**注意：** 序列長度增加約 30%，需重新確認 `max_seq_len` 設定。

---

## 決策三：參數作為訓練訊號的定位

| 應用 | 採用 | 原因 |
|---|---|---|
| 主要分類訊號 | 否 | PCA 49% variance，families 重疊；sequence grammar 更強 |
| Task 3 異常偵測 | 否 | 競賽 Task 3 規則為 sequence ordering，非 parameter range 異常 |
| Level 2 輔助 conditioning | 條件式 | 僅在 Top-1 到達瓶頸時才加入；以 temperature_c, time_min 為主 |
| 跨 family 共享步驟消歧 | 是 | 73 個共享步驟中，IGBT RTA 1050°C vs MOSFET/IC 1000°C 可作為 soft signal |

**結論：** 參數是補充訊號，不是替代訊號。Sequence grammar 才是主要分類依據。

---

## 不採用的方案

| 方案 | 原因 |
|---|---|
| Cross-attention parameter embedding（步驟 token + 參數向量雙路架構） | 36 小時內無法穩定訓練；架構複雜度超過預期效益 |
| Per-family fine-tuning（shared backbone + 三個 adapter） | 需要 PEFT 框架；Gate 1 前不應引入新依賴 |
| Full 12-key parameter feature matrix | 每行 null rate >50%，PCA 降維結果不穩定（已驗證於 eda_parameters.py） |

---

## 實作順序

```
Gate 1  ─→  Level 0 + Level 1 同時訓練（單次改動：加 family prefix token）
Gate 2  ─→  確認 Top-1 ≥ 0.40
Gate 2+ ─→  視 Top-1 plateau 決定是否加 Level 2
Gate 3  ─→  Task 2 token accuracy 是主要改善目標（從 6.2% 到 ≥40%）
```

---

## 10k 資料集的新發現與影響

1. **Family-exclusive bigram 比例降低（40–46% → 21–32%）**：原 1k 估計因樣本稀疏而高估了家族排他性。10k 資料揭示更多跨 family 共享的 bigram transition。這稍微削弱了「三個獨立模型」的動機，但 combined H 仍高於 per-family H，故 family conditioning 的決定不變。

2. **Optional 步驟佔多數（68–73 個，>50% 詞彙）**：Task 2 token accuracy 低（6.2%）的根本原因更清晰——bigram 在遇到 optional branch 時會選錯，後續所有步驟跟著錯。GPT-2 fine-tuning 的主要目標是**學習 optional step 的出現條件**，而非學習 mandatory backbone。

3. **詞彙已收斂**：10k 新增的 9,000 條序列沒有帶來新詞彙（198 個 step name 不變）。額外資料的主要貢獻是 optional step 頻率的統計精度，而非詞彙擴充。

4. **10k 資料位置**：`ic_10k.csv`、`igbt_10k.csv`、`mosfet_10k.csv` 目前在 project root。應搬移至 `data/raw/training_data/` 並更新訓練 config 路徑（待 Gate 1 後執行）。

---

## 後果

- `data/processed/parameters_parsed.json` 在 Level 2 之前不進入訓練管線，保留為 EDA reference
- 改變訓練格式的唯一允許時機是通過對應 Gate 後（Rule 5：一次只改一個變數）
- Task 3 已由 `validate_sequence()` 解決，不需要任何模型改進（ADR-010）
- Task 2 的主要難點是 **optional step variation**（68–73 個 per family），GPT-2 fine-tuning 的目標是 token accuracy 從 6.2% 提升至 ≥40%
- 10k 資料搬移至 `data/raw/training_data/` 並更新 `gpt2_finetune_v1.yaml` 路徑是 Gate 1 前的 **blocking 工作**
