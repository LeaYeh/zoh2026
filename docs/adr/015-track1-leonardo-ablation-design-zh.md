# ADR 015 — Track 1：Leonardo 三路消融實驗設計（繁體中文版）

**狀態：** 已採納  
**日期：** 2026-05-30  
**相關：** ADR-012（EDA 發現）、ADR-013（模型選擇）、ADR-014（描述 Embedding 初始化）

---

## 背景

在 Leonardo A100 上安排三組實驗，用於在確定最終訓練設定前，評估最佳 embedding 策略。
每組實驗相對前一組恰好改變一個變數（Rule 5）。

---

## 共用模型架構

三組實驗使用完全相同的 transformer 設定：

| 元件 | 數值 |
|---|---|
| 模型類型 | GPT-2 風格 decoder-only transformer（從零訓練）|
| 詞彙 | 205 tokens — 198 步驟名 + `<PAD>/<BOS>/<EOS>/<UNK>` + `[IC]/[IGBT]/[MOSFET]` |
| `n_embd` | 256 |
| `n_layer` | 6 |
| `n_head` | 8 |
| FFN 寬度 | 1024（4× n_embd）|
| `max_seq_len` | 256（序列最長 ~160 tokens，有 6× 餘裕）|
| 總參數量 | ~4.86M |
| `lm_head` | weight-tied to `wte`（不額外增加參數）|

---

## 共用訓練設定

| 超參數 | 數值 | 理由 |
|---|---|---|
| 訓練資料 | `ic_10k.csv`, `igbt_10k.csv`, `mosfet_10k.csv` | 30k 序列，198 步驟詞彙已收斂 |
| `val_ratio` | 0.10 | 3k 驗證序列 |
| `family_prefix` | `true` | 每個序列前置 `[IC]/[IGBT]/[MOSFET]` — 降低熵 0.16–0.33 bits |
| `steps` | 3,000 | |
| `batch_size` | 64 | |
| `lr` | 5e-4 | AdamW |
| `warmup_steps` | 300 | warmup 後 cosine decay |
| `grad_clip` | 1.0 | |
| `eval_every` | 300 | 300 個 val 樣本評估 Top-1/3/5 + MRR |
| `seed` | 42 | torch + numpy 統一 |
| PAD masking | PAD 位置 `labels = -100` | Rule 3 |

---

## 三路消融對照表

| | **Exp A** | **Exp B** | **Exp C** |
|---|:---:|:---:|:---:|
| Config 檔案 | `leonardo_A_family_token.yaml` | `leonardo_B_desc_embed.yaml` | `leonardo_C_desc_categ.yaml` |
| WandB run name | `leonardo_A_family_token\|fam` | `leonardo_B_desc_embed\|fam\|desc` | `leonardo_C_desc_categ\|fam\|desc\|cat` |
| **`wte` 初始化** | 隨機（σ=0.02）| 描述特徵初始化 | 描述特徵初始化 |
| **`CategoryAwareEmbedding`** | — | — | ✓ |
| 額外可訓練參數 | 0 | 0 | +2,048（8 × 256）|
| **相對前一組改變的變數** | —（baseline）| 僅初始化方式 | 僅加上 cat_embed |

### Exp A — 隨機初始化（baseline）

`wte` 以標準 GPT-2 Gaussian（σ=0.02，mean=0）初始化。
不注入任何先驗知識，序列 co-occurrence 是唯一的學習訊號。

### Exp B — 描述 Embedding 初始化（ADR-014）

`wte` 從 27-dim 特徵向量投影至 256-dim 後初始化：

```
[0:8]   製程類別 one-hot  — 8 類（litho/thermal/deposit/etch/implant/measure/clean/other）
[8:11]  量化參數          — temperature_c、time_min、litho_level（來自 parameters_parsed.json）
[11:27] TF-IDF SVD        — 步驟名稱降維至 16-dim

→ Linear(27, 256, no bias)
→ L2 normalize，縮放至 GPT-2 初始範數：0.02 × √256 ≈ 0.32
```

- 136/198 個步驟有 `parameters_parsed.json` 記錄；62 個步驟以 numeric=0 補值
- Special tokens（`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`, `[IC]`, `[IGBT]`, `[MOSFET]`）維持隨機初始化
- `wte` **不 freeze** — 描述初始化提供更好的梯度起點，而非限制

### Exp C — 描述初始化 + CategoryAwareEmbedding

用 `CategoryAwareEmbedding` 包覆 `model.transformer.wte`：

```python
embedding(token_id) = wte(token_id) + cat_embed(category_id)
```

- `cat_embed`：`nn.Embedding(8, 256)`，zero-initialized（初始為 no-op，逐步學習）
- `step_to_cat_ids`：靜態 buffer，將每個詞彙 token 映射到 8 個類別 ID 之一
- `.weight` property 代理 `wte.weight`，保留 `lm_head` weight tying
- 假說：共享的 per-category 殘差有助於所有 litho 步驟一起收斂

---

## 決策標準

以 step 3,000 的 `eval/top1` 為判斷依據：

| 結果 | 動作 |
|---|---|
| A 最佳 | 使用 Exp A 設定做最終訓練；捨棄 B 和 C |
| B 最佳 | 使用 Exp B 設定；確認描述初始化有效 |
| C 最佳 | 使用 Exp C 設定；確認 category embedding 提供額外訊號 |
| B ≈ C | 使用 B（較簡單）；category embedding 無可量測效益 |
| 全部 < 0.40 | Gate 2 失敗 — 降低 LR、增加資料、或擴大模型 |

獲勝的 config 成為後續實驗的基礎（每次改一個變數）。

---

## 執行指令（Leonardo）

```bash
sbatch scripts/slurm/train.slurm configs/exp/leonardo_A_family_token.yaml
sbatch scripts/slurm/train.slurm configs/exp/leonardo_B_desc_embed.yaml
sbatch scripts/slurm/train.slurm configs/exp/leonardo_C_desc_categ.yaml
```

跑完後同步 WandB：

```bash
wandb sync ~/zoh2026/wandb/offline-run-*/
```

---

## Local Smoke Test 結果（僅供參考）

在 2-layer / 128-dim toy 模型上執行，200 steps，每 family 1k 序列，MPS（Apple Silicon）。
**不可與 Leonardo 結果直接比較。** 先前數字（0.047/0.057）已作廢——由 EOS inference bug 導致，
已於 2026-05-30 修復：`predict_next_step` 將 `[BOS…EOS]` 傳入模型，導致 `logits[-1]` 預測的是
EOS 之後的 token（只學過 PAD），而非真正的下一個製程步驟。

| Exp | WandB run | step 100 Top-1 | step 200 Top-1 | Top-3 | MRR | 趨勢 |
|---|---|:---:|:---:|:---:|:---:|---|
| A 隨機 + fam | `quick_A_rand\|fam` | 0.750 | **0.787** | 0.970 | 0.883 | 200 步最佳；step 100 後趨平 |
| B 描述 + fam | `quick_B_desc\|fam\|desc` | 0.707 | 0.753 | 0.960 | 0.862 | 仍在上升；−3.4% vs A |
| C 描述+cat + fam | `quick_C_categ\|fam\|desc\|cat` | 0.687 | 0.760 | 0.967 | 0.866 | 成長幅度最大（+7.3pp）；−2.7% vs A |

**200 步排名：A > C > B。** C 的 step 100→200 漲幅最陡，與 `cat_embed` zero-initialized
（初始 no-op，逐步學習）一致。最終排名需要在完整 6-layer/256-dim 模型跑 3,000 步後才能確定，
留待 Leonardo。

參考基準：`gpt2_mac_local`（6L/256d，10k 序列，無 family token）在 MPS 上 step 200 達到
Top-1=0.810，確認完整尺寸模型遠優於 smoke test toy 模型。Leonardo Exp A（相同完整模型 + family token）
預期 Top-1 > 0.810。

---

## 後果

- 三個 job 可以並行提交，互不依賴
- Gate 2 評估使用獲勝實驗的 checkpoint
- 若三組實驗 Top-1 均低於 0.40，下一步是資料擴增或調低 LR，再考慮擴大模型
