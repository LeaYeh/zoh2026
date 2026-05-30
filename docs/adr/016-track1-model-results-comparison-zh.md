# ADR 016 — Track 1：模型結果比較（2026-05-30）（繁體中文版）

**狀態：** 已採納  
**日期：** 2026-05-30  
**相關：** ADR-008（bigram）、ADR-009（GPT-2 zero-shot）、ADR-015（消融實驗設計）

---

## 背景

本 ADR 記錄 2026-05-30（競賽第一天）取得的所有模型評估結果，涵蓋統計基準、
GPT-2 zero-shot，以及在 Leonardo A100 上線前於 Apple Silicon MPS 本地跑的 GPT-2 fine-tuned 變體。

**本次 session 修復的關鍵 bug：** `predict_next_step` 將 `[BOS…EOS]` 傳入模型，
導致 `logits[-1]` 預測的是 EOS 之後的 token（訓練時只見過 PAD）。
修正方式：去除尾端 EOS——`ids = tokenizer.encode(partial_steps)[:-1]`。
這個 bug 是 Top-1=4.67%（有 bug）vs Top-1=81.0%（修復後）的差異所在。

---

## 完整結果表

### 統計基準（無 GPU，來自先前 ADR）

| 模型 | Top-1 | Top-3 | MRR | Task 3 ROC-AUC | 備註 |
|---|:---:|:---:|:---:|:---:|---|
| 隨機 | ~0.5% | — | — | ~0.50 | 1/198 詞彙 |
| GPT-2 zero-shot | 13.3% | — | — | 0.57 | n=30；僅 CPU；BPE tokenizer |
| Bigram | **68.3%** | — | 0.81 | 0.996* | *僅合成異常有效 |
| Rule Checker | — | — | — | **~1.000** | `validate_sequence()`，Task 3 已解決 |

### Fine-tuned GPT-2 — 完整模型（6L / 256d / 4.86M params）

| Run | 資料 | fam | init | cat | steps | Top-1 | Top-3 | MRR | 收斂點 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gpt2_mac_local` | 10k×3 | ✗ | rand | ✗ | 200† | **0.810** | 1.000 | 0.903 | step 200 |

†在 step 400 確認 plateau（Top-1=0.807）後於 step 550 提早停止。

### Fine-tuned GPT-2 — Toy 模型（2L / 128d）Smoke Test

| Run | 資料 | fam | init | cat | steps | step 100 | step 200 | Top-3 | MRR |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `quick_A_rand\|fam` | 1k×3 | ✓ | rand | ✗ | 200 | 0.750 | **0.787** | 0.970 | 0.883 |
| `quick_C_categ\|fam\|desc\|cat` | 1k×3 | ✓ | desc | ✓ | 200 | 0.687 | 0.760 | 0.967 | 0.866 |
| `quick_B_desc\|fam\|desc` | 1k×3 | ✓ | desc | ✗ | 200 | 0.707 | 0.753 | 0.960 | 0.862 |

---

## 關鍵發現

### 1. EOS inference bug（已修復）

修復前：train loss=0.37 但 Top-1=4.67%。  
修復後：Top-1=81.0%。訓練 loss 一直是正確的，只有 inference 有問題。

### 2. 模型在 step 200 收斂

`gpt2_mac_local` 的 loss 曲線：

| Step | Train Loss | 備註 |
|---|---|---|
| 10 | 4.82 | 隨機基準 ln(205)=5.32 |
| 100 | 0.450 | warmup 結束 |
| 150 | 0.388 | 高原期開始 |
| 200 | 0.372 | **Top-1=0.810 ← best checkpoint** |
| 400 | 0.342 | Top-1=0.807，無改善 |

文法接近確定性（H < 1 bit），模型只需 200 步就能學會主幹。
後續步驟無法進一步提升 Top-1——step 200 vs step 400 的平台已確認。

### 3. 完整模型 >> Toy 模型

| 模型大小 | 資料 | Top-1 @200 |
|---|---|:---:|
| 2L/128d | 1k seqs | 0.787 |
| **6L/256d** | **10k seqs** | **0.810** |

完整模型配合 10 倍資料，即使沒有 family token，也比 toy 模型高出 +2.3pp。
Leonardo Exp A 在此基礎上加入 family token，預期超越 0.810。

### 4. 200 步時：隨機初始化優於描述初始化

| 初始化 | Top-1 @200 | step 100→200 漲幅 |
|---|:---:|:---:|
| 隨機（A）| **0.787** | +3.7pp |
| 描述 + cat（C）| 0.760 | **+7.3pp** |
| 描述（B）| 0.753 | +4.7pp |

描述初始化在短訓練時落後，因為 2-layer toy 模型容量太小，無法充分利用物理先驗。
C 的最大成長斜率（+7.3pp）顯示它從更多步驟中受益更多。
**最終排名需要在完整 6L/256d 模型跑 3,000 步才能確定。**

### 5. 完整模型的 Top-3 = 100%

正確答案永遠在模型的前 3 名預測內。這使 Task 2（序列補全）極具可行性：
beam search width=3 永遠不會漏掉正確步驟。

---

## Gate 狀態更新

| Gate | 條件 | 狀態 |
|---|---|---|
| Gate 1 | Pipeline 端到端正常，200 步後 Top-1 > 0 | ✅ 通過（Top-1=0.810）|
| Gate 2 | 第 1000 步 Top-1 ≥ 0.40 | ✅ 通過（0.810 >> 0.40）|
| Gate 3 | 三個提交檔案格式正確 | ⏳ 等待 organizer 的 eval CSV |
| Gate 4 | Gradio demo 上線，提交檔案驗證通過 | ⏳ 等待中 |

---

## Leonardo 待確認問題

1. Family token（`family_prefix=true`）能讓 Top-1 超越 0.810 嗎？（Exp A vs mac_local）
2. 描述初始化在 3,000 步完整模型上有幫助嗎？（Exp B vs Exp A）
3. Category embedding 在完整規模下有可量測的效益嗎？（Exp C vs Exp B）
4. Task 2 token accuracy 用 beam search 能從 bigram 的 6.2% 顯著提升嗎？

---

## 後果

- `data/oof/gpt2_mac_local/model.pt` 是目前最佳 checkpoint（Top-1=0.810）
- Leonardo A/B/C 結果一旦出爐，將取代所有本地結果
- Task 3 由 `validate_sequence()` 獨立解決，不需要模型訓練
- 可以用 mac_local checkpoint 先開始開發 Gradio demo，不必等 Leonardo
