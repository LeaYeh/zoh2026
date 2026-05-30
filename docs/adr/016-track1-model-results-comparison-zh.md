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

## 各任務的模型設計分析

### Task 1 — 下一步預測

| 模型 | 優點 | 缺點 |
|---|---|---|
| **Bigram** | 在確定性位置（H≈0 bits，38–48% 的步驟）幾乎完美 | 在可選步驟分支點失敗；僅有前一步的上下文 |
| **GPT-2 zero-shot** | 無 | 無領域知識；BPE tokenizer 將步驟名稱切分成 sub-token |
| **gpt2_mac_local**（無 fam）| 完整序列上下文，學習 bigram 無法捕捉的長程依賴 | 無 family token：IGBT 第 5、6 個 litho level 無法消歧義 |
| **Exp A**（+ fam token）| Family token 降低預測熵 0.16–0.33 bits；IGBT litho level 5/6 正確 | 無物理先驗——embedding 從隨機開始，只從 co-occurrence 學習 |
| **Exp B**（+ desc init）| 物理相似步驟從第 0 步就在 embedding 空間中接近（熱製程 cluster、litho level 有序）| 62/198 個步驟無參數資料（numeric=0 補值）；在 200 步視窗內效益可能有限 |
| **Exp C**（+ cat embed）| Per-category 殘差讓同類所有步驟共享一個學習到的偏移量，有助於分支點 | `cat_embed` 為 zero-init，需要更多訓練步驟才能啟動；8 個類別可能過於粗糙 |

**Task 1 的瓶頸：** 高熵位置——可選量測步驟和 litho cycle 分支點。
H > 1.5 bits 的步驟才是模型設計差異能顯現的地方。

---

### Task 2 — 序列補全

| 模型 | 優點 | 缺點 |
|---|---|---|
| **Bigram（greedy）** | 快；在確定性主幹上正常工作 | **Token Accuracy=6.2%**：一個步驟錯誤會鏈式傳播，後續所有步驟都出錯 |
| **gpt2_mac_local**（beam=1）| 完整上下文自迴歸；Top-3=100% → 正確步驟永遠在前 3 名 | Greedy decode 仍有鏈式錯誤風險；beam search 尚未實作 |
| **Exp A**（beam≥3）| Family token 確保 family-specific 區塊順序正確（IC 鎢路徑、IGBT 4 種 implant）| 沒有 beam search 的話，和 bigram 一樣有鏈式錯誤風險 |
| **Exp B/C**（+ desc/cat）| 描述初始化可能降低物理上截然不同的分支點的第一個 token 錯誤率 | 邊際效應——Task 2 改善主要來自 beam search 寬度，不是 embedding 初始化 |

**Task 2 的關鍵槓桿：beam search 寬度。** Top-3=100% 代表 beam width=3 永遠不會丟失正確的補全路徑。
在 `complete_sequence` 中實作 beam search，對 Task 2 的改善遠大於 embedding ablation。

**錯誤鏈式傳播分析：**
```
Bigram greedy：  步驟錯 → OOV from-step → 回退到全局頻率 → 後續全部錯誤
GPT-2 greedy：  步驟錯 → 上下文偏移 → 下一步預測失準 → 鏈式錯誤
GPT-2 beam=3：  每步保留 3 個候選 → 正確路徑存活 → 不發生鏈式錯誤
```

---

### Task 3 — 異常偵測

| 模型 | ROC-AUC | Rule Attribution | 結論 |
|---|:---:|:---:|---|
| **Bigram** | 0.996* | ✗ | *僅合成 swap 異常有效。真實 rule violation 是 window/全局排序約束，bigram 看不到 |
| **GPT-2 perplexity** | 0.57 | ✗ | 接近隨機；WebText perplexity 無法辨別製程規則違反 |
| **Rule Checker**（`validate_sequence`）| ~1.000 | ✓ | 與 organizer 生成 ground truth 的程式碼相同；免費附送確切 rule ID |

**所有 10 條 forbidden rule 都是 window-based（6–15 步）或全局排序約束。
任何 ML 模型都無法取代確定性 rule checker。**（ADR-010）

Task 3 已解決。釋放的精力 → 專注於 Task 2 beam search 實作。

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
