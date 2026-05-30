# ADR 012 — Track 1：EDA 發現總結與架構基準（繁體中文版）

**狀態：** 已採納  
**日期：** 2026-05-30

## 背景

完成 EDA（`notebooks/eda/track1_eda.ipynb`）並建立統計基準（ADR-008、ADR-009、ADR-010、
ADR-011）後，需要一份合併參考文件，彙整所有發現及其對 Leonardo 上 GPT-2 fine-tuning 的直接影響。

本 ADR 不取代任何先前 ADR，僅作為競賽當天的綜合參考紀錄。

---

## 語料庫統計

| 指標 | 數值 |
|---|---|
| 總序列數 | 3,000 筆（每個 family 各 1,000）|
| 總步驟 token 數 | 388,294 |
| 完整詞彙 | **198 個唯一步驟名稱** |
| 跨三 family 共享 | **94 步（47%）**|
| Family 專屬 | 76 步（MOSFET 20 / IGBT 27 / IC 29）|

**序列長度：**

| Family | 最短 | 最長 | 平均 |
|---|---|---|---|
| IC | 107 | 122 | 115 |
| MOSFET | 117 | 134 | 125 |
| IGBT | 139 | 155 | 148 |

GPT-2 預設 context window（1024 tokens）可覆蓋最長序列（~160 tokens），有 6 倍餘裕。
無需 sliding window 或截斷。

---

## 製程文法

**條件熵 H(next step | current step) = 0.89–0.95 bits（per family）**

- 38–48% 的步驟只有一個合法後繼（熵 = 0 bits）。
- 所有序列 100% 以 `RECEIVE WAFER LOT` 開頭。
- 高熵步驟僅出現在可選量測步驟和 litho cycle 的分支點。

意涵：文法從 1,000 筆序列即可學會；模型需要捕捉的是**可選步驟的組合變異**，
而非從雜訊中發現隱藏文法。

---

## 三個 Family 的結構差異

**共同骨架（~16 個 anchor 步驟，所有 family 100% 出現）：**

```
RECEIVE WAFER LOT → LOT IDENTIFICATION → ... → HF DIP
→ ILD block（5 個 synonym-pair 步驟）
→ CURE PASSIVATION → CLEAN PAD OPENING → MEASURE PAD OPENING
→ LEAKAGE TEST → SWITCHING TEST → WAFER SORT TEST → YIELD ANALYSIS → SHIP LOT
```

**關鍵分歧點：**

| 模組 | MOSFET | IGBT | IC |
|---|---|---|---|
| Family 準備 | 磊晶沉積 | 磊晶晶圓檢查 | 背面研磨＋濕式蝕刻 |
| 微影層數 | **4 層** | **6 層** | **4 層** |
| 離子佈植 | WELL / LDD / SOURCE DRAIN | P BODY / N BUFFER / CHANNEL STOP / DRAIN-CATHODE / SOURCE | 僅 N-TYPE |
| 通孔填充 | FILL VIA METAL | FILL VIA METAL | 鎢晶種層＋FILL VIA TUNGSTEN |
| 最終測試 | 臨界電壓測試 | 崩潰電壓測試 | 臨界電壓測試 |

IGBT 的 6 個 litho level vs MOSFET/IC 的 4 個，是 `[FAMILY]` BOS conditioning token
最關鍵的使用場景——沒有此 token，模型無法判斷 `ALIGN MASK LEVEL 5`、`LEVEL 6` 是否合法。

---

## 效能基準階梯

| 方法 | Task 1 Top-1 | Task 2 Token Acc | Task 3 ROC-AUC | 備註 |
|---|---|---|---|---|
| 隨機基準 | ~0.5% | — | ~0.50 | 1/198 詞彙 |
| GPT-2 zero-shot | 13.3% | 略過 | 0.57 | n=30；僅 CPU |
| Bigram 模型 | **68.3%** | **6.2%** | 0.996* | *僅合成異常有效 |
| GPT-2 fine-tuned | 目標 | **最大改善空間** | N/A | Task 3 由 Rule Checker 負責 |

**Task 2 是主要的改善目標。** Bigram 的 6.2% token accuracy 源於貪心解碼的錯誤累積：
一個步驟預測錯誤，後續所有步驟都受影響。Fine-tuned GPT-2 搭配 beam search 可打破此鏈式錯誤。

---

## Task 3 — 確定性 Rule Checker（已解決）

所有 10 條 forbidden rule 都是 window-based（回看 6–15 步）或全局排序約束，
沒有一條是 bigram 可偵測的。`scripts/generate_sequences.py` 中的 `validate_sequence()`
涵蓋全部 10 條規則，並直接回傳違反的 rule ID。

**Task 3 在任何 GPU 訓練之前就已解決。** 詳見 ADR-010。

---

## 參數 EDA（ADR-011 摘要）

- 353/374 個步驟（94–95%/family）有至少一個可解析的數值參數。
- 73 個步驟被三個 family 共享。
- 最具辨識力的參數（PCA PC1 loading 排序）：
  `temperature_c` > `time_min` > `thickness_nm` > `pressure_mtorr`
- 跨 family 的溫度差異小（≤50°C），集中在熱製程步驟。
- PCA（2 個主成分，解釋 49% 變異）顯示三 family 在 parameter space 部分可分，但重疊顯著。
- **參數是補充訊號。序列文法（bigram 轉移）才是 family 識別的主要依據。**

最強的基於參數的 family 辨識訊號：
- Litho overlay tolerance：IGBT ±120–150 nm vs MOSFET/IC ±80–100 nm
- RTA 溫度：IGBT 1050°C vs MOSFET/IC 1000°C
- 離子佈植能量分佈：IGBT 有 5 種不同佈植（30–150 keV）；MOSFET/IC 各有 3 種

---

## 架構決策

| 決策 | 選擇 | 理由 |
|---|---|---|
| 模型數量 | 單一共享模型 | 47% 詞彙共享；骨架三 family 相同 |
| Family 訊號 | `[FAMILY]` BOS conditioning token | IGBT litho level 消歧義必要 |
| Tokenization | 一個步驟 = 一個 custom token | 保留步驟名語意；不使用 subword 切割（Rule 2）|
| 訓練格式 | `[MOSFET] STEP1 \| STEP2 \| ... [EOS]` | pipe 分隔；每序列最多 ~160 tokens |
| Task 3 解法 | 確定性 rule checker | 統計模型無法表達全局排序約束 |
| 計算資源 | Fine-tuning 在 Leonardo A100 | 本地 CPU 每個預測點需 11 秒，訓練不可行 |

---

## ADR 建立時的 Gate 狀態

| Gate | 條件 | 狀態 |
|---|---|---|
| Gate 1 | GPT-2 pipeline 端到端正常，200 步後 Top-1 > 0 | 等待 Leonardo |
| Gate 2 | 第 1000 步 Top-1 ≥ 0.40 | 等待中 |
| Gate 3 | 三個提交檔案格式正確 | 等待中 |
| Gate 4 | Gradio demo 上線，提交檔案驗證通過 | 等待中 |

## 後果

- 本文件是競賽當天架構討論的單一參考來源。
- Task 2 token accuracy（從 6.2% 提升至 ≥40%）是 GPT-2 fine-tuning 的主要目標。
- 參數特徵（`temperature_c`、`time_min`、`thickness_nm`）保留作為 Gate 2 通過後
  潛在 OOD conditioning extension 的輸入。
- 在 Gate 1 確認 pipeline 健康之前，不應開始任何新架構實驗。
