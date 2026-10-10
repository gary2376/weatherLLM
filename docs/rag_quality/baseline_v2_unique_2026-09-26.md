# RAG 品質 Baseline v2：唯一指向性問題（2026-09-26）

## 結果摘要

- Eval set：66 筆
- 有效評分：66 筆
- 查無 context：0 筆
- Faithfulness 平均：0.8765
- Context Recall 平均：0.8889
- Answer Relevancy 平均：0.4963

完整逐筆結果位於 `code/rag_quality/results_v2_20260926.json`。

## 方法

v2 沒有重新抽樣文件。`code/rag_quality/build_eval_set_v2.py` 直接讀取 v1 的 `eval_set.json`，完整沿用同一批 66 筆 `id`、`reference` 與 `metadata`；唯一變因是問題設計，因此 v1／v2 的差異可用來觀察題型對指標的影響。

每筆資料仍使用 `gpt-4o-mini` 生成一個自然語言問題，但 v2 要求問題包含至少兩項可鎖定該文件的獨特細節，例如年份、距離、特殊活動、價格、認證、歷史事實、設施或服務，同時禁止直接出現店家／景點／住宿完整名稱，避免退化成名稱字串比對。輸出固定保存於 `code/rag_quality/eval_set_v2.json`，後續不重新生成。

評估流程與 v1 相同：現有 `code/rag.py::query()` 取 Top-5 context，`gpt-4o-mini` 根據 context 生成答案，再由 ragas 計算 Faithfulness、Context Recall、Answer Relevancy。所有 LLM 與 embedding 呼叫均使用 `DiskCacheBackend`。

## 與 v1 對照

兩組測試使用完全相同的 66 筆文件與相同評估流程，只改變問題寫法。

| 指標 | v1 泛用查詢 | v2 唯一指向 | v2 − v1 |
|---|---:|---:|---:|
| Faithfulness | 0.8906 | 0.8765 | -0.0141 |
| Context Recall | 0.5439 | 0.8889 | +0.3450 |
| Answer Relevancy | 0.5266 | 0.4963 | -0.0304 |
| 查無 context | 0 | 0 | 0 |

Context Recall 提升 0.3450，支持 v1 的主要限制確實來自「泛用分類問句＋單一 ground truth」不相容。v2 問題加入唯一辨識細節後，66 筆中有 53 筆 Context Recall 為 1.0，只有 4 筆為 0.0，較適合衡量目標文件是否被實際取回。

Faithfulness 小幅下降 0.0141，Answer Relevancy 小幅下降 0.0304；這兩項沒有因 Recall 提升而同步上升。Faithfulness 衡量生成答案是否受 context 支持，Answer Relevancy 衡量答案與問題的語意對齊，兩者都不是純檢索命中指標。因此 v2 的「RAG 真實檢索精準度」主要應從 Context Recall 解讀，不能把三個分數合併成單一精準度。

## 同一文件的問題範例

### 南投集集觀光工廠

- v1：「南投縣有什麼適合親子互動和生態教育的觀光工廠嗎？」
- v2：「在南投縣集集鎮，有哪個觀光工廠結合了超過100種香蕉品種的生態園區和提供DIY課程的親子遊樂區，並且所有產品均使用100%天然原料？」

v2 加入香蕉品種數、DIY 親子區與天然原料三項細節，但沒有直接寫出名稱。

### 屏東恆春民宿

- v1：「找屏東恆春的舒適民宿，想體驗當地慢活氛圍。」
- v2：「請問在屏東縣恆春鎮，有哪家民宿結合了電影『海角七號』的拍攝場景，並提供機車租借服務，讓遊客能輕鬆探索恆春半島的美景？」

同一 reference 的 Context Recall 從 v1 的 0.0 提升至 v2 的 0.6。

### 臺中北屯登山步道

- v1：「請推薦臺中市北屯區的登山步道或健身步道。」
- v2：「請問在臺中市北屯區，有哪條登山步道結合了遊戲元素，適合健身活動？」

同一 reference 的 Context Recall 從 v1 的 0.0 提升至 v2 的 1.0；Answer Relevancy 也從 0.0 提升至 0.4857。

### 高雄桃源高山湖泊

- v1：「高雄桃源區有沒有好看的高山湖泊？」
- v2：「在高雄市桃源區，海拔2200公尺的高山湖泊被稱為心湖，這裡終年不乾且深約2公尺，經常雲霧繚繞，湖面如明鏡般映照周圍的山林，請問這裡的具體位置距離南橫公路約多少公里？」

v2 以海拔、別稱、深度與地景特徵鎖定同一景點，仍未直接出現景點名稱。

## 定位與後續使用

- **v2 唯一指向性測試**：作為衡量目標文件檢索精準度的主要 baseline；未來 RAG 核心有變更時，使用固定的 `eval_set_v2.json` 做同條件比較。
- **v1 泛用查詢壓力測試**：繼續保留，用於 retrospective regression，觀察系統面對開放式推薦問句時是否退步；其單一 ground truth 限制仍須搭配人工解讀。

兩者用途不同，結果不可互相覆蓋。v2 仍有少數 reference 資訊極少，能寫入問題的唯一辨識細節有限；因此它比 v1 更接近檢索精準度評估，但不是沒有測試集天花板。

