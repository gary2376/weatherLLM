# 名詞對照

專案文件與面試問答裡常出現的名詞，每一條都對應到這個專案裡實際用到的地方。

## 檢索（RAG）

| 名詞 | 白話說明 | 在本專案 |
|---|---|---|
| RAG（Retrieval-Augmented Generation） | 先從自己的資料庫檢索相關內容，再讓 LLM 根據檢索到的內容回答，而不是只靠模型記憶 | 景點/住宿/餐廳資料由 `code/rag.py` 檢索後交給 agent |
| Embedding（向量嵌入） | 把一段文字轉成一串數字，語意相近的文字數字也相近 | `BAAI/bge-small-zh-v1.5`，用 `fastembed` 在本機 CPU 計算，免費 |
| Chroma | 儲存向量並做相似度搜尋的資料庫 | `Data/chroma/`，collection 名稱 `travel` |
| BM25 | 以關鍵字出現頻率計分的傳統檢索演算法，擅長精確字面比對 | 用 `jieba` 斷詞 + `rank_bm25`，索引存 `Data/bm25_index.pkl` |
| Hybrid Search | 同時用向量檢索（懂語意）與 BM25（懂字面），再合併排序 | `code/hybrid_search.py`，兩邊各取 Top-50 |
| RRF（Reciprocal Rank Fusion） | 合併多份排名的方法：每份排名給 `1/(k+名次)` 分後加總 | `RRF_K = 60`（業界常用預設，沒有特別調過） |
| Reranker（重排序） | 用較精細但較慢的模型，對候選結果逐筆重新打分排序 | `BAAI/bge-reranker-base`，從候選中選出 Top-5 |
| Metadata 過濾 | 檢索前先用結構化欄位（如縣市）縮小範圍 | query 裡偵測到縣市名，就只在該縣市內檢索（`detect_cities()`） |
| doc_id / artifact | 每筆資料的識別碼；工具回傳時附帶的「給程式用」的資料 | `search_travel_info` 用 `content_and_artifact` 回傳命中的 doc_id，地圖據此標示 |

## Agent

| 名詞 | 白話說明 | 在本專案 |
|---|---|---|
| Agent | 能自己決定要呼叫哪些工具、呼叫幾次的 LLM 應用 | `code/agent.py` 的 `WeatherAgent` |
| ReAct | Reasoning + Acting：模型輪流「思考」與「呼叫工具」，直到能回答 | LangGraph `create_react_agent` |
| LangGraph | 把 agent 流程寫成節點與狀態的框架 | 提供 ReAct agent、`recursion_limit`、checkpointer |
| MemorySaver / checkpointer | 儲存對話狀態讓多輪對話能接續 | 純記憶體，以 `thread_id` 區分對話；不做跨重啟持久化 |
| `recursion_limit` | LangGraph 單次執行的步數上限，防止無限迴圈 | `RECURSION_LIMIT = 10` |
| Guardrail（防線） | 限制 agent 行為、避免失控的機制 | 步數上限、單 thread 累計成本上限 US$0.08、RAG 失敗硬覆寫、prompt injection 防護 |
| Prompt Injection | 在輸入或檢索內容裡藏指令，誘使 LLM 偏離原本任務 | `SYSTEM_PROMPT` + 檢索內容加「不可信資料」邊界標記 |
| Faithfulness / 防幻覺 | 回答是否完全受檢索內容支持，沒有憑空編造 | RAG 工具失敗時，程式碼強制回固定訊息，不採用後續 LLM 訊息 |

## MCP 與工具

| 名詞 | 白話說明 | 在本專案 |
|---|---|---|
| MCP（Model Context Protocol） | 讓 LLM 應用以統一協定呼叫外部工具的標準 | 天氣工具由 `cwa-mcp-server` 提供 |
| stdio | 用標準輸入輸出和子程序溝通 | agent 啟動 `cwa-mcp-server` 子程序，透過 stdio 連線 |
| `isError` | MCP 回傳結果中標示「這是錯誤」的旗標 | `mcp_tools.py` 檢查此旗標，錯誤不快取、不當成功回傳 |
| TTL cache | 資料保留一段時間後失效的快取 | 天氣工具結果快取 45 分鐘（`WEATHER_CACHE_TTL_SECONDS`） |
| in-flight 去重 | 同一請求同時多筆進來時只真正執行一次 | 併發的相同天氣查詢只打一次 API |
| `lru_cache` | Python 標準庫的「最近最少使用」快取 | 包住 `hybrid_search.search()`，`maxsize=256`，key 是原始查詢字串 |

## 評估與觀測

| 名詞 | 白話說明 | 在本專案 |
|---|---|---|
| ragas | 評估 RAG 品質的套件 | `code/rag_quality/` 用它算三個指標 |
| Context Recall | 正確答案所需資訊有多少被檢索到 | v1 0.5439、v2 0.8889 |
| Faithfulness | 回答被檢索內容支持的程度 | v1 0.8906、v2 0.8765 |
| Answer Relevancy | 回答與問題的語意對齊程度 | v1 0.5266、v2 0.4963 |
| Langfuse | LLM 應用的追蹤與觀測服務 | 選用；斷線時停用 trace 並重試一次 |
| smoke test | 用真實服務做的快速驗收腳本 | `smoke_test_rag.py`、`smoke_test_agent.py`，不在 CI 裡 |

## 工程化

| 名詞 | 白話說明 | 在本專案 |
|---|---|---|
| CI（持續整合） | 每次推程式碼自動測試與驗證 | `.github/workflows/ci.yml`：pytest + docker build（不推送） |
| CD（持續交付） | 通過後自動把成品發布出去 | `.github/workflows/cd.yml`：pytest 通過後 build 並推到 GHCR |
| GHCR | GitHub 的 container image 倉庫 | `ghcr.io/gary2376/weatherllm`，tag 為 `latest` 與 commit SHA |
| `GITHUB_TOKEN` | GitHub 每次執行自動發的臨時憑證 | CD 用它登入 GHCR，權限只開 `contents: read`、`packages: write` |
| Volume 掛載 / bind mount | 把主機的檔案或資料夾掛進 container | `Data/chroma`、`Data/travel.db`、`Data/bm25_index.pkl` |
| HEALTHCHECK | container 定期自我檢查 | 每 30 秒打 `/_stcore/health`，只證明 Streamlit 還活著，不代表資料正確 |
| BYOK（Bring Your Own Key） | 使用者自己提供 API key | OpenAI key 由網頁使用者輸入，只存在自己的 session |
| `.env` / `env_file` | 以檔案提供環境變數 | 操作者的 `CWA_API_KEY`、`LANGFUSE_*`；不進 image、不進 git |
