# weatherLLM

天氣感知的台灣旅遊助理：使用者用自然語言提問（例如「南投日月潭有什麼景點？」「台北市現在有沒有生效中的天氣特報？」），系統用一個 ReAct agent 判斷要查旅遊資料（RAG）還是查天氣（MCP），整合後回答，並在地圖上標出推薦的地點。

- **旅遊資料**：觀光署景點／住宿／餐飲開放資料，存進 SQLite，再建 Chroma 向量索引與 BM25 索引，用 Hybrid Search + Reranker 檢索。
- **天氣資料**：透過 `cwa-mcp-server`（中央氣象署開放資料的 MCP 工具，PyPI 套件）即時查詢。
- **Agent**：LangGraph `create_react_agent`，單一 agent、5 個工具（1 個 RAG + 4 個天氣）。
- **介面**：Streamlit，對話與地圖並排。
- **工程化**：Docker 單一 container、GitHub Actions CI（pytest + image build）與 CD（pytest 通過後推到 GHCR）。
- **範圍**：本機執行 + Docker + CI/CD。**不做公開部署**（2026-11-05 決定）。

完整的設計原因、踩過的坑與決策記錄在 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)；面試用的問答在 [`docs/interview_talking_points.md`](docs/interview_talking_points.md)；名詞對照在 [`docs/GLOSSARY.md`](docs/GLOSSARY.md)。

## 整體資料流

```
【離線：建立資料，只在更新資料時手動跑】
  觀光署資料集(7777/7780/7779) ─┐
  TDX公車/YouBike/停車場(選用) ─┼─> collector/fetch_*.py ─> Data/travel.db (SQLite)
                                 │         └─> link_*.py、derive_district_coords.py(衍生欄位/關聯表)
                                 └─> collector/build_embeddings.py ─> Data/chroma/ + Data/bm25_index.pkl

【線上：每次提問】
  使用者 ─> Streamlit(code/app.py) ─> WeatherAgent(code/agent.py, ReAct, gpt-4o-mini)
              ├─ 旅遊問題 ─> search_travel_info ─> rag.py ─> hybrid_search.py(BM25+向量+RRF) ─> reranker_fn.py ─> Top-5
              ├─ 天氣問題 ─> mcp_tools.py ─> cwa-mcp-server(子程序，stdio) ─> 中央氣象署API (含45分鐘TTL快取)
              └─ 防線：recursion_limit=10、單thread累計成本上限US$0.08、RAG失敗硬覆寫、prompt injection防護
            回答 + 命中的地點doc_ids ─> 對話區顯示、地圖標示
```

## 目錄

| 路徑 | 內容 |
|---|---|
| `code/` | 線上執行的程式：`app.py`（UI）、`agent.py`、`mcp_tools.py`、`rag.py`、`hybrid_search.py`、`reranker_fn.py`，以及 pytest 測試與 smoke test |
| `code/rag_quality/` | RAG 評估集建立與執行（ragas） |
| `collector/` | 離線資料收集與索引建置腳本 |
| `Data/` | 產出的資料（SQLite、Chroma、BM25、模型快取）。產出物不進版控，需自己跑 collector 產生 |
| `docs/` | 架構文件、面試問答、流程圖、RAG 評估基準 |
| `Dockerfile`、`docker-compose.yml`、`warmup_fastembed.py`、`.dockerignore` | 容器化 |
| `.github/workflows/` | `ci.yml`、`cd.yml` |

## 環境變數（放在專案根目錄的 `.env`，範本見 `.env.example`）

| 變數 | 誰提供 | 用途 | 沒有會怎樣 |
|---|---|---|---|
| `CWA_API_KEY` | 操作者（免費申請） | 傳給 `cwa-mcp-server` 查天氣 | agent 建構時啟動天氣子程序失敗，網頁顯示通用錯誤訊息 |
| `OPENAI_API_KEY` | 網頁使用者在側邊欄自己貼 | LLM 呼叫（`gpt-4o-mini`） | 網頁不讓 agent 執行。命令列的 smoke test 與 eval 才從環境變數讀 |
| `TDX_CLIENT_ID`、`TDX_CLIENT_SECRET` | 操作者（免費會員） | 只有 `collector/fetch_transit.py` 用 | 不跑交通資料收集就不需要 |
| `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST` | 操作者（選用） | 可觀測性 trace | 沒設定就不送 trace，不影響功能 |

`.env` 已被 `.gitignore` 與 `.dockerignore` 排除，不會進 git 或 image。

## 從零開始跑起來

需求：Python 3.11 以上（專案用 3.11.15；`cwa-mcp-server` 要求 Python ≥ 3.11）。以下指令都在專案根目錄執行。

```bash
# 1. 建環境、裝套件（requirements.txt 全部釘版本）
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 設定金鑰
cp .env.example .env      # 然後編輯 .env 填入 CWA_API_KEY 等

# 3. 建立資料（Data/ 裡的資料庫與索引不在版控裡，必須自己產生；順序有依賴）
python3 collector/fetch_tourism.py              # 景點/住宿/餐飲 -> Data/travel.db（免key，每日更新）
python3 collector/derive_district_coords.py     # 行政區中心點（依賴上一步的資料）
python3 collector/build_embeddings.py           # 全量重建 Chroma + BM25 索引（會下載本地模型，需幾分鐘）

# 選用（目前UI/RAG用不到，是資料層保留的內容）：交通站與關聯表
python3 collector/fetch_transit.py              # 需要 TDX_CLIENT_ID/SECRET；內建每13秒節流，耗時長
python3 collector/link_transit.py               # 注意：會刪除沒被任何景點/住宿引用的交通站
python3 collector/link_lodging_attraction.py

# 4. 驗證資料與檢索
python3 code/smoke_test_rag.py                  # 22縣市 + 邊界案例；失敗時 exit code 非0

# 5. 啟動介面
streamlit run code/app.py                       # 開 http://localhost:8501，在側邊欄貼自己的 OpenAI key
```

## 測試

```bash
(cd code && pytest -q)         # 35 個測試，全部 mock，不打真實 API、不需金鑰與 Data/
python3 code/smoke_test_agent.py   # 真實 OpenAI + CWA（會花錢）：多輪對話、多需求、guardrail、天氣工具
python3 code/smoke_test_rag.py     # 需要建好的 Data/（見上）
python3 code/rag_quality/run_eval.py --eval-set eval_set_v2.json   # RAG 評估（ragas，會呼叫 OpenAI）
```

CI 只跑 `pytest -q`（35 個）與 Docker image build；smoke test 與 RAG eval 需要真實金鑰或本機資料，不在 CI 裡。各測試的涵蓋範圍見 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 第 8 節。

## Docker

```bash
docker compose up --build       # 以 Dockerfile 現場 build 並啟動，http://localhost:8501
docker compose pull             # 或拉 CD 推到 GHCR 的 image（ghcr.io/gary2376/weatherllm:latest）
```

重點：image 裡**沒有** `Data/`；`docker-compose.yml` 會把主機的 `Data/chroma`、`Data/travel.db`、`Data/bm25_index.pkl` 掛進 container，所以**要先在主機完成上面第 3 步**。這三個路徑在主機上不存在時，Docker 會建出空目錄頂替，container 看起來正常、查詢卻查不到資料。在 Mac 用 colima 時，專案若放在外接磁碟，需 `colima start --mount /Volumes/<磁碟名>:w`。細節見 `docs/ARCHITECTURE.md` 第 6 節。

## 文件地圖

| 想了解 | 讀這個 |
|---|---|
| 設計原因、踩過的坑、決策記錄 | `docs/ARCHITECTURE.md` |
| 面試怎麼講 | `docs/interview_talking_points.md` |
| 名詞（RAG、RRF、MCP、ReAct…） | `docs/GLOSSARY.md` |
| 請求流程圖 | `docs/diagrams/agent-workflow.html` |
| RAG 評估結果 | `docs/rag_quality/` |
| 資料來源與授權 | `DATA_LICENSE.md` |
