# weatherLLM — 全台天氣＋旅遊建議系統

Streamlit 打造的天氣查詢＋景點/住宿推薦系統，資料涵蓋全台 22 縣市（不是只有台中）。開發過程與每個階段的決策記錄在 [ROADMAP.md](ROADMAP.md)。

**目前狀態：尚未正式公開上線**，開發環境已可跑（見下方「本機執行」），Streamlit Community Cloud 部署與上線前檢查清單還在 ROADMAP.md 的 Phase 0/1/4 追蹤中，尚未完成。

## 系統長什麼樣

```
使用者
  │
  ▼
streamlit_app.py::main()  ── Streamlit UI（地圖／CCTV圖釘／聊天框）
  │
  ▼
orchestrator.py（LangGraph）
  ├─ lightweight_node → WeatherChatbot（關鍵字比對，免費，涵蓋台中/台北/新竹/台南/高雄5大城市＋部分行政區）
  ├─ supervisor_node  → 判斷要不要走進階路徑（天氣關鍵字觸發）
  └─ advanced_node    → system.py::get_weather_data()（需使用者自己貼 OpenAI key）
                            └─ weather_graph.py（LangGraph）
                                 intent_node（LLM 解析問句）
                                 → weather_node（Open-Meteo 即時/預報）
                                 → attraction_node / lodging_node（平行，查 RAG）
                                 → synthesis_node（LLM 生成自然語言建議）

資料層：repository.py（SQLite，Data/travel.db）
       + hybrid_search.py / rag.py（Chroma 向量庫 + BM25，本地免費 embedding，不用付費 API）
可觀測：Langfuse（選配，追蹤 LLM 呼叫延遲與 RAG fallback 比例）
```

## 功能

- **全台天氣查詢**：即時雷達/降水摘要、指定日期時段預報（資料源：Open-Meteo，免費無需金鑰）
- **景點/住宿/餐廳推薦**：混合檢索（BM25 + 語意向量 RRF 融合），支援「下雨天室內景點」這類天氣條件過濾
- **地圖 + CCTV 圖釘顯示**（輕量路徑，不需要任何 API key）
- **進階自然語言建議**（需使用者自行在側欄貼上 OpenAI API key，Bring-Your-Own-Key，app 擁有者不需負擔費用）
- **可觀測性**：Langfuse 追蹤（選配，未設定金鑰時優雅降級為無 tracing 版本）

## 資料來源

全部來自官方公開資料，不是手動蒐集：

- 觀光資料：交通部觀光署觀光資訊資料庫（景點/住宿/餐廳，含 `Facilities`/`RestaurantFeatures` 等結構化欄位）
- 交通站點：TDX 運輸資料流通服務
- 天氣：Open-Meteo（即時 + 預報）
- 行政區座標：從上述景點/住宿/餐飲三張表在每個縣市＋行政區內取平均座標推算（見 `collector/derive_district_coords.py`），不是手打的清單

收集流程與每個資料表的欄位/筆數/資料品質稽核記錄在 ROADMAP.md Phase 0。

## 技術重點

- **RAG**：`code/hybrid_search.py`（BM25 + 語意檢索 + RRF 融合）比純語意檢索的 Hit Rate@5 高 8.3 個百分點（91.67% vs 83.33%，24 組人工核對過的 ground truth，見 ROADMAP.md Phase 1）
- **本地免費 embedding**：`BAAI/bge-small-zh-v1.5`（透過 `fastembed`），避免使用者查詢時持續累積 OpenAI embedding 費用
- **LangGraph 多節點編排**：天氣查詢拆成 `intent_node → weather_node → attraction_node/lodging_node（平行）→ synthesis_node`；另一層 supervisor + worker graph 決定要不要呼叫 OpenAI 進階路徑，取代原本寫死在 UI 裡的 if/else
- **確定性修正取代模型猜測**：意圖解析的日期加減、行政區↔縣市對應這類「答案本來就在資料庫裡」的判斷，改用程式碼查表而不是讓 LLM 猜（詳見 ROADMAP.md Phase 6，把準確率從 76.5% 修到 93.8%）
- **兩層測試**：`tests/`（deterministic pytest，不打 LLM，接 CI）+ `eval/`（真的呼叫 OpenAI 的 live eval，手動/排程觸發，避免每次 push 都花錢）

## 本機執行

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 OPENAI_API_KEY（選填，沒填只能用輕量路徑）
streamlit run code/streamlit_app.py
```

首次執行前需要先跑過 `collector/` 底下的資料收集腳本產生 `Data/travel.db`（`.gitignore` 排除，不進版控）；RAG 索引則需要 `python collector/build_embeddings.py`。詳細維護流程見 [SKILL.md](SKILL.md)。

```bash
pytest tests/                          # deterministic 測試，免費，接 CI
python eval/run_eval.py hybrid         # RAG 檢索品質 eval，免費，需先建好索引
python eval/run_intent_eval.py         # 意圖解析準確率 eval，會打 OpenAI API
```

## MCP tool server

`code/mcp_server.py` 把景點/住宿/餐廳/行政區座標查詢包成 MCP tool，讓 Claude Desktop／Claude Code 等 MCP client 能直接查這些資料，不用透過 Streamlit 介面（跟既有 [cwa-mcp-server](https://github.com/gary2376/cwa-mcp-server) 同一套 SDK／寫法）。需要 Python >=3.10（`mcp[cli]` 套件要求，見 ROADMAP.md Phase 2）。

```json
{
  "mcpServers": {
    "weatherllm": {
      "command": "python3",
      "args": ["/絕對路徑/weatherLLM/code/mcp_server.py"]
    }
  }
}
```

提供 `get_attractions`／`get_lodgings`／`get_restaurants`／`get_district_coordinates` 四個 tool，跟一個 `weatherllm://cities`（全台22縣市清單）resource。

## 授權

程式碼採 MIT License（見 [LICENSE](LICENSE)）；各資料源（觀光資料標準 V2.1／TDX／Open-Meteo）各自的授權條款與顯名聲明要求，見 [DATA_LICENSE.md](DATA_LICENSE.md)。
