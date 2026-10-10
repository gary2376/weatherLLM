# weatherLLM 架構規劃

天氣感知旅遊agent = weatherLLM（旅遊RAG）+ cwa-mcp-server（已完成、已發布PyPI的天氣MCP）整合。

Project 1，2026-09-13起算，時程上限10週（見第7節）。完成定義：技術面熟到面試能講出設計原因，不只是能跑。

本文件邊討論邊更新，每節標狀態：✅確認（使用者明確拍板）/ 📋預設（AI建議、使用者同意採用）/ 🔲待討論（還沒談）。舊`ROADMAP.md`已不存在，不沿用其內容當事實。

**環境**（2026-09-29升級）：`.venv`從Python 3.9.6升級到**3.11.15**——因為Agent層要用的`langchain-mcp-adapters`(≥3.10)、`cwa-mcp-server`(≥3.11)都超出3.9支援範圍。升級過程踩了兩個版本相容坑：(1)不鎖版本重裝時`fastembed`解析到0.8.1，該版把`cache_dir`從有預設值改成必填參數，導致`embedding_fn.py`/`reranker_fn.py`初始化時卡死(實測26分鐘無回應、連第一筆查詢都沒印出)——修法：釘回驗證過的`fastembed==0.7.4`；(2)`langchain`家族(langchain/langchain-core/langchain-community/langchain-openai/langgraph/langsmith/openai/instructor)若不整組鎖定版本，pip會解析到互不相容的新世代版本(langchain 1.x系列)，導致`ragas`內部import `langchain_community.chat_models.vertexai`失敗——修法：整組8個套件一起釘回原本驗證過的版本(`langchain==0.3.30`等，完整清單見`requirements.txt`)。第三個坑：`langchain-mcp-adapters`舊版(0.1.x)要老的`langchain-core<0.4`(跟ragas相容)，但同時又要老的`mcp` SDK API(`streamablehttp_client`)，而`cwa-mcp-server`依賴的是新版`mcp==2.2.0`(該函式名已改成`streamable_http_client`)；新版`langchain-mcp-adapters`(0.3.x)則反過來要新的`langchain-core≥1.0`，跟ragas衝突。兩個世代互不相容，函式庫本身的斷層，不是版本沒挑對就能解。**決策：不用`langchain-mcp-adapters`，直接用`mcp` SDK本身寫stdio連線邏輯**(`mcp.client.stdio.stdio_client`+`ClientSession.list_tools()`/`call_tool()`)，只是4個工具，20-30行就能包完，換掉一個函式庫比讓兩個核心套件世代打架划算，也讓RAG那邊驗證過的舊世代`langchain-core`完全不用動。

升級後重跑`pytest`(11 passed)+`smoke_test_rag.py`(22縣市100%、邊界case全過)確認RAG層無迴歸，且乾淨檢查了ragas+langgraph+langchain_openai+mcp SDK四邊import鏈都沒被破壞。舊venv備份在`.venv_py39_backup`(暫留，之後可刪)，套件清單存在`requirements.txt`。

---

## 1. 資料層（已完成）✅

**Collector pipeline**（`collector/`）：
- `fetch_tourism.py`：觀光署景點/住宿/餐飲（data.gov.tw 7777/7780/7779），免key，每日更新
- `fetch_transit.py`：TDX公車/YouBike/停車場靜態資料，需TDX_CLIENT_ID/SECRET
- `link_transit.py`、`link_lodging_attraction.py`：haversine+網格加速算最近站點/景點關聯
- `derive_district_coords.py`：從既有資料算367個行政區中心點，取代手打dict
- `build_embeddings.py`+`embedding_fn.py`：本地`BAAI/bge-small-zh-v1.5`(fastembed，CPU跑，免費)寫入Chroma persistent collection
- `bm25_util.py`：jieba斷詞+BM25Okapi，lazy singleton快取索引檔

**現有資料量**：景點6084、住宿15673（篩>20字描述後進索引）、餐飲3686、交通站12046、行政區367。

**已知設計決策**：embedding/rerank選本地免費模型的原因一致——使用者查詢時即時算向量，若打付費API，訪客越多越花錢。

**踩過的坑：本地模型快取放在系統temp目錄，意外把「本地免費、不依賴外部」的設計意圖打破**(2026-10-03)——使用者回報Web UI查詢「反應時間好像有點久」，追查發現`collector/embedding_fn.py`/`code/reranker_fn.py`呼叫`fastembed.TextEmbedding()`/`TextCrossEncoder()`時都沒傳`cache_dir`，預設值落在系統temp目錄(`$TMPDIR/fastembed_cache`)，這個目錄macOS會定期清掉。一旦快取被清掉，下次呼叫會觸發`huggingface_hub`重新向HuggingFace Hub下載模型檔(~1.1GB)，而這個環境對`cdn-lfs.huggingface.co`等LFS CDN子網域DNS解析失敗(能連到`huggingface.co`本體，但抓不到實際模型binary檔)，導致卡在「Fetching 5 files: 0%」掛住不動，單次卡住實測超過6分鐘——使用者說的「有點久」其實是這種接近無限卡住的情況。修法：`cache_dir`改成指到專案內固定路徑(`Data/fastembed_cache`)，不再依賴系統temp目錄的生命週期；用`HF_ENDPOINT=https://hf-mirror.com`這個鏡像站繞過目前環境的DNS限制，把模型真的下載進新快取路徑一次，之後的查詢都直接讀本地檔案，不再需要任何網路。這個bug的根本教訓：「選本地免費模型」這個設計意圖，不能只看「程式碼邏輯上有沒有打付費API」，函式庫自己的cache/快取機制預設值也可能悄悄接回網路依賴，要實際驗證快取路徑穩不穩定。修完重跑`pytest`(27 passed)+`smoke_test_rag.py`(110/110)確認無迴歸，且重複呼叫`WeatherAgent.ask()`多次setup時間都在1秒內(修復前第一次可能卡到6分鐘以上)。

---

## 2. RAG查詢層 ✅

檔案預留：`code/rag.py`（純向量查詢）、`code/hybrid_search.py`（BM25+向量RRF融合）。

- **Hybrid search**：BM25 Top-50 + Chroma向量 Top-50 → RRF融合排名
- **Reranker**：`BAAI/bge-reranker-base`（fastembed內建`TextCrossEncoder`，跟embedding模型同家族、同樣CPU免費，不多裝依賴），精選Top-5
- **Eval**：ragas套件算Context Recall、Faithfulness、Answer Relevancy三指標。原規劃結果要送進Langfuse，**實際上`code/rag_quality/build_eval_set.py`/`run_eval.py`完全沒整合Langfuse**，結果只存本地json檔(`results_*.json`)+`docs/rag_quality/baseline_*.md`——2026-09-30盤點時發現這個落差，還原本規劃跟實作不一致。Langfuse目前只在Agent層(`code/agent.py`)有接，連線問題已排除確認trace正常送達(見第3節)

**測試集/門檻**（✅2026-09-26更新）：
- 兩組測試共用完全相同的66筆文件：從RAG使用的同一份BM25語料依22縣市分層抽樣，每縣市3筆，原始文件作為ground-truth reference。v2直接沿用v1的`id`、`reference`、`metadata`，只更換問題設計，兩組都固定保存、不重新抽樣
- **v1泛用查詢壓力測試**（2026-09-25，66/66有效、查無context 0筆）：Faithfulness **0.8906**、Context Recall **0.5439**、Answer Relevancy **0.5266**。不少問題可由多筆文件合理回答，但每題只有單一ground truth，因此主要用於retrospective regression，不把低Recall直接視為檢索失敗；詳見`docs/rag_quality/baseline_2026-09-25.md`
- **v2唯一指向性測試**（2026-09-26，66/66有效、查無context 0筆）：Faithfulness **0.8765**、Context Recall **0.8889**、Answer Relevancy **0.4963**。與v1使用同一批文件，但問題加入年份、距離、活動、認證、設施等唯一辨識細節且不直接寫名稱；Context Recall較v1提升**0.3450**，作為衡量RAG目標文件檢索精準度的主要版本。詳見`docs/rag_quality/baseline_v2_unique_2026-09-26.md`
- v1回歸門檻維持Faithfulness≥0.80、Context Recall≥0.50、Answer Relevancy≥0.50，且查無context維持0筆。v2先保存本次實測baseline供後續同條件比較；Faithfulness與Answer Relevancy分別衡量答案受context支持程度及語意對齊，不是純檢索命中率，不能因v2 Recall提升就預期同步上升，也不能把三項合成單一「檢索精準度」

**實作狀態**（2026-09-26）：hybrid search+reranker已完成，`code/hybrid_search.py`（BM25+向量RRF融合，含lru_cache）、`code/reranker_fn.py`（bge-reranker-base）、`code/rag.py`（對外入口，agent之後接這支）。Eval的v1泛用壓力測試與v2唯一指向性測試、baseline及用途區分均已完成（見上面結果）。

**踩過的坑：跨縣市查詢命中率低**——上線前手動跑9筆跨縣市查詢(`code/smoke_test_rag.py`)驗收，發現「花蓮原住民文化景點」「南投溫泉住宿」「屏東海鮮餐廳」這類「縣市+全台通用熱門關鍵字」的查詢，Top-5命中率只有1/5：其他縣市同類型文件的語意/BM25分數常贏過目標縣市較弱的匹配，city名稱在純文字排名裡訊號太弱蓋不過關鍵字。修法：`hybrid_search.py`新增`detect_cities()`，query裡偵測到明確縣市名就先用metadata(`where={"city":{"$in":[...]}}`給Chroma、篩candidate給BM25)過濾限定範圍再排名，不是排完名次事後篩——跟FDE系統設計裡「RBAC先過濾權限範圍再進vector search」同一個道理，只是這裡過濾地理範圍不是權限。`detect_cities()`重用`fetch_transit.py`裡現成的22個縣市中文名`CITY_ZH`，沒有另外自己編一份清單。

**全台22縣市完整驗收**(2026-09-21，`code/smoke_test_rag.py`)：第一輪只挑9個縣市測，這輪22個全測，又抓出兩個之前沒測到的真bug：
1. `CITY_ZH`用正式全形「臺」字(臺北市/臺中市/臺南市/臺東縣)，但口語幾乎都打「台」，兩字字面不同比對不到——臺北/臺中/臺南/臺東這4個城市過濾完全沒生效(`detect_cities()`回傳`None`)，臺北命中率掉到1/5
2. 「馬祖」是連江縣的俗名，字面上完全對不到

修法：比對前把「臺」正規化成「台」(query跟字典key兩邊都轉)，另外手動補一張「馬祖→連江縣」的別名表。修完22縣市總命中率96%(106/110)。當時把剩下2筆「新竹市/新竹縣」「嘉義市/嘉義縣」誤判成「設計本來就這樣、測試寫太嚴格」，不是真的去查證——後來Codex複核直接戳破這個判斷。

**主題性查詢人工檢視**：不帶縣市的查詢("適合帶小孩的親子景點"、"背包客青年旅館推薦"、"素食友善餐廳")語意相關性都好，結果都真的扣題。"適合雨天的室內景點"這筆結果比較弱，文字裡沒明顯扣題雨天/室內——這剛好印證前面的已知設計邊界：RAG完全靠文字語意/關鍵字，不吃`weather_fit`這個metadata，天氣情境過濾要留給agent層做，不是RAG的鍋。

**Codex獨立複核**(2026-09-21)：請Codex對`hybrid_search.py`/`reranker_fn.py`/`rag.py`/兩支測試檔案做一次獨立review，抓出幾個真問題並修掉：
1. **「新竹市/新竹縣」誤判成非bug其實是真bug**：`detect_cities()`比對用的是去掉市/縣的裸名「新竹」，就算query明確寫「新竹市」也還是被錯誤放寬成兩個都算。修法：每個裸名群組先檢查query有沒有literal寫出完整縣市名，有就只精確匹配那個，沒有(只寫裸名)才維持兩個都算的寬鬆邏輯
2. `reranker_fn.rerank_indices()`沒擋`documents`為空、`top_k<=0`(負數slice會回傳「除了最後幾筆以外的全部」)——補了guard
3. `rag.py`的`_doc_lookup`用`if not _doc_lookup`判斷初始化，跟「索引真的是空dict」這兩種狀態會混淆；且多執行緒下第一次查詢有競態風險——改用`None` sentinel+threading lock(double-checked locking)
4. `hybrid_search.clear_cache()`原本只清`search()`的lru_cache，Chroma collection handle跟bm25_util的singleton還是舊的，同一process內重建索引不會真的生效——改成三層一次清掉，`bm25_util.py`也補了對應的`reset_cache()`
5. BM25斷詞後沒token(如空查詢)時，`get_scores()`回傳全0分數，原本排序只是照原始index順序選、不是相關性排名——補guard直接回傳空list
6. `rag.query()`候選id在doc lookup裡一筆都找不到時(Chroma/BM25索引不同步的訊號)原本靜默回傳`[]`，改成丟明確錯誤
7. 測試檔案本身也有假陽性風險：RRF測試用平手分數斷言、`smoke_test_rag.py`把`docs==[]`判定成成功、cache測試只比對結果沒比對`cache_info().hits`有沒有真的+1、exception只印出來但exit code還是0——都補上明確斷言，腳本結尾用失敗筆數決定`sys.exit(1)`

修完全部重跑：`pytest test_hybrid_search.py`累計11 passed（原7個+Codex抓到的4個新case）；`smoke_test_rag.py`22縣市100%(110/110)、cache/多縣市/連江縣稀疏資料等邊界case全部真的通過，exit code 0。

---

## 3. Agent層 ✅

框架：LangGraph（使用者偏好現成框架，理由：業界多半也是框架上修改，非手刻）。

- **架構**：單一ReAct agent，非multi-agent（工具數5個：hybrid_search + cwa-mcp-server四個天氣工具，遠低於「<10個工具用單一agent」門檻）
- **防線**：LangGraph自帶`recursion_limit`當迭代上限；cost累計中斷**獨立於Langfuse運作**——直接從LLM回應的`usage_metadata`(input/output tokens)換算費用累加，Langfuse斷線/沒設定guardrail仍然生效(原2026-09-14規劃寫的是靠Langfuse trace算費用，實作時改成更穩健的獨立算法，這裡同步更新，詳見下面「實作狀態」)
- **不做**：人工審核層（此agent無刪除/退款等高風險操作，跳過）

**防線數值/LLM選型**（✅2026-09-14確認）：
- recursion_limit=10（工具簡單，正常查詢2-4次tool call應該夠，太緊再依實測數字放寬）
- cost threshold：單一session US$0.05-0.10，實作取中位US$0.08，直接從token usage_metadata累加cost比對(不靠Langfuse)，純防呆天花板非預期日常會碰到
- LLM維持OpenAI（`.env.example`原本預設）。曾考慮換Anthropic用prompt caching省成本，但使用者判斷：這個project的cache價值該來自自己設計的cache機制（見第4節，key設計/失效策略能講出原因），不是provider內建打勾功能，換Anthropic理由不成立，故維持OpenAI

**實作狀態**(2026-09-29，改由我直接寫，不是Codex——過程中Codex用量快到上限，使用者要求換手)：`code/mcp_tools.py`(把cwa-mcp-server的4個工具包成LangChain `StructuredTool`，直接用`mcp` SDK寫stdio連線，不用`langchain-mcp-adapters`，見上面「環境」段落的決策原因)、`code/agent.py`(單一ReAct agent，`create_react_agent`+`MemorySaver`+`recursion_limit=10`+cost guardrail)、`code/smoke_test_agent.py`(多輪對話+guardrail驗收，累積失敗項目用`sys.exit(1)`回報)。

**cwa-mcp-server連線方式**：沒有用uvx，直接用專案`.venv`裡已經pip安裝好的`cwa-mcp-server`console script路徑(`sys.exec_prefix/bin/cwa-mcp-server`)當subprocess command——升級到Python 3.11後這個套件能正常裝進我們自己的venv，不用再多一層uvx依賴。

**驗收結果**：3輪對話("南投日月潭有什麼景點"→"附近有推薦餐廳嗎"→"有素食的嗎")，第2、3輪都正確從對話歷史解析出「日月潭/南投」這個地點脈絡再疊加素食條件，拿到真的扣題答案(不是答非所問)，證明`search_travel_info` tool的docstring設計(要求LLM傳入脈絡完整的查詢字串，不要照抄使用者原句)有效。3輪真實對話總花費US$0.0063，遠低於US$0.08門檻。Guardrail測試(手動把某thread的累積花費設超過門檻)正確擋下，回傳安全上限訊息，沒有繼續呼叫LLM/工具。

**天氣工具驗證**：`smoke_test_agent.py`的3輪對話都只問景點/餐廳，沒有真的經過agent觸發過天氣工具，另外補一次直接問「台北市現在有沒有生效中的天氣特報？」，確認agent真的透過ReAct流程呼叫`get_weather_warning`(實際打了CWA的`W-C0033-001`特報API)並正確回答「目前沒有生效中的特報」——weather tool整合也驗證通過，不是只在`mcp_tools.py`獨立測試裡跑得動。

**已解決(原「已知待查項目」，2026-10-06驗證)**：先前懷疑`cloud.langfuse.com` region跟帳號對不上導致連線逾時，2026-10-06直接用`Langfuse` SDK的`auth_check()`+真的打public API查最近trace列表驗證——**認證正常、trace真的有送達伺服器端**(查得到今天跟前幾天測試時產生的trace記錄)，不是region設定問題，原本的逾時應該是當時環境的暫時性網路狀況，不是系統性bug。agent這端執行期的容錯(斷線時停用trace重試一次，見下方「全端驗證」)仍然保留，屬於防禦性設計，不是因為真的還有問題。

**Docker前最後一輪bug清查**(2026-10-06，委派Codex獨立審查`agent.py`/`mcp_tools.py`/`app.py`，我審查+真實多輪對話驗證)：使用者要求所有bug修完才進Docker。這輪Codex審查沒照我prompt列的5個具體檢查項找到真bug(那5項都驗證沒問題)，但自己多讀出2個我完全沒料到、藏在MemorySaver累積歷史裡的真bug，等級都是「核心安全機制本身壞掉」：

1. **歷史RAG失敗會污染後續輪次**：`ask()`原本掃描`result["messages"]`**整個累積歷史**找`search_travel_info`的失敗訊息——如果對話第1輪RAG查詢失敗過，就算第5輪只是單純問天氣(根本沒呼叫RAG工具)，舊的失敗訊息還是會被掃到，整個對話被鎖死在固定的失敗訊息，使用者會以為系統一直故障。這個bug本身還蠻諷刺的：防幻覺硬覆寫這個安全機制，因為沒有限定掃描範圍，反而變成「錯誤地拒答」的bug。
2. **多輪對話cost被重複累加**：`_response_cost_usd()`原本也是對整個累積歷史算token cost，每輪都把所有歷史輪次的AI訊息cost**重新加總一次**再疊加到累計花費——等於每多聊一輪，前面所有輪次的花費都被多算一次(接近二次方成長)，guardrail的US$0.08上限會在使用者根本還沒花到那麼多錢的時候提早觸發，使用者會覺得系統「莫名其妙就擋住了」。

修法：新增`_current_turn_messages()`共用輔助函式，從訊息列表尾端往回找最後一個`HumanMessage`，只回傳從那裡到結尾的切片——cost計算跟RAG失敗/doc_ids掃描都改吃這個切片，不是整份歷史，一次修兩個bug用同一個邏輯來源。另外`WeatherAgent.__aenter__()`跟`WeatherMCPTools.__aenter__()`補上失敗時的資源清理(`BaseException`時手動呼叫`__aexit__`/`AsyncExitStack.aclose()`)——Python的async context manager規範是`__aenter__`丟例外就不會呼叫`__aexit__`，之前如果LLM/graph初始化在MCP連線建立後失敗，MCP子行程會沒人收，變成孤兒process。

新增4個測試(`test_previous_rag_failure_does_not_poison_later_weather_turn`、`test_agent_cost_only_adds_latest_turn_from_accumulated_history`、`test_weather_only_turn_does_not_reuse_previous_rag_doc_ids`、`test_agent_entry_failure_closes_started_weather_lifecycle`)，每個都直接模擬「累積歷史」情境去驗證，不是只測單輪。`pytest`34 passed，真實環境重跑`smoke_test_agent.py`全過，多輪對話累積花費從之前常見的US$0.006左右降到US$0.0037(修掉重複計算後數字正確變小)。另外用`playwright-core`跑了一次「RAG查詢→純天氣問題→RAG查詢(不同地點)」三輪真實瀏覽器測試，確認：cost每輪正確遞增不重複、地圖命中地點正確只反映最新一輪(不會沿用前面輪次的舊地點)、天氣輪正確不顯示任何地圖標示。

**全端驗證與錯誤處理補強**(2026-09-30，委派Codex做系統性測試，我審查+在正式環境重跑驗證)：對照`docs/diagrams/agent-workflow.json`每個節點(`planner`/`router`/`guardrail`/`aborted`/`rag_tool`/`weather_tool`/`trace`/`final`，`webui`因尚未開工不算)逐一補「錯誤/例外路徑」測試，不只測happy path。抓到並修好4個真bug：
1. **RAG工具storage失敗會外洩**：Chroma collection遺失/BM25 index損毀時，原本`rag.query()`的例外沒被`search_travel_info`工具接住，會直接讓agent看到raw exception。補`try/except`，回傳固定的「旅遊資料庫暫時無法查詢」訊息。
2. **防幻覺safeguard只在docstring層，不夠硬**：即使RAG工具已經誠實回報失敗，LLM後續還是可能自己編一個看起來合理的答案(docstring只是軟性建議，非強制)。補程式碼層的**硬性覆寫**——`ask()`偵測到`search_travel_info`回傳失敗訊息時，直接忽略後續LLM訊息，強制回傳固定失敗訊息，不給LLM編答案的機會。
3. **Langfuse只在初始化時防護，執行期沒有**：trace callback真的斷線時(不是建立handler失敗，是呼叫過程中失敗)原本沒接住，這次補上——執行期呼叫失敗會捕捉、停用trace、拿掉callback後重試一次，最終使用者拿到正常答案而非錯誤。
4. **MCP `CallToolResult(isError=True)`被當成正常天氣資料**：`mcp_tools.py`只組裝了回傳內容文字，沒檢查MCP協定本身的`isError`旗標，代表CWA那端回錯誤時，錯誤訊息會被當成正常天氣資料回傳並寫進cache。補上旗標檢查，`isError=True`時轉成例外，不快取、不當成功回傳。

另外訂正一個cache邊角案例：只有owner（沒有其他等待者）且底層呼叫失敗時，內部`concurrent.futures.Future`的例外沒人領走，會在GC時噴「exception was never retrieved」警告(非功能性bug，純日誌雜訊)——改成owner也跟waiter一樣統一透過`asyncio.wrap_future()`拿結果/例外。

新增`code/test_error_handling.py`(9個測試，對齊上面4個修復點+guardrail/recursion的乾淨錯誤訊息)，`code/test_mcp_tools_cache.py`補2個測試(併發失敗會讓所有等待者收到例外、MCP isError不會被快取)，`code/smoke_test_agent.py`補`run_weather_test()`(真實走一次agent→weather tool→final，不只測`mcp_tools.py`自己的cache邏輯，避免漏測agent routing這段)。全部27個pytest案例+`smoke_test_rag.py`(110/110)+`smoke_test_agent.py`(多輪對話+多需求併發+天氣routing+guardrail)在正式環境用真實OpenAI/CWA API重跑，全數通過。

**踩過的坑：多需求單輪query的併發競態**(2026-09-30)——使用者問「除了景點還有住宿/餐廳資料，多需求query系統處理得了嗎」，實測「我想去南投日月潭玩，順便找住宿跟推薦餐廳」這句：agent正確把它拆成2個獨立`search_travel_info`工具呼叫(住宿、餐廳分開查)，證明多需求分解設計有效；但這2個工具呼叫是LangGraph**併發**執行的，撞上`hybrid_search.py::_get_collection()`跟`bm25_util.py::load()`的lazy singleton初始化都沒上鎖——兩個thread同時判斷「還沒初始化」，各自建立一個`chromadb.PersistentClient`，導致底層Rust binding壞掉(`RustBindingsAPI object has no attribute 'bindings'`)。**更嚴重的連鎖問題**：工具報錯後，agent沒有老實跟使用者說查詢失敗，反而自己編了一個聽起來合理但完全沒有資料庫根據的答案(直接違反Faithfulness設計目標，且不會被現有eval發現——eval跑的是單一query，測不到這種併發場景)。

修法：(1)`hybrid_search.py`的`_get_collection()`、`bm25_util.py`的`load()`都補上`threading.Lock()`double-checked locking，跟`rag.py::_get_doc_lookup()`當初Codex複核修的同一個模式；(2)`agent.py`的`search_travel_info`工具docstring補一句「查詢失敗要老實說，不要自己編答案」當第二層防線。修完重跑同一句多需求query：兩個併發工具呼叫都正常回傳，答案裡的民宿/餐廳都是資料庫裡真實存在的record，不再瞎掰。

**這個bug原本不會被抓到**——原因是`smoke_test_agent.py`的3輪測試是循序的(每輪只問一件事)，從沒測過「單輪就要agent同時處理多個需求」這種會觸發併發工具呼叫的情境。已補進`run_multi_need_test()`當永久回歸測試(不是只靠這次手動發現)，重跑整套`smoke_test_agent.py`(多輪對話+多需求併發+guardrail)全部通過，exit code 0。

---

## 4. Caching層 ✅

- **RAG cache**：✅已完成——資料只在手動重跑`build_embeddings.py`時變動，用Python標準庫`functools.lru_cache`包`hybrid_search()`，key用正規化查詢字串。重建索引時順便清cache
- **天氣cache**：✅已完成(2026-09-30，Codex實作、我審查套用)——`code/mcp_tools.py`裡`WeatherMCPTools._build_tool()`的`_acall`加了TTL cache：手刻dict存(結果,timestamp)，TTL取30-60分鐘中位數45分鐘(`WEATHER_CACHE_TTL_SECONDS`常數)，key用「工具名稱+排序後的完整JSON參數」組成，過期才重打API，沒裝Redis(單process單container量體，外部cache過度設計)。

  **比原規劃多做的一層**：吸取剛發現的併發競態bug教訓，這次一開始就用`threading.Lock()`保護cache讀寫，還多加了「in-flight請求去重」——同一組參數同時被多個併發工具呼叫miss到時，只有一個真的打API，其他等同一個`concurrent.futures.Future`共用結果，不會重複打API也不會重複扣guardrail的cost額度。Lock不跨`await`持有，避免卡住event loop。

  **驗收**：新增`code/test_mcp_tools_cache.py`(pytest風格，4個測試：同key命中cache、過期重打、不同參數各自獨立cache、併發同key只打一次API)，用假的session計數器驗證不需要真的CWA_API_KEY。連同`test_hybrid_search.py`一起跑，15 passed。重跑`smoke_test_agent.py`(多輪對話+多需求+guardrail)確認整合進agent後沒有迴歸，全部通過。

  **這次委派的小插曲**：Codex這次是同步執行(不是背景Task)，直接撞到沙箱寫入`/Volumes/...`外部路徑的權限牆，但它自己想辦法在暫存目錄複製一份`mcp_tools.py`實作+驗證過，測試真的通過才回報。我讀了暫存目錄裡驗證過的程式碼，審查沒問題後直接套用到正式路徑，改寫成pytest格式補上，不用再重新委派一輪耗Codex用量。

---

## 5. Web UI ✅

Streamlit+pydeck（2026-10-01實際安裝——盤點時發現`requirements.txt`沒列、venv裡也沒裝，跟文件原本寫的「已裝」不符，先裝起來+重跑全部27個pytest+RAG smoke確認沒弄壞既有依賴鏈，才開始寫UI）。

**實作**(`code/app.py`，委派Codex實作+我審查+在正式環境真實瀏覽器驗證)：

- **對話頁籤**：`st.chat_message`/`st.chat_input`，`st.session_state`存訊息列表供畫面顯示，`asyncio.run()`呼叫`WeatherAgent.ask()`(非streaming——token-level streaming需要重構`ask()`內部的guardrail/Langfuse重試/防幻覺覆寫邏輯，這些都是圍繞`ainvoke()`完整結果設計的，這輪不做，列為下一輪待辦)
- **多輪記憶跨Streamlit rerun**：Streamlit每次互動都重跑整支script，沒辦法让`async with WeatherAgent()`長駐。解法：`MemorySaver()`存一份在`st.session_state`，每則訊息都重新開一個`async with WeatherAgent(checkpointer=st.session_state.checkpointer)`——MCP子行程(cwa-mcp-server)每輪重新啟動，但對話記憶(MemorySaver內部用`thread_id`分流)跨輪存活。這是刻意的簡化(重開本地stdio子行程成本小)，換掉背景常駐event loop的複雜度；之後若單輪延遲變成真的問題，升級路徑是背景常駐thread持有單一長生命週期的`WeatherAgent`
- **多使用者API key隔離**：`WeatherAgent.__init__`原本只吃process環境變數的`OPENAI_API_KEY`，Streamlit多使用者共用一個process，直接改環境變數會洩漏key給其他並發session。補了`openai_api_key`建構子參數，有傳才覆寫`ChatOpenAI(api_key=...)`，沒傳維持原行為(所有既有測試呼叫`WeatherAgent()`不用改)。Key只存`st.session_state`，`st.text_input(type="password")`輸入，驗證非空+`sk-`開頭，沒填/格式錯不讓對話輸入框啟用
- **地圖頁籤**：獨立瀏覽功能(2026-10-06前)。直接讀`collector.db.DB_PATH`的三張表，`st.cache_data`快取讀取結果；城市多選+類別勾選框篩選；`st.pydeck_chart`，景點/住宿/餐廳各一個`ScatterplotLayer`，配色用`dataviz` skill驗證過的3類別調色盤(景點藍`#2a78d6`/住宿橘`#eb6834`/餐廳青綠`#1baf7a`，這三個slot是文件裡明確驗證過all-pairs CVD-safe、適合scatter/map這種同時顯示多類別重疊點的場景，不是隨便挑的)

**驗證**(2026-10-01，兩階段)：
1. Codex在自己的sandbox裡做邏輯驗證(`py_compile`+streamlit in-process script test)，受限於sandbox沒有對外網路/無法bind socket，沒辦法真的跑`smoke_test_agent.py`或開真實server——這次連sandbox預設的scratch暫存目錄都不能寫(`No usable temporary directory found`)，比之前任何一次都嚴格，最後是靠明確加`-s workspace-write -C <目標路徑>`這兩個旗標重跑才真的落地到正式路徑(細節見決策記錄)
2. 我在正式環境重新驗證：`pytest`27 passed(agent.py建構子改動零迴歸)、`smoke_test_agent.py`全部通過(真實OpenAI/CWA API)、真的啟動`streamlit run`+用`playwright-core`連線headless Chrome(`chromium.connectOverCDP`)驅動真實瀏覽器——對話頁籤畫面正確、地圖頁籤一開始deck.gl因為Chrome `--disable-gpu`導致WebGL被關掉整個不會畫(`Failed to create WebGL context`)，改用`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`軟體渲染後地圖正常顯示全台+金門離島資料點(配色正確對應三類別)；最後真的填入OpenAI key、送出「南投日月潭有什麼景點？」，看到agent真實回答完整渲染在對話框，cost累計正確顯示US$0.0007

**對話狀態/session**（✅2026-09-14確認，已照此實作）：
- 同一次瀏覽器session內：`st.session_state`(streamlit原生)存對話訊息列表
- agent多輪記憶：LangGraph `MemorySaver`(純記憶體checkpointer)+`thread_id`綁streamlit session id或session_state裡的隨機uuid，讓agent接住上一輪脈絡(如追問「附近有推薦餐廳嗎」)
- 不做跨session/重啟持久化，YAGNI。之後真要做，`MemorySaver`換`SqliteSaver`即可，介面不用大改
- **API Key輸入**：`st.text_input(type="password")`讓使用者自己貼OpenAI key，只存`st.session_state`，不落地/不寫log/不進Langfuse trace明文；基本格式驗證(非空、`sk-`開頭)，沒填不讓agent跑

**下一輪待辦**(刻意排除在這輪之外，不是忘記)：token-level streaming、持久化背景event loop(如果per-message延遲變成真問題)。

**對話結果跟地圖連動**(2026-10-06，委派Codex實作+我審查+真實瀏覽器驗證)：使用者要求問某個地點時地圖要自動標示agent推薦的地點。核心機制用LangChain `@tool(response_format="content_and_artifact")`——`search_travel_info`現在回傳`(文字給LLM讀, {"doc_ids": [...]})`二元組，`artifact`這個side-payload掛在**那一次呼叫**的`ToolMessage`物件上，天生跟那次invocation綁定、不共享狀態，跟先前`openai_api_key`建構子參數要解決的是同一類問題(Streamlit多使用者共用一個process，絕對不能用module-level可變變數暫存單次請求的資料，會洩漏給其他並發session)。

`WeatherAgent.ask()`回傳簽章從`str`改成`tuple[str, list[str]]`(reply文字+這輪命中的doc_ids)，所有既有呼叫點(`smoke_test_agent.py`/`test_error_handling.py`/`app.py`)都要改。防幻覺硬覆寫邏輯沒被動到，在同一個訊息掃描迴圈裡延伸，所有失敗/中斷路徑(RAG失敗/recursion limit/cost cap)都回傳空list。有個真的正確性細節：LangGraph的checkpointer回傳的是**累積的整個thread歷史**，不是只有這一輪的訊息，所以doc_ids的累加器要在每個`HumanMessage`重置，不然地圖會一直疊加顯示前面所有輪次命中過的地點，不是只顯示最新這輪——這是Codex在實作時自己抓到的細節，不是我在prompt裡明講的。

`app.py`新增`load_highlighted_points()`(`st.cache_data`，吃`tuple[str,...]`當hashable cache key)，把`"attraction:A12345"`這種doc_id拆開查對應SQLite表拿到座標，疊一層`ScatterplotLayer`在原本的點位圖層上面——跟一般點位用同樣的類別色，但半徑變大+白色外框區分「這是推薦的」。命中地點時地圖view自動縮放置中到那些點附近(zoom 10.5)，而不是維持全台視角讓使用者自己找。

驗證：`pytest`29 passed(新增2個測試，一個驗證多輪對話only回傳最新輪doc_ids、一個驗證RAG失敗路徑doc_ids必須是空list)；`smoke_test_agent.py`真實環境重跑全過，`run_multi_need_test()`新增斷言確認多需求query真的回傳非空doc_ids列表；真實瀏覽器填key送「南投日月潭有什麼景點？」，確認地圖真的自動縮放定位到日月潭並標示10個白框地點。

**視覺/UX改版**(2026-10-05，委派Codex實作+我審查+真實瀏覽器驗證，純前端調整，agent/RAG/MCP後端零改動)：
- `.streamlit/config.toml`自訂主題，`primaryColor`沿用地圖「景點」類別已驗證過的藍`#2a78d6`，介面跟地圖視覺一致，不另外發明新色
- cost追蹤從純文字`st.caption`改成`st.metric`stat tile，帶每輪增量`delta`，且`delta_color="inverse"`——花費增加不該顯示成預設的「up=好事」綠色，這是依`dataviz` skill的「delta方向性」規則特意設定，不是Streamlit預設值
- 地圖頁籤加一排3欄`st.metric`，即時顯示篩選後的景點/住宿/餐廳數量(不是固定的全台總數)，真實驗證過勾掉類別/選城市後數字會跟著變
- 側邊欄用`st.divider()`分成「帳號設定」/「使用量」兩區塊，不再是一條直線堆到底
- 對話頁籤加空狀態提示(3個範例問題)，有訊息後自動消失
- 全程只用Streamlit原生元件，沒加CSS/JS/新依賴
- **後續更新(同日)**：使用者看了screenshot覺得跟原本太像、不喜歡，問清楚方向是要「更有品牌感的客製化外觀」。這輪改由我自己直接寫CSS(不委派Codex，純視覺調整需要邊看畫面邊調，用既有的headless Chrome截圖pipeline自己迭代更有效率)，注入`st.markdown(..., unsafe_allow_html=True)`：標題區加漸層底色塊、側邊欄用淺色調跟主畫布區隔、對話泡泡卡片化(使用者/助理用不同底色，不只靠avatar icon)、stat tile加陰影圓角、輸入框圓角化。色票沒有另外發明新色，沿用地圖三類別已驗證過的CVD-safe配色。過程中抓到一個真的bug：`font-weight:800`加重metric數字字重後，金額字串在側邊欄窄版面被截斷(`US$0.0...`)，補`font-size`+`white-space:normal`修正，確認`US$0.0000`完整顯示才算過。
- **第三輪(同日)**：使用者又反應「排版還是跟剛剛一樣」——第二輪其實只換色/加陰影，沒動版面結構本身，這次批評成立。真的重寫版面：`.block-container`收斂最大寬度置中(wide layout在大螢幕上edge-to-edge太空洞)、側邊欄改成浮動玻璃感面板(`backdrop-filter: blur()`+獨立圓角陰影，不貼邊)、hero區塊從`st.title`換成自訂HTML(漸層icon徽章+標題+副標二行結構)、tabs從瀏覽器預設底線樣式換成膠囊式分段控制、全域背景改多層radial-gradient營造氛圍(不是單一線性漸層)、對話泡泡改非對稱圓角(貼近聊天app手感)。過程中發現一個真bug：Streamlit幫沒設定logo的app保留一段`stSidebarHeader`空白高度，疊加我自己的padding後側邊欄頂部出現一大塊不該有的空白，補CSS縮掉這段保留高度才修正。
- **第四輪(同日)**：使用者要求「參考別人網站設計風格」而非自己憑空改。查了`artifact-design` skill才發現前兩輪其實踩中它明確點名的「AI通用設計套路」(Inter字體、rounded-lg到處用、淺色漸層hero)——這解釋了為什麼改了兩輪使用者還是覺得沒差。另外派agent實際抓Windy.com/Linear.app/Kayak三個真實網站的live CSS token(不是憑空想像)當參考，重新定調成「氣象觀測儀面板」概念：換成深色底(呼應Windy/Linear)、字體換IBM Plex Sans/Mono(拉丁字母/數字用等寬字體呈現「儀器讀數」感，這點是刻意貼合這個app本身就是天氣應用的屬性，不是裝飾)、圓角統一縮小到8-10px(參考Linear的節制)、陰影幾乎拿掉只留hairline邊框(同樣參考Linear)、地圖三類別配色(橘/藍/綠)同時當主要accent(不是單一孤色，刻意避開「近黑+單一孤色跳色」這個也被skill點名的套路)。`.streamlit/config.toml`改成`base="dark"`，這是蓄意的單一深色主題選擇，不是忘記做light/dark雙模式。

---

## 6. Docker ✅已實作並真實驗證（Dockerfile/docker-compose/模型warmup，`docker compose up`跑通）

**單一container**（2026-09-14確認，2026-09-30依Agent層實際決策更新第一點）：
- Streamlit app process(agent+RAG+UI)，cwa-mcp-server**直接用venv裡pip安裝好的console script路徑當subprocess command**(不用`langchain-mcp-adapters`——這個函式庫跟環境版本衝突，Agent層改用`mcp` SDK本身寫stdio連線，見「環境」段落決策；原文寫的是用langchain-mcp-adapters，已過時，這裡同步修正)
- `Data/`(chroma+travel.db+bm25_index.pkl)用volume掛載，不bake進image——重跑collector更新資料後容器不用rebuild
- fastembed兩個模型(embedding+reranker)在image build階段跑warmup腳本先下載進image layer，避免runtime依賴外網才能抓模型
- 環境變數：`CWA_API_KEY`+`LANGFUSE_*`(操作者自己的，非使用者風險)走docker-compose `env_file`；`OPENAI_API_KEY`改由使用者自己在Web UI輸入(見第5節)，不放進env/docker-compose

**Langfuse**：用Cloud免費版(Hobby：50k units/月、30天資料保留、2 users、免信用卡，見[官方定價](https://langfuse.com/pricing))，對portfolio demo流量規模夠用，不自架(自架要多幾個container，postgres/clickhouse/redis，跟"單container"規劃衝突)。

**公開部署 🔲待辦**（2026-10-07確認方向：Oracle Cloud Always Free VM + 全自動CICD，使用者要求手把手引導、非現在立刻做）：

部署平台選型比較過的選項：
- **Oracle Cloud Always Free VM**(✅已選定)：永久免費ARM小型VM，直接跑現有`docker compose up`(跟本機colima驗證過的流程一致，不用改架構)，無冷啟動。缺點：帳號申請需要信用卡做身分驗證(留在免費額度內不會扣款，但卡號是必填)，偶有帳號被誤判濫用而停權的狀況。
- Google Cloud Run：免費額度更單純，但不吃`docker-compose.yml`(單container+無volume)，`Data/`要改成烤進image、拿掉compose——跟現有Docker設定路線不同，已排除。
- Streamlit Community Cloud：免費但完全不用這次做的Docker——直接吃GitHub repo+`requirements.txt`佈署，`Data/`(~1.2GB)沒地方放(repo不該塞binary資料)，需另外寫「啟動時從外部儲存下載資料」的程式碼——等於這輪Docker工作用不到，已排除。
- Fly.io/Railway/VPS：都能直接跑現有`docker compose up`，但要付費(~US$5/月起)，暫不選。

**CICD範圍確認：CI+CD全自動**(不是只做CI)：
1. 🔲 **Oracle VM建立**——使用者要求之後手把手引導(用`oci` CLI，需使用者自己產生API key並授權)，目前還沒動手。
2. ✅ **CI基礎(`/.github/workflows/ci.yml`已建立)**：push/PR觸發，跑`pytest`(code/目錄)+驗證Docker image build得起來(`docker/build-push-action`，`push: false`)。**還沒實際在GitHub上跑過一次**(依賴VM之外的另一個待確認項目：這個repo有沒有接GitHub remote，尚未確認)。
3. 🔲 **CD(build→推registry→SSH部署)，還沒寫**，需要先有VM才能定案細節，規劃方向：
   - Docker image build完後推到GHCR(`ghcr.io/<repo>`，用`GITHUB_TOKEN`免額外申請credential，但repo的Workflow permissions要開`packages: write`)
   - `docker-compose.yml`同時保留`build:`(本機開發用)跟加上`image: ghcr.io/...`(VM用`docker compose pull`拉這個名字的image)，不用拆兩份compose檔
   - GitHub Actions跑SSH deploy step上VM，執行`docker compose pull && docker compose up -d`，需要的GitHub Secrets：VM的host/user/SSH私鑰(VM建好後才能設定)
   - 觸發條件：push到main才跑CD(test+docker-build的CI每個branch/PR都跑，part of gate)
4. 🔲 **CICD要測什麼(具體清單，目前只做了pytest+image build，以下尚未覆蓋)**：
   - 真實smoke test(`smoke_test_agent.py`)要不要納入CI？目前依賴真實OPENAI_API_KEY/CWA_API_KEY，若要在CI跑，金鑰要放GitHub Secrets，且每次CI都會真的打API花錢——需評估要不要only在CD(部署前)跑一次，而非每個PR都跑
   - 容器啟動後的「資料是不是真的掛進去」驗證(這次本機驗證抓到colima bind mount變空目錄的真bug，見下方Docker/部署章節)——CD流程裡部署後要不要自動打一次真實查詢確認RAG有資料，而不是只看container health狀態
   - Dockerfile/docker-compose.yml本身的設定正確性(YAML語法、`.dockerignore`排除項目)目前只有這次人工審查過，沒有自動化检查
   - 是否要在CI加`py_compile`/`ruff`/型別檢查這類靜態分析(目前完全沒有，只有pytest)——待評估是否划算

---

## 7. 時程對應 ✅

（2026-09-14確認：抓上限10週而非硬性2個月/9週，換取彈性buffer）

依技術依賴關係排序，非均分。真實時數核對：地基期W1-5週三是通用軟工技能，不算project時數，project只吃週一二+週六(抓中位數12hr/週)；W6+回常態週三併回project(14.5hr/週)。

- **W1-3**：RAG查詢層(hybrid search+reranker)——這份collector程式碼歷史上踩過好幾次資料細節坑(city中英轉換、tdx_id跨縣市重複key、lodging/attraction關聯搞混)，抓3週而非2週，實測debug量通常比寫code本體多
- **W4-5**：Agent層(LangGraph+MCP stdio整合)——subprocess/async這塊常卡在意外地方，抓2週而非1週
- **W6**：Caching層(小，dict TTL+lru_cache)+Docker前置(Dockerfile/compose骨架)
- **W7**：Docker完整整合(warmup腳本、volume、MCP在container內跑起來驗證)——不跟caching擠在一起，獨立抓一週
- **W8**：Eval(50+筆測試集建置+baseline跑+門檻定案)
- **W9-10**：Web UI(對話+地圖+API key輸入)+整合測試+最終打磨——這塊風險最低，若前面超時可從這裡挪buffer

---

## 決策記錄

- 2026-09-14：Agent用現成框架非手刻；RAG的hybrid search+reranker+eval全做不砍；地圖UI用簡單網頁呈現；reranker/cache TTL採用AI建議之預設值（BAAI/bge-reranker-base、天氣cache 30-60分鐘TTL、RAG cache用lru_cache無時間TTL）。
- 2026-09-14：RAG eval測試集抓50筆以上，門檻不預設死、先跑baseline再定。
- 2026-09-14：Agent防線數值(recursion_limit=10、cost threshold US$0.05-0.10)確認；LLM維持OpenAI(cache機制由使用者自己設計，不靠provider prompt caching換Anthropic)。
- 2026-09-14：Web UI對話狀態用`st.session_state`+LangGraph `MemorySaver`，不做跨session持久化；OpenAI API Key改由使用者自行輸入(不用操作者的錢包)。
- 2026-09-14：Docker單一container(streamlit app+cwa-mcp-server經stdio子process)，Data/掛volume、模型warmup進image；Langfuse用Cloud免費版(Hobby，對portfolio流量夠用)。
- 2026-09-14：時程抓上限10週(非硬性9週)換彈性；依技術依賴排序W1-3 RAG、W4-5 Agent、W6 Caching+Docker前置、W7 Docker整合、W8 Eval、W9-10 Web UI+整合。

**至此原訂待討論的5個主題(eval測試集/agent防線與LLM/web UI session/Docker/時程)皆已確認，整體系統規劃完成。**

- 2026-09-30：盤點文件跟實際進度的一致性，抓到4處落差並修正：(1)第4節Caching層天氣TTL cache從未實作(RAG cache有做)，✅改標🔶部分完成；(2)第5節Web UI完全沒動手寫，✅改標📋設計已定/🔲尚未實作；(3)第6節Docker完全沒動手寫且內容過時(還寫著已放棄的`langchain-mcp-adapters`)，✅改標📋/🔲並修正內容跟第3節同步；(4)第2節「eval結果送進Langfuse」從未實作，eval腳本只存本地json，同步修正說明；另外第3節「防線」描述的cost guardrail算法(靠Langfuse trace)跟實際實作(直接算token usage，不依賴Langfuse)不一致，也同步更新。
- 2026-09-30：驗證「多需求單輪query」時抓到真的併發競態bug——`hybrid_search.py`/`bm25_util.py`的singleton初始化沒上鎖，LangGraph併發執行多個工具呼叫時會撞壞Chroma client，且agent工具報錯後會瞎掰答案(違反Faithfulness)。修法：兩處補`threading.Lock()`+`search_travel_info`docstring補「查詢失敗要老實說」防線；新增`smoke_test_agent.py::run_multi_need_test()`當永久回歸測試，不能只靠手動發現。詳見第3節。
- 2026-09-30：天氣TTL cache補完(第4節)——委派Codex實作，這次同步執行撞到沙箱寫入外部路徑的權限牆，Codex自己在暫存目錄驗證過才回報，我審查通過後直接套用到正式路徑。除了原規劃的TTL dict，多加了threading.Lock+in-flight Future去重複(吸取剛發生的併發bug教訓)，補`test_mcp_tools_cache.py`(4測試)驗證，`smoke_test_agent.py`重跑確認無迴歸。至此Caching層(RAG+天氣)全部完成。下一步依序：Web UI(Streamlit)→Docker。
- 2026-09-30：對照流程圖節點做系統性全端驗證(委派Codex背景任務測試每個節點的正常流程+錯誤處理，我審查後在正式環境用真實API重跑確認)。Codex這次背景任務的sandbox workspace root被釘在呼叫端(Claude Code worktree)而非weatherLLM專案路徑，導致它對正式路徑的寫入被當成「工作區外」擋下(Chroma DB報`readonly database`)——跟先前「Codex同步執行撞沙箱牆」是同一類問題的變體，一樣靠Codex在暫存目錄驗證產出patch、我審查後手動套用解決。這次sandbox額外擋了對外網路(OpenAI/CWA DNS)，所以Codex只能驗證錯誤處理邏輯本身，真實API的happy path由我在正式環境重跑補齊。抓到並修好4個真bug(RAG工具例外外洩、防幻覺safeguard只在docstring層不夠硬、Langfuse只防護初始化沒防護執行期斷線、MCP `isError`旗標被忽略導致錯誤被當成功回傳並快取)，詳見第3節「全端驗證與錯誤處理補強」。新增`code/test_error_handling.py`+補強`test_mcp_tools_cache.py`/`smoke_test_agent.py`，27個pytest案例+兩支smoke test全數在正式環境通過。流程圖每個已實作節點(`webui`除外，尚未開工)現在都有對應測試覆蓋。
- 2026-10-01：委派Codex的方式從`codex:codex-rescue` subagent改成直接`cd`進目標目錄+`codex exec`背景跑——codex-rescue會把Codex的sandbox workspace root釘在呼叫端(Claude Code)自己的worktree，不是目標專案路徑，這正是前一條決策記錄提到的sandbox錯位根因。純`codex exec`預設sandbox更嚴格，連`/tmp`都無法寫，後來補上`-s workspace-write -C <目標路徑>`才真的能落地寫檔，兩個旗標缺一不可。詳見記憶檔`feedback_codex_invocation_method.md`。
- 2026-10-01：Web UI(Streamlit+pydeck)完成，詳見第5節。盤點時發現`requirements.txt`沒列streamlit/pydeck、venv裡也沒裝，跟文件原本寫的「已裝」不符——又一次文件跟實作落差，先裝起來並重跑全部測試確認無迴歸才開始寫UI。用新的Codex委派方式(見上條)背景跑完後，我在正式環境實際啟動server+用`playwright-core`連線headless Chrome做真實瀏覽器驗證(不是只信Codex自己說測過)，過程中抓到headless Chrome的`--disable-gpu`會讓pydeck的deck.gl WebGL canvas完全畫不出來，換成`--use-gl=angle --use-angle=swiftshader`軟體渲染才看到地圖；最後真的填入OpenAI key跑一次真實對話，確認回答正確渲染、cost累計正確。這輪也補了一個真實的設計修正：`WeatherAgent`原本直接吃process環境變數的OpenAI key，Streamlit多使用者共用一個process會洩漏key給其他並發session，改成建構子可選參數覆寫。
- 2026-10-03：使用者回報Web UI查詢「反應時間好像有點久」，追查抓到真bug——`embedding_fn.py`/`reranker_fn.py`的fastembed模型快取預設落在系統temp目錄，被OS清掉後會卡在重新向HuggingFace Hub下載模型檔，而且這個環境對HF的LFS CDN子網域DNS解析失敗，單次卡住實測超過6分鐘。修法：`cache_dir`改指到專案內`Data/fastembed_cache`固定路徑，用鏡像站`hf-mirror.com`把模型重新下載進新快取路徑一次，之後查詢都讀本地檔案不再碰網路。詳見第1節。重跑`pytest`(27 passed)+`smoke_test_rag.py`(110/110)確認無迴歸，且重複驗證`WeatherAgent`多次setup時間都壓在1秒內。
- 2026-10-05：Web UI視覺/UX改版，詳見第5節。委派Codex實作(用已修正的委派方式一次就成功落地，沒再撞sandbox牆)，純前端調整、後端零改動。我審查程式碼後，重新走一次完整驗證流程：`pytest`27 passed、真實啟動server、`playwright-core`連線headless Chrome做互動測試(不只截圖，實際點掉地圖篩選checkbox確認數字真的即時變動、真實填key送真實問題確認cost delta顯示跟方向性顏色正確)。經過三輪視覺改版(漸層卡片→玻璃感側邊欄→氣象儀面板深色主題)使用者都覺得「排版沒差」，最後放棄純視覺調整路線，改成明確的版面需求：對話跟地圖拆開成tabs切換使用者覺得不方便，要同一頁並排顯示。第5節的「這輪刻意不跟對話結果連動」設計決策維持不變——並排顯示只是版面位置變化，不是功能連動，地圖依然是獨立瀏覽功能。`st.columns([5, 7])`對話欄位較窄、地圖欄位較寬(地圖需要的視覺空間比純文字多)，`.block-container`最大寬度同步從1040px放寬到1440px騰出並排空間，地圖高度從620px降到520px配合新版面比例。
- 2026-10-05：使用者接著說「不要暗色系，但也不要白色」。CSS架構本來就是token-based(`:root`集中定義顏色變數，所有規則引用變數不寫死色碼)，這次只需要把token值從深色組換成淺色組，結構(hairline邊框、小圓角、等寬數字、tabs樣式)完全不動——canvas選帶灰綠調的石板色`#eef1f0`(不是純白，也刻意避開跟前幾輪類似的淺藍色調)，地圖本身的深色底圖不受影響，淺色canvas+深色地圖形成對比。`.streamlit/config.toml`的`base`同步改回`light`。
- 2026-10-05：使用者再提三個具體需求(地圖還是深色、對話要像聊天室固定輸入框+內部捲動、對話跟地圖要並排水平對齊)。前兩個純前端、我自己直接做：`pdk.map_styles.LIGHT`換掉deck.gl預設的dark底圖；`st.container(height=560)`包住訊息列表、`st.chat_input`寫在container外面(Streamlit官方「inline chat」寫法)，讓輸入框貼齊固定高度訊息框下方而非釘死在整頁最底部。並排對齊在前一天(10-04)已經用`st.columns([5,7])`處理過，這次沒有重複改。第三個需求(地圖呼應對話推薦地點)判斷要動`agent.py`核心回傳結構，委派Codex處理，詳見第5節。
- 2026-10-06：地圖呼應對話推薦地點完成，詳見第5節。用`content_and_artifact` tool response格式讓結構化資料(doc_ids)跟著單次tool呼叫的`ToolMessage`走，避免Streamlit多使用者process共用module-level狀態的洩漏風險——跟`openai_api_key`建構子參數那次是同一類設計原則的延伸應用。Codex在實作時自己抓到一個我prompt沒明講的正確性細節：LangGraph checkpointer回傳累積的整個thread歷史，doc_ids累加器要在每個`HumanMessage`重置才不會讓地圖疊加顯示所有歷史輪次命中過的地點。
- 2026-10-06：使用者問「還有沒有需要改的」，主動用窄視窗(800px)截圖檢查，抓到一個真bug——地圖頁籤3欄stat tile跟類別勾選列在窄螢幕下數字會壓成直的一個字一行疊起來，完全看不懂。第一次嘗試(全域`flex-wrap`)修壞了對話/地圖主欄位原本的5:7比例(被拉成1:1)，退回來後改用`st.container(key=...)`給這兩個3欄區塊各自掛穩定CSS class，CSS只精準對這兩個class生效，不動到主欄位排版。驗證過兩種寬度：窄螢幕3個stat tile正確換行堆疊成單欄可讀、正常寬度維持原本3欄並排，`pytest`29 passed無迴歸。
- 2026-10-06：使用者要求Docker前把目前已知問題全部修完，Streaming明確排除在這輪範圍外(已確認)。查證Langfuse「已知待查」其實已經失效解決(見第3節，不是bug)。委派Codex對`agent.py`/`mcp_tools.py`/`app.py`做一次獨立審查，抓到2個我完全沒預期到的真bug：MemorySaver回傳累積歷史導致「歷史RAG失敗污染後續輪次」跟「多輪cost重複累加」，兩個都是靠同一個新輔助函式`_current_turn_messages()`(只取最後一個HumanMessage之後的切片)修掉，另外補上`__aenter__`失敗時的資源清理。詳見第3節「Docker前最後一輪bug清查」。`pytest`34 passed、真實smoke test全過、用`playwright-core`跑三輪真實多輪對話(RAG→天氣→RAG)在真實UI上逐項驗證cost正確遞增、地圖標示正確只反映最新一輪。至此判斷可以進入Docker階段。
- 2026-10-07：使用者問「流程圖更新了嗎」，查證後發現**流程圖從9/30之後就沒再更新過**，`webui`節點還顯示🔲未開始、`final`節點還寫「地圖🔲待Web UI」，跟實際進度(Web UI早已完成並驗證)嚴重脫節——這是我的疏漏，每次做完功能性變更後沒有養成同步回頭更新`agent-workflow.json`的習慣，只顧著更新`ARCHITECTURE.md`。補上：`webui`/`final`節點改✅、`rag_tool`tag加註doc_ids→地圖連動、subtitle跟三張卡片內容更新到2026-10-07狀態(含10/6的MemorySaver兩個bug修復)。教訓：之後每輪只要涉及功能完成度變化(不是純視覺調整)，都要跟`ARCHITECTURE.md`同一輪一起檢查流程圖是否也要同步，不能只靠使用者提醒才想到。
- 2026-10-07：Docker單一container基礎設施完成：新增Python 3.11 slim的`Dockerfile`、單一app service的`docker-compose.yml`、`.dockerignore`與fastembed build-time warmup腳本；從`.env`載入操作者的CWA/Langfuse設定，Streamlit綁定`0.0.0.0:8501`並啟用headless模式，`cwa-mcp-server`跟其他requirements安裝在同一Python環境。已在既有Python 3.11 venv執行warmup確認embedding與reranker都能初始化，Streamlit in-process startup/render檢查通過，`pytest` 34 passed，YAML/語法與secret boundary檢查通過；本執行環境沒有Docker CLI，且sandbox禁止綁定本機port，因此image build、container啟動、container內MCP subprocess與HTTP health endpoint尚未實機驗證，需在有Docker的主機補跑`docker compose up --build`。Codex一開始把整個`Data/`掛成volume(`./Data:/app/Data`)，我審查時發現這跟同一節「fastembed模型warmup進image build layer」的設計互相打架——`Data/fastembed_cache`本身也在`Data/`底下，整個目錄掛volume會在container啟動時蓋掉build time就下載好的模型快取，變成每次啟動都要重新連外網抓模型，等於白做了warmup這一步。修法：改成只掛`Data/chroma`、`Data/travel.db`、`Data/bm25_index.pkl`三個實際由host端collector腳本產生、需要跨rebuild保留的路徑，`Data/fastembed_cache`留在image layer裡不被覆蓋。教訓：volume掛載範圍跟image build階段「烤」進去的內容如果共用同一個父目錄，掛載要刻意避開build time產出的子路徑，不能圖方便整個父目錄掛進去。
- 2026-10-07：這台機器當時沒裝Docker，用Homebrew裝`docker`/`docker-compose`/`colima`補齊(Docker Desktop是GUI app不適合這種操作模式，colima是mac上標準的headless CLI替代方案)。`docker compose build`+`up -d`真的跑起來後，發現一個colima預設設定的真bug：colima VM預設只mount `$HOME`，看不到專案所在的外接磁碟`/Volumes/gary_expand`，導致docker-compose裡對該路徑下單一檔案的bind mount(`./Data/travel.db`等)在VM裡找不到來源檔案，自動被建成空目錄而非真檔案——RAG會讀到完全空的資料庫卻不會報錯(sqlite對空檔案一樣能connect，只是query不到資料)，這種失敗模式不會讓container crash或health check失敗，很容易被誤判為「跑起來了=沒問題」。修法：`colima start --mount /Volumes/gary_expand:w`明確掛上外接磁碟後重跑，確認`travel.db`/`bm25_index.pkl`/`chroma.sqlite3`在container裡都是真檔案(有正確檔案大小、sqlite能查到資料)。確認無誤後，用`playwright-core`對container的8501埠跑三輪真實多輪對話(RAG日月潭景點→真實CWA天氣查詢→RAG台南住宿)，確認：真實資料被RAG正確檢索、`cwa-mcp-server`在container內以subprocess正常啟動並透過真實API回應、OpenAI真實呼叫成功、cost累計與地圖橘色光暈標示都正確反映最新一輪、`docker compose logs`無錯誤訊息。至此Docker容器化從「檔案寫好、未實機驗證」升級成「build+run+真實全端功能都驗證過」。教訓：bind mount來源路徑在VM/容器平台上「看不到」時，很多平台的預設行為是靜默生成空目錄而非報錯，這類狀態要主動用真實資料查詢去驗證「资料真的在裡面」，不能只看container有沒有健康啟動。
- 2026-10-07：使用者問「整個系統算完成了嗎」——核對後本機RAG+Agent+Caching+Web UI+Docker都✅並驗證過，唯二例外是刻意排除的streaming(使用者10/6已確認不列入)跟還沒做的公開部署。使用者接著確認要公開部署，比較Oracle Cloud Always Free VM/Google Cloud Run/Streamlit Community Cloud三個選項(見第6節新增的「公開部署」小節)，選定Oracle Free VM(能直接沿用剛驗證過的docker-compose，不用像Cloud Run改架構或像Streamlit Cloud另外寫資料下載邏輯)。接著使用者要求CICD(先CI後CD，最後確認要做**CI+CD全自動**)當作專案的正式一環，不只是基建雜務：已建立`.github/workflows/ci.yml`(pytest+docker build驗證)，CD(build→推GHCR→SSH部署到VM)規劃方向寫進第6節，但使用者明確表示VM建立要之後再手把手引導，這輪先把待辦(CICD要測什麼、VM怎麼建、CD細節)記錄完整即可，不動手實作。同時把這次Docker設計的「為什麼」(單一container理由、cwa-mcp-server子程序如何被container找到、build-time烤入vs. volume掛載的判斷依據、兩種API key注入方式為何不同)補寫進`interview_talking_points.md`的Docker/部署章節。
