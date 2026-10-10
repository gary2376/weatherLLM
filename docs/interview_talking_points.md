# 面試技術問答集（Agent/RAG開發經驗）

這份文件記錄weatherLLM開發過程中遇到的真實技術問題跟解法，格式是「面試可能被問的問題 → 實際解法」。每一條都對應真的遇到、修過、有測試守著的案例（詳細過程見`ARCHITECTURE.md`），不是背答案。

**收錄標準**：只收「為什麼這樣開發/設計」的架構決策類問題（cache怎麼設計、怎麼證明品質、怎麼防護、怎麼監控…）。單純的環境建置/套件版本衝突這種操作性雜務不算——那是踩坑排除，不是設計決策，不放進來。

**隨開發進度持續新增，不要事後補、不要寫沒做過的事。**

---

## 成本/效能

**Q：重複查詢會不會一直花錢？**
A：兩層cache。RAG用`functools.lru_cache`包`hybrid_search()`，key是正規化查詢字串，資料只在手動重建索引(`build_embeddings.py`)時變，重建時連帶清cache。天氣用手刻TTL dict，45分鐘(CWA資料以小時計更新，30-60分鐘中位數)，沒裝Redis——單process單container量體，外部cache是過度設計。

**Q：併發情境下，同一秒多個請求miss到同一組cache key，會不會重複打API、重複扣費？**
A：in-flight request去重複。miss時先查有沒有其他thread已經在打同一組參數的API，有的話全部共用同一個`concurrent.futures.Future`，只有第一個真的打API，其他人等結果。Lock不跨`await`持有，避免卡住event loop。

**Q：Agent會不會因為推理繞圈/工具卡住而燒爆預算？**
A：Guardrail兩道防線：`recursion_limit=10`限制推理步驟數；cost cap直接從LLM回應的`usage_metadata`(token數)算錢累加，**不依賴**Langfuse這類旁路trace系統——guardrail是安全機制，不能讓它依賴一個可能斷線的第三方服務。

**Q：「選本地免費模型避免依賴外部API」這個設計決策，怎麼確保真的做到、不會意外破功？**
A：選`fastembed`跑embedding/reranker而非付費API的理由是「訪客越多、查詢越多，付費API就越貴」。但函式庫自己的快取機制預設把模型檔放在系統temp目錄，這個目錄會被OS定期清掉——一旦被清，下次呼叫就得重新連網路下載模型檔，等於「本地免費、不依賴外部」的設計意圖被函式庫的預設值偷偷打破，且這類快取失效通常在某次環境變動後才會發生，不會在一般開發流程裡被及時發現。解法：把`cache_dir`明確指向專案內固定路徑，不依賴系統temp目錄的生命週期。教訓：**架構決策落地要看到底層函式庫的隱性預設值，不能只看自己的程式碼邏輯有沒有打API**——快取、session、連線池這些「輔助機制」的預設行為，常常才是真正決定一個設計決策有沒有兌現的地方。

---

## RAG品質驗證

**Q：怎麼證明RAG不是瞎掰、檢索結果真的對？**
A：ragas三指標：**Faithfulness**(答案有沒有被檢索到的context支持)、**Context Recall**(有沒有撈到正確文件)、**Answer Relevancy**(答案語意有沒有對齊問題)。不是跑一次就收工——做了兩組測試集：v1泛用壓力測試(Faithfulness 0.89/Recall 0.54)跟v2唯一指向性測試(Recall拉到0.89)。發現v1很多題目可以被多篇文件合理回答，單一ground truth結構性拉低Recall，不代表retrieval真的差，所以v2改成「問題帶唯一辨識細節(年份/距離/認證)但不寫名稱」才是真正衡量檢索精準度的版本。

**Q：地理範圍查詢會不會撈到別縣市的資料？**
A：跨縣市查詢實測發現「城市名稱」在純文字/語意排名裡訊號太弱，會被其他縣市更熱門的同類關鍵字蓋過去(例：花蓮原住民景點查詢，Top-5命中率只有1/5)。解法是metadata**先過濾再排序**(`where={"city":{"$in":[...]}}`限定candidate範圍)，不是排完名次事後篩——跟RBAC「先過濾權限範圍再進搜尋」同一個設計模式，只是這裡過濾的是地理範圍不是權限。

**Q：資料裡的用詞變體/別名會不會讓過濾邏輯失效？**
A：22縣市全量測試才抓到：正式用字「臺北/臺中」跟口語「台北/台中」字面不同導致4個城市過濾完全沒生效；「馬祖」是連江縣俗名字面對不到。解法：查詢/字典雙邊正規化(臺→台)+手動別名表。樣本數不夠大的測試集很容易漏掉這種edge case。

---

## Agent/多工具協調

**Q：一句話包含多個需求(找景點+順便找住宿+餐廳)，agent會不會只處理一個就漏掉其他？**
A：驗證agent能正確把單輪query拆解成多個獨立工具呼叫，不是只回答第一個子需求（`run_multi_need_test()`永久回歸測試）。

**Q：LangGraph並發執行多個工具呼叫，底層資源(DB client/索引)安全嗎？**
A：實測到真的race condition：兩個工具呼叫同時初始化同一個lazy singleton(Chroma client)，沒上鎖導致底層binding壞掉。解法：`threading.Lock()` double-checked locking。教訓：**工具函式庫內部的thread-safety假設要自己驗證，不能假設框架幫你處理好**。

**Q：工具呼叫失敗時，agent會不會自己編一個看起來合理但沒有根據的答案？（Faithfulness在agent層的延伸）**
A：兩層防護。工具docstring層級的軟性指示(要求LLM誠實回報失敗)不夠硬，LLM還是可能忽略；補程式碼層硬性覆寫——偵測到工具回傳失敗訊息時，直接忽略後續LLM訊息，強制回傳固定失敗訊息，不給LLM編答案的機會。

**Q：防幻覺這種安全機制本身也可能有bug嗎？怎麼發現的？**
A：有，而且這個bug等級蠻高的——用`MemorySaver`做多輪對話記憶的框架，每次`ainvoke()`回傳的是**整個累積歷史**，不是只有這一輪的訊息。防幻覺硬覆寫原本掃描全部歷史找「RAG工具失敗」的訊息，結果變成：只要對話中**任何一輪**RAG查詢失敗過，之後每一輪(就算完全沒呼叫RAG工具，純問天氣)都會被鎖死在同一句固定的失敗訊息，使用者會以為系統一直故障。安全機制本身因為沒有限定「只看這一輪」的範圍，從「防止幻覺」變成「錯誤拒答」。同一個根因還牽連到cost計算——token成本也是對整個累積歷史重新加總再疊加，等於每輪都把前面所有輪次的花費重複算一次，guardrail會在使用者根本沒花到那麼多錢時就提早觸發。修法是寫一個共用的「只取最後一個使用者訊息之後的切片」輔助函式，cost計算跟失敗掃描都改吃這個切片，一次修兩個bug。教訓：**用了framework的checkpointer/記憶機制時，要搞清楚它回傳的是「這一輪」還是「全歷史」，不能憑直覺假設**，而且這種bug在單輪測試裡完全測不出來，一定要寫「多輪累積歷史」的情境去驗證。

**Q：整合第三方協定(MCP)時，回傳內容有文字就代表成功嗎？**
A：不行，MCP協定本身有`isError`旗標，只看`content`有沒有文字組出字串、沒檢查這個旗標，會把CWA端的錯誤訊息當成正常天氣資料回傳**並寫進cache**(錯誤被快取，比單純回傳錯誤更嚴重)。教訓：整合外部協定要理解協定語義，不能只看回傳格式像不像資料。

**Q：非核心的可觀測性工具(Langfuse)斷線，會不會拖垮整個agent？**
A：執行期(不只初始化)包try/except，斷線就停用trace、拿掉callback重試一次，使用者拿到正常答案而非錯誤——**核心功能不該依賴非核心旁路系統的健康狀態**，這是個通用的reliability設計原則。

---

## Web UI/多使用者部署

**Q：Web服務讓多個使用者各自輸入自己的API key，會不會互相洩漏？**
A：原本`WeatherAgent`直接吃process環境變數的`OPENAI_API_KEY`，但Streamlit是多使用者共用同一個process，若把使用者輸入的key寫進環境變數，會洩漏給其他並發session用。解法：建構子加`openai_api_key`可選參數，有傳才覆寫`ChatOpenAI(api_key=...)`，key全程只存在該使用者自己的`session_state`，不落地、不寫log、不進Langfuse trace。

**Q：Web框架每次互動都重跑整支script(不像傳統長駐process)，LLM agent的多輪對話記憶要怎麼保留？**
A：`MemorySaver`這個checkpointer本身用`thread_id`分流對話記憶，所以只要讓它跨rerun存活就好——存一份在`st.session_state`，每則訊息都重新開一個`async with WeatherAgent(checkpointer=...)`。MCP子行程(天氣工具的stdio連線)因此每輪重新啟動，但這是刻意的簡化：重開本地子行程成本小，換掉的是「背景常駐event loop」的複雜度，是不是要做取決於per-message延遲是否真的變成問題——先選簡單可行的，留明確的升級路徑。

**Q：怎麼讓使用者對LLM花費有感，而不是只在背後默默扣錢？**
A：把cost從一行純文字說明升級成`st.metric`這種帶delta的stat tile，每輪顯示這次新增花多少。關鍵是`delta_color`要設成`inverse`而不是用預設值——預設的「數字上升=綠色/好事」對一般指標(營收、使用人數)是對的，但對「花費」這種指標，上升反而該是需要注意的方向，用預設色會誤導使用者。這其實是資料視覺化一個很基本但常被忽略的原則：delta的顏色語意要看這個指標「往上是不是真的是好事」，不能無腦套用框架預設值。

**Q：想讓agent回傳一些「順便給UI用」的結構化資料(例如這次查詢命中了哪幾筆資料庫record)，但不想弄壞原本的文字回答、也不想在多使用者環境留下洩漏風險，怎麼設計？**
A：用LangChain工具的`content_and_artifact`回傳格式——工具可以回傳`(給LLM讀的文字, 給程式自己用的結構化資料)`二元組，後者(`artifact`)會掛在**那一次呼叫**產生的`ToolMessage`物件上，天生跟那次invocation綁定。這跟直接用一個模組層級的全域變數暫存「這次查到的東西」是完全不同等級的設計——全域變數在Streamlit這種多使用者共用一個process的環境下，兩個人同時送出請求就會互相覆寫、看到別人的資料，是跟API key沒做好隔離同一類的洩漏風險。善用框架本來就設計好的「per-invocation側載資料」機制，比自己手刻一個執行緒安全的暫存區穩妥得多。

**Q：LangGraph接了checkpointer做多輪對話記憶，如果要從對話歷史裡抓「這一輪」用到了什麼，要注意什麼？**
A：checkpointer回傳的`messages`是**累積的整個thread歷史**，不是只有最新這一輪的訊息。如果直接掃整個`messages`列表收集資料(這裡是RAG工具命中的地點id)，會把前面每一輪的結果都疊加進來，不是只有這輪。解法是在掃描時遇到`HumanMessage`就重置累加器——用這個訊號切出「這一輪」的邊界。這種「checkpointer回傳的是全歷史、不是增量」的行為，是多輪對話框架常見但容易漏掉的細節，寫測試時最好直接模擬「多輪累積歷史」的情境去驗證，不要只測單輪。

---

## Docker/部署

**Q：為什麼是單一container，不是每個服務(Streamlit/天氣工具)各自一個container？**
A：`cwa-mcp-server`不是獨立網路服務，是透過stdio(標準輸入輸出)跟Agent通訊的本地子程序(`code/mcp_tools.py`裡`StdioServerParameters`+`stdio_client`)，本質上是同一個Python process呼叫的一支CLI工具，不是要跨網路呼叫的microservice。拆成兩個container反而要多繞一層網路(TCP/socket)去模擬原本就是本地subprocess呼叫的東西，是不必要的複雜度——先判斷「這兩個東西的溝通方式是什麼」，再決定要不要拆，不是看到兩個功能就預設該拆成兩個service。

**Q：`cwa-mcp-server`這個子程序在container裡怎麼被找到並啟動？**
A：`CWA_MCP_SERVER_COMMAND = Path(sys.exec_prefix) / "bin" / "cwa-mcp-server"`——它是pip套件(`requirements.txt`裡的`cwa-mcp-server==0.1.1`)，正常`pip install`就會把它的console script裝到當下Python環境的`bin/`目錄。這行程式碼本地開發(venv)、container(系統Python，無venv)兩種環境都能用同一套邏輯解析路徑，因為兩種情況下`sys.exec_prefix`都會準確指向「這支程式自己所在的Python環境」。好處是不用寫死任何絕對路徑假設，也不用為了container環境另外寫一套路徑判斷邏輯。

**Q：Docker image裡，什麼東西該在build time烤進image、什麼東西該用volume掛載？怎麼判斷？**
A：判斷依據是「這份資料誰會改、多久改一次」。fastembed的兩個模型(embedding+reranker)只在套件版本更新時才會變，而且每次container啟動都要重新連外網下載會拖慢啟動、也讓runtime多一個對外部網路的依賴，所以選擇在image build階段跑一次`warmup_fastembed.py`預先下載，直接烤進image layer。相對地，`Data/`裡的`chroma`索引、`travel.db`、`bm25_index.pkl`是跑`collector/build_embeddings.py`之類的腳本重建資料後就會變的東西，選volume掛載，這樣重新跑collector更新資料後，不需要重新build整個image才能讓container讀到新資料。這兩種資料的「變動頻率」不同，資料生命週期管理策略也該跟著不同，不是整包全部bake進image圖方便，也不是整包全部volume掛載圖彈性。

**Q：公開部署時，使用者自己的OpenAI key、跟操作者自己的CWA/Langfuse key，為什麼用不同方式注入？**
A：`OPENAI_API_KEY`完全不進容器設定(不寫docker-compose、不寫env_file)，因為這是每個訪客自己要輸入的錢包——如果放進容器環境變數，變成操作者一個人的key要幫所有訪客的LLM呼叫買單，而且Streamlit是多使用者共用一個process，環境變數也沒辦法依訪客分流(這個設計決策在Web UI章節本來就定案過，Docker化只是沒有打破它)。`CWA_API_KEY`/`LANGFUSE_*`則是操作者自己申請、免費額度就夠portfolio demo流量用的key，風險屬於「操作者自己的成本」而非「訪客互相洩漏」，所以可以安心走docker-compose的`env_file`，整個容器共用一份即可。

**Q：Docker容器啟動成功、health check也過，就代表部署正確嗎？**
A：不一定——驗證時真的撞到一個「容器健康但資料是空的」案例。開發機原本沒裝Docker，裝`colima`(mac上headless、CLI-only的Docker runtime)補上後，第一次`docker compose up`容器確實正常啟動、health check也顯示healthy，但實際查詢RAG完全查不到資料。追查發現colima的VM預設只mount `$HOME`，看不到專案放在外接磁碟的路徑；docker-compose對單一檔案的bind mount(`travel.db`等)在VM端找不到來源檔案時，不會報錯，而是靜默建一個空目錄頂替，SQLite對空檔案一樣能正常connect，只是查不到任何資料。這種失敗模式不會讓container crash、不會讓health check失敗，純粹測「服務有沒有啟動」完全抓不到。教訓跟MemorySaver那次（見上面）同一個類型：**「看起來沒出錯」不等於「資料是對的」**，任何牽涉外部資料掛載/注入的環境（volume、secret、config injection），上線前都要用一次會真正讀到資料的功能性查詢去驗證，不能只看服務存活狀態。
