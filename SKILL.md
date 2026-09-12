# weatherLLM 維護手冊

記錄兩個最常見的維護動作的標準流程。

## 補景點／住宿／餐廳資料

資料來源是官方 API（`collector/fetch_tourism.py`），不是手動維護的清單。

```bash
cd /Volumes/gary_expand/project/weatherLLM
.venv/bin/python collector/fetch_tourism.py
```

會重新抓取並覆寫 `Data/travel.db` 的 `attractions`/`lodgings`/`restaurants` 三張表
（`load_attractions`/`load_hotels`/`load_restaurants`，見該檔 40-142 行）。跑完印出
三個表各自的筆數。

**跑完後必做，不能省略**：

1. `district_coords` 是從這三張表的座標平均算出來的（見下一節），資料變了要重跑：
   ```bash
   .venv/bin/python collector/derive_district_coords.py
   ```
2. RAG 向量索引（`Data/chroma/`、`Data/bm25_index.pkl`）是資料庫在某個時間點的快照，
   不會跟著 DB 自動更新——`build_embeddings.py` 目前是 `DELETE + 全量重建`，
   資料有異動就要重跑，不然 RAG 查到的還是舊資料：
   ```bash
   .venv/bin/python collector/build_embeddings.py
   ```
   這步骤會重新 embed 全部 attractions/lodgings/restaurants（上次約 14000+ 筆），
   跑一次要幾分鐘，本地跑（fastembed 是免費本地模型，不花 OpenAI 額度）。

3. 重跑 deterministic 測試確認沒壞：
   ```bash
   .venv/bin/python -m pytest tests/ -v
   ```

## 新增／修正行政區座標

`district_coords` 表（`city`+`district` 為主鍵）是查天氣座標用的行政區中心點，
不是手打的 dict——從已有的 attractions/lodgings/restaurants 三張表座標平均算出來
（見 `collector/derive_district_coords.py`）。

**不要手動 INSERT 這張表**。正確流程：

1. 確認來源資料（三張內容表）已經是最新的（見上一節第一步跑過 `fetch_tourism.py`）。
2. 重跑：
   ```bash
   .venv/bin/python collector/derive_district_coords.py
   ```
   會先 `DELETE FROM district_coords` 再重新 `GROUP BY city, district` 算平均座標，
   印出總筆數與樣本數低於3筆的行政區（座標較不可靠，通常是偏遠鄉鎮）。
3. 極少數完全沒有景點/住宿/餐飲資料的鄉鎮不會出現在這張表——這是預期行為，不是
   bug，`system.py::_resolve_coords()` 查無該區時會自動退回同縣市其他區的平均座標，
   查無該縣市才用寫死的臺中市中心當最後備援（`system.DEFAULT_COORDS`）。不用為了
   補齊這些鄉鎮手動塞座標。

## 兩者共通：schema 只改一處

`district_coords`（以及所有表）的 `CREATE TABLE` 都在 `collector/db.py` 的中央
`SCHEMA` 字串裡——`get_conn()` 會確保表存在。**不要**在個別收集腳本裡重複寫
`CREATE TABLE`，之前 `derive_district_coords.py` 曾經自己重複建表，導致從零重建
資料庫時如果漏跑該腳本，`system.py` 查詢會噴 `no such table` 而不是優雅退回預設值
（已修，教訓記在這裡避免重犯）。
