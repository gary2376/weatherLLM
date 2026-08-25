"""離線建索引：把 attractions 全部、lodgings 篩過的、restaurants 篩過的
description 塞進 Chroma persistent collection（Data/chroma/，語意檢索用）跟
BM25 lexical 索引（Data/bm25_index.pkl，關鍵字檢索用，見 bm25_util.py）——兩份
索引用同一份 docs 建，供 code/rag.py（純語意）、code/hybrid_search.py（BM25+語意
RRF融合）查詢。

不 embed 樣板句/空描述——垃圾進垃圾出。住宿description抽查發現雙峰分布：
中位數只有8字（如「位於臺中市的旅館」這種樣板句），>50字才是真敘述，20字是兩群之間
乾淨的斷層，取這個門檻篩掉樣板句。詳見 ROADMAP.md Phase 1 決策。

embedding 用本地免費模型（見 embedding_fn.py 的 bge-small-zh-v1.5），不打 OpenAI
API，因為 code/rag.py 之後要即時 embed 使用者輸入，用付費 API 訪客越多越花錢。

用法：python3 collector/build_embeddings.py（跑一次全量重建，不做增量 diff——
資料量小，重跑成本比維護 diff 邏輯低）
"""
from pathlib import Path

import chromadb

from db import get_conn
from embedding_fn import get_embedding_function
import bm25_util

CHROMA_PATH = Path(__file__).resolve().parents[1] / "Data" / "chroma"
COLLECTION_NAME = "travel"
BATCH_SIZE = 100
LODGING_MIN_DESC_LEN = 20


def _fetch_attractions(conn):
    rows = conn.execute(
        "SELECT id, name, city, district, category, description, weather_fit FROM attractions"
    ).fetchall()
    docs = []
    for id_, name, city, district, category, description, weather_fit in rows:
        text = "、".join(p for p in [name, category, description] if p)
        docs.append({
            "id": f"attraction:{id_}",
            "text": text,
            "metadata": {
                "type": "attraction",
                "city": city or "",
                "district": district or "",
                "category": category or "",
                "weather_fit": weather_fit or "",
            },
        })
    return docs


def _fetch_lodgings(conn):
    rows = conn.execute(
        "SELECT id, name, city, district, description, price_range FROM lodgings "
        "WHERE description IS NOT NULL AND length(trim(description)) > ?",
        (LODGING_MIN_DESC_LEN,),
    ).fetchall()
    docs = []
    for id_, name, city, district, description, price_range in rows:
        text = "、".join(p for p in [name, description] if p)
        docs.append({
            "id": f"lodging:{id_}",
            "text": text,
            "metadata": {
                "type": "lodging",
                "city": city or "",
                "district": district or "",
                "price_range": price_range or "",
            },
        })
    return docs


def _fetch_restaurants(conn):
    rows = conn.execute(
        "SELECT id, name, city, district, category, description FROM restaurants "
        "WHERE description IS NOT NULL AND length(trim(description)) > 0"
    ).fetchall()
    docs = []
    for id_, name, city, district, category, description in rows:
        text = "、".join(p for p in [name, category, description] if p)
        docs.append({
            "id": f"restaurant:{id_}",
            "text": text,
            "metadata": {
                "type": "restaurant",
                "city": city or "",
                "district": district or "",
                "category": category or "",
            },
        })
    return docs


def main():
    conn = get_conn()
    docs = _fetch_attractions(conn) + _fetch_lodgings(conn) + _fetch_restaurants(conn)
    conn.close()
    print(f"待建索引文件數：{len(docs)}")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = get_embedding_function()

    # 每次重建都整批換掉，資料量小（約1.4萬筆），重跑成本比維護增量邏輯低
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        print(f"  已寫入 {min(i + BATCH_SIZE, len(docs))}/{len(docs)}")

    print(f"完成，collection 內文件數：{collection.count()}")

    n_bm25 = bm25_util.build_and_save(docs)
    print(f"BM25 lexical 索引也建好了：{n_bm25} 筆，存在 {bm25_util.BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
