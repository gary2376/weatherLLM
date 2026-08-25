"""RAG 查詢層：對 Data/chroma/ 的 travel collection 做語意檢索＋metadata 精確過濾。
只服務 attractions/lodgings(篩過)/restaurants(篩過) 這三類有意義的自由文字內容，
索引怎麼建見 collector/build_embeddings.py。transit_stations/district_coords 等
純結構化表不在這裡，走 repository.py 的 SQL 查詢——不是每個表都適合語意檢索。

embedding 用本地免費模型（collector/embedding_fn.py 的 bge-small-zh-v1.5），不是
OpenAI API：使用者每次打字查詢都要重新算一次 embedding，用付費 API 訪客越多越花錢，
本地 ONNX 模型下載一次後永久免費、CPU 就能跑。
"""
import sys
from pathlib import Path
from typing import List, Optional, Union

import chromadb

# embedding_fn.py 放在 collector/（跟 build_embeddings.py 共用同一份實作，
# 兩邊必須用同一個模型算出來的向量才能互相比對），這裡加進 sys.path 才 import 得到
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collector"))
from embedding_fn import get_embedding_function  # noqa: E402

CHROMA_PATH = Path(__file__).resolve().parents[1] / "Data" / "chroma"
COLLECTION_NAME = "travel"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=get_embedding_function())
    return _collection


def search(
    query: str,
    city: Optional[str] = None,
    district: Optional[str] = None,
    type_: Optional[str] = None,
    weather_fit: Optional[Union[str, List[str]]] = None,
    top_k: int = 5,
) -> list[dict]:
    """語意檢索＋metadata 硬過濾。city/district/type/weather_fit 是精確條件，
    query 是模糊偏好。city/district 一律「台→臺」正規化，跟 repository.py 同規則
    （資料源存官方全名）。weather_fit 只有 attractions 有這個 metadata 欄位
    （indoor_ok/outdoor_only/mixed/unknown，見 collector/attraction_classes.py），
    傳list（如["indoor_ok","mixed"]）可以一次接受多個值，用 Chroma 的 $in。"""
    where_clauses = []
    if city:
        where_clauses.append({"city": city.replace("台", "臺")})
    if district:
        where_clauses.append({"district": district.replace("台", "臺")})
    if type_:
        where_clauses.append({"type": type_})
    if weather_fit:
        if isinstance(weather_fit, str):
            where_clauses.append({"weather_fit": weather_fit})
        else:
            where_clauses.append({"weather_fit": {"$in": list(weather_fit)}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    collection = _get_collection()
    result = collection.query(query_texts=[query], n_results=top_k, where=where)

    hits = []
    for id_, doc, meta, dist in zip(
        result.get("ids", [[]])[0],
        result.get("documents", [[]])[0],
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0],
    ):
        hits.append({"id": id_, "text": doc, "distance": dist, **meta})
    return hits
