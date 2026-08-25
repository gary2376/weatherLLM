"""Hybrid 檢索：BM25(詞彙比對) + 向量語意檢索，用 RRF（Reciprocal Rank Fusion）
融合排序。city/district/type 這些精確條件仍然是融合前就先篩掉的硬 filter，不參與
RRF——跟 code/rag.py 同一套「硬條件用 filter、模糊偏好才交給檢索」設計原則。

用途：純語意檢索對「查詢字面就直接出現在文件裡」的情況不一定排最前面（見
eval/ 的評估結果），BM25 補上這塊字面比對的訊號，兩者互補。

用法：先跑 collector/build_embeddings.py 建好 Data/chroma/ 與 Data/bm25_index.pkl，
再呼叫 search()。
"""
import sys
from pathlib import Path
from typing import List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collector"))
import bm25_util  # noqa: E402

import rag  # code/rag.py 提供語意檢索與 Chroma collection

RRF_K = 60
CANDIDATE_POOL = 50  # 兩邊各取多少候選再做RRF融合。實測發現候選池太小時（試過30），
# 某一邊排名較後但另一邊排名很前面的正解會直接被排除在候選外，融合階段根本看不到它
# ——例如宜蘭三星鄉包棟民宿密集（數十家用詞高度相似），正解在語意檢索排第31名，
# 候選池=30時完全被排除；拉大候選池才有機會靠 BM25 的高排名把它撈回來。


def _bm25_ranked_ids(query: str, city, district, type_, weather_fit, index: dict) -> list:
    tokens = bm25_util.tokenize(query)
    scores = index["bm25"].get_scores(tokens)
    weather_fit_set = None
    if weather_fit:
        weather_fit_set = {weather_fit} if isinstance(weather_fit, str) else set(weather_fit)
    candidates = []
    for i, score in enumerate(scores):
        meta = index["metadatas"][i]
        if city and meta.get("city") != city.replace("台", "臺"):
            continue
        if district and meta.get("district") != district.replace("台", "臺"):
            continue
        if type_ and meta.get("type") != type_:
            continue
        if weather_fit_set and meta.get("weather_fit") not in weather_fit_set:
            continue
        candidates.append((index["ids"][i], score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [id_ for id_, _ in candidates[:CANDIDATE_POOL]]


def search(
    query: str,
    city: Optional[str] = None,
    district: Optional[str] = None,
    type_: Optional[str] = None,
    weather_fit: Optional[Union[str, List[str]]] = None,
    top_k: int = 5,
) -> list:
    index = bm25_util.load()
    bm25_ids = _bm25_ranked_ids(query, city, district, type_, weather_fit, index)
    semantic_hits = rag.search(
        query, city=city, district=district, type_=type_, weather_fit=weather_fit, top_k=CANDIDATE_POOL
    )
    semantic_ids = [h["id"] for h in semantic_hits]

    rrf_scores = {}
    for rank, id_ in enumerate(bm25_ids, start=1):
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (RRF_K + rank)
    for rank, id_ in enumerate(semantic_ids, start=1):
        rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (RRF_K + rank)

    ranked_ids = sorted(rrf_scores, key=lambda id_: rrf_scores[id_], reverse=True)[:top_k]

    id_to_text = dict(zip(index["ids"], index["texts"]))
    id_to_meta = dict(zip(index["ids"], index["metadatas"]))
    results = []
    for id_ in ranked_ids:
        results.append({
            "id": id_,
            "text": id_to_text.get(id_, ""),
            "score": rrf_scores[id_],
            **id_to_meta.get(id_, {}),
        })
    return results
