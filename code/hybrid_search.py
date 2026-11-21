"""BM25 Top-50 + Chroma向量 Top-50 → RRF融合排名，回傳doc id清單（未rerank，
rerank在 rag.py 做）。

BM25/embedding兩邊索引和斷詞規則都跟 collector/build_embeddings.py 共用同一份
實作（下面sys.path那行就是為了直接import collector/裡現成的 bm25_util.py、
embedding_fn.py，不重寫一份）——建索引跟查詢用不同斷詞/模型的話，詞頻表/向量
空間對不上，分數就沒有意義。

同一份查詢字串快取（lru_cache）：資料只在手動重跑 build_embeddings.py 時會變，
runtime過程中不變，process生命週期內快取一次是安全的（跟 bm25_util.py 的
lazy singleton同一個道理）。
"""
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb

COLLECTOR_DIR = Path(__file__).resolve().parents[1] / "collector"
sys.path.insert(0, str(COLLECTOR_DIR))

import bm25_util  # noqa: E402  （collector/裡的現成模組，見上方說明）
from embedding_fn import get_embedding_function  # noqa: E402
# ponytail: 只為了拿CITY_ZH這個常數就import整支fetch_transit.py，會連帶執行它
# 頂層的load_dotenv()等初始化；且collector/沒package化，若之後有地方改用
# `collector.bm25_util`這種正式import，會跟這裡sys.path插入的bm25_util產生
# 兩份獨立singleton cache(Codex複核抓到)。upgrade path：collector/加__init__.py
# package化、CITY_ZH搬進無副作用的常數模組。單人demo專案風險低，先不動已測過的
# collector/。
from fetch_transit import CITY_ZH  # noqa: E402  （22個官方縣市中文名，重用不重打）

CHROMA_PATH = Path(__file__).resolve().parents[1] / "Data" / "chroma"
COLLECTION_NAME = "travel"
TOP_K_EACH = 50
RRF_K = 60  # RRF標準常數，不特別調

# 「新竹」這種去掉市/縣後的名字 -> 有可能對應的完整縣市名清單(新竹市/新竹縣都算)。
# 手動測試發現：查「花蓮/南投/屏東」這類縣市+全台通用熱門關鍵字(溫泉/海鮮/原住民
# 文化)時，其他縣市同類型文件的語意分數常贏過目標縣市較弱的匹配，city名稱在純文字
# 排名裡訊號太弱蓋不過關鍵字。修法：query裡偵測到明確城市名，就先用metadata過濾
# 限定候選範圍再排名(pre-filter，不是排完名次事後再篩)——跟FDE系統設計裡「RBAC先
# 過濾權限範圍再進vector search」是同一個道理，只是這裡過濾的是地理範圍不是權限。
#
# 22縣市全部手動測過、又找Codex獨立複核過後，累積踩了三個坑：
# 1) CITY_ZH用正式全形「臺」字(臺北市/臺中市/臺南市/臺東縣)，但口語幾乎都打「台」，
#    兩個是不同字元，字面比對不到——臺北/臺中/臺南三個直接測出命中率只剩1/5~4/5。
#    修法：比對前把「臺」正規化成「台」，字典key、查詢字串兩邊都轉，不用整份多存一份。
# 2) 「馬祖」是連江縣的俗名，字面上完全對不到，手動補一張別名表。
# 3) (Codex找到的)原本查「新竹市」這種明確講市/縣的query，也會因為比對用的是去掉
#    市/縣後的裸名「新竹」而被錯誤放寬成新竹市+新竹縣都算，「嘉義市」同樣的問題。
#    修法：每個裸名群組裡先檢查query有沒有literal寫出完整縣市名(如「新竹市」)，
#    有就只用那個精確匹配；沒有(query只寫裸名「新竹」)才維持原本兩個都算的寬鬆邏輯。
def _normalize_tai(text: str) -> str:
    return text.replace("臺", "台")


_BARE_CITY_TO_FULL: Dict[str, List[str]] = {}
for _full in CITY_ZH.values():
    _bare = _normalize_tai(_full.rstrip("市縣"))
    _BARE_CITY_TO_FULL.setdefault(_bare, []).append(_full)

_KNOWN_ALIASES: Dict[str, List[str]] = {
    "馬祖": ["連江縣"],
}


def detect_cities(query: str) -> Optional[List[str]]:
    """query裡有出現已知縣市名(含台/臺兩種寫法、少數俗名)就回傳對應的完整縣市名
    清單，都沒有就回傳None(不加城市過濾)。query寫出完整縣市名(如「新竹市」)時只
    精確匹配那一個；只寫裸名(如「新竹」)時新竹市/新竹縣都算(見上方註解)。"""
    normalized_query = _normalize_tai(query)
    matched: List[str] = []
    for bare, fulls in _BARE_CITY_TO_FULL.items():
        if bare not in normalized_query:
            continue
        exact_fulls = [f for f in fulls if _normalize_tai(f) in normalized_query]
        matched.extend(exact_fulls if exact_fulls else fulls)
    for alias, fulls in _KNOWN_ALIASES.items():
        if alias in query:
            matched.extend(fulls)
    return list(dict.fromkeys(matched)) or None


_collection_lock = threading.Lock()


def _get_collection():
    # Agent層讓LLM一次呼叫多個tool時，LangGraph會併發跑多個search_travel_info——
    # 2026-09-30實測撞到過："_collection is None"兩個thread同時判斷成立，各自
    # new一個chromadb.PersistentClient，導致底層Rust binding壞掉
    # (`RustBindingsAPI object has no attribute 'bindings'`)，工具報錯後agent沒
    # 察覺、自己編了沒有根據資料庫的答案——不是本地跑單一查詢測不出來的bug，一定
    # 要上鎖，跟rag.py::_get_doc_lookup()同一個double-checked locking模式。
    global _collection
    if _collection is None:
        with _collection_lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=str(CHROMA_PATH))
                _collection = client.get_collection(COLLECTION_NAME, embedding_function=get_embedding_function())
    return _collection


_collection = None


def _vector_search(query: str, k: int, allowed_cities: Optional[List[str]]) -> List[str]:
    where = {"city": {"$in": allowed_cities}} if allowed_cities else None
    res = _get_collection().query(query_texts=[query], n_results=k, where=where)
    return res["ids"][0]


def _bm25_search(query: str, k: int, allowed_cities: Optional[List[str]]) -> List[str]:
    idx = bm25_util.load()
    tokens = bm25_util.tokenize(query)
    if not tokens:
        # 空查詢/斷詞後沒半個token時，get_scores()會回傳全0分數，排序只是照原始
        # index順序選、不是相關性排名(Codex複核抓到)——直接回傳空，讓RRF只吃向量端。
        return []
    scores = idx["bm25"].get_scores(tokens)
    candidates = range(len(scores))
    if allowed_cities:
        allowed = set(allowed_cities)
        candidates = [i for i in candidates if idx["metadatas"][i].get("city") in allowed]
    ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
    return [idx["ids"][i] for i in ranked]


def rrf_fuse(rankings: List[List[str]], k: int = RRF_K) -> List[str]:
    """Reciprocal Rank Fusion：同一個doc在多份排名裡出現，名次越前貢獻分數越高，
    加總後排序。純函式，不吃外部資源，方便單獨測試。"""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


@lru_cache(maxsize=256)
def search(query: str, top_k: int = TOP_K_EACH) -> Tuple[str, ...]:
    """回傳RRF融合後的doc id清單（未rerank），最多top_k筆。query裡偵測到縣市名
    會先用metadata過濾限定範圍再排名，見detect_cities()說明。"""
    allowed_cities = detect_cities(query)
    vec_ids = _vector_search(query, TOP_K_EACH, allowed_cities)
    bm25_ids = _bm25_search(query, TOP_K_EACH, allowed_cities)
    fused = rrf_fuse([vec_ids, bm25_ids])
    return tuple(fused[:top_k])


def clear_cache() -> None:
    """重建索引(build_embeddings.py)後呼叫。原本只清`search()`的lru_cache，但
    Chroma collection handle跟bm25_util的singleton還是舊的，同一個process內重建
    索引不會生效(Codex複核抓到)——這裡一次把三層都重置。"""
    search.cache_clear()
    global _collection
    _collection = None
    bm25_util.reset_cache()
