"""RAG查詢層對外唯一入口，agent的retrieval tool呼叫這支就好。

流程：hybrid_search.search()拿RRF融合後的候選id → 用bm25_util現成的
ids/texts/metadatas當id→文件內容的查表(建索引時是同一份docs，兩邊id保證對得
上，不用再查一次Chroma拿文字) → reranker_fn精選Top-5 → 組成給LLM讀的context。
"""
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

COLLECTOR_DIR = Path(__file__).resolve().parents[1] / "collector"
sys.path.insert(0, str(COLLECTOR_DIR))
import bm25_util  # noqa: E402

import hybrid_search
import reranker_fn

TOP_K_FINAL = 5


class Document(TypedDict):
    id: str
    text: str
    metadata: dict


# None當「還沒建過」的sentinel，跟「索引本身是空dict」分開——原本用`if not _doc_lookup`
# 判斷，兩種狀態會混在一起，索引真的是空的話每次查詢都會白跑一次bm25_util.load()
# (Codex複核抓到)。lock是防雙重初始化：多個request同時撞到第一次查詢時，若沒鎖，
# 後到的thread可能在字典還沒建完就讀到不完整的lookup，漏掉一大批candidate id。
_doc_lookup: Optional[Dict[str, Document]] = None
_doc_lookup_lock = threading.Lock()


def _get_doc_lookup() -> Dict[str, Document]:
    global _doc_lookup
    if _doc_lookup is None:
        with _doc_lookup_lock:
            if _doc_lookup is None:  # double-checked locking，避免每次查詢都搶鎖
                idx = bm25_util.load()
                ids, texts, metadatas = idx["ids"], idx["texts"], idx["metadatas"]
                if not (len(ids) == len(texts) == len(metadatas)):
                    raise RuntimeError(
                        f"bm25索引裡ids/texts/metadatas長度不一致"
                        f"(len={len(ids)}/{len(texts)}/{len(metadatas)})，"
                        "索引檔可能壞掉，重跑collector/build_embeddings.py"
                    )
                lookup: Dict[str, Document] = {}
                for doc_id, text, metadata in zip(ids, texts, metadatas):
                    lookup[doc_id] = {"id": doc_id, "text": text, "metadata": metadata}
                _doc_lookup = lookup
    return _doc_lookup


def clear_cache() -> None:
    """重建索引後呼叫，把這層的doc lookup跟底下hybrid_search/bm25/collection的
    快取一次清掉。"""
    global _doc_lookup
    _doc_lookup = None
    hybrid_search.clear_cache()


def query(user_query: str, top_k: int = TOP_K_FINAL) -> List[Document]:
    """回傳rerank後Top-k筆文件(含text/metadata)，供agent組context用。"""
    candidate_ids = hybrid_search.search(user_query)
    lookup = _get_doc_lookup()
    candidates = [lookup[cid] for cid in candidate_ids if cid in lookup]
    if candidate_ids and not candidates:
        # hybrid_search回傳了候選id，但doc lookup一筆都對不到——Chroma跟BM25
        # 索引不同步的訊號，不是「剛好沒結果」，靜默回傳[]會誤導agent(Codex複核抓到)。
        raise RuntimeError(
            f"hybrid_search回傳{len(candidate_ids)}筆候選id，但doc lookup一筆都對不到，"
            "Chroma/BM25索引可能不同步，重跑collector/build_embeddings.py"
        )
    if not candidates:
        return []
    texts = [c["text"] for c in candidates]
    top_indices = reranker_fn.rerank_indices(user_query, texts, top_k=top_k)
    return [candidates[i] for i in top_indices]


def format_context(docs: List[Document]) -> str:
    """把檢索到的文件轉成LLM可讀的context字串。"""
    lines = []
    for d in docs:
        meta = d["metadata"]
        loc = "、".join(p for p in [meta.get("city"), meta.get("district")] if p)
        lines.append(f"[{meta.get('type', '')}] {d['text']}（{loc}）")
    return "\n".join(lines)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "台中室內景點"
    results = query(q)
    print(f"查詢：{q}")
    print(format_context(results))
