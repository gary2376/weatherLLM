"""共用reranker：fastembed內建TextCrossEncoder包BAAI/bge-reranker-base，
跟collector/embedding_fn.py選本地免費模型同一個理由——查詢時即時rerank，
若打付費API，訪客越多越花錢；純CPU跑，模型只需下載一次。
"""
from pathlib import Path
from typing import List

from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"

# ponytail: 跟collector/embedding_fn.py同一個理由——cache_dir預設落在系統temp目錄，
# 2026-10-03發現temp快取被清掉會卡在重新向HuggingFace Hub抓檔案，改成專案內固定路徑。
CACHE_DIR = Path(__file__).resolve().parents[1] / "Data" / "fastembed_cache"

_model = None


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    return _model


def rerank_indices(query: str, documents: List[str], top_k: int = 5) -> List[int]:
    """回傳documents裡分數最高的top_k個的原始index，高到低排序。"""
    if not documents or top_k <= 0:
        # documents空的話沒東西可rerank；top_k<=0的話`ranked[:top_k]`是負數slice，
        # 會回傳「除了最後幾筆以外的全部」而不是空清單(Codex複核抓到)。
        return []
    scores = list(_get_model().rerank(query, documents))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]
