"""Download and initialize the fastembed models used by the application."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from collector.embedding_fn import get_embedding_function  # noqa: E402
from reranker_fn import rerank_indices  # noqa: E402


def main() -> None:
    embedding = get_embedding_function()
    embedding.embed_query("天氣模型預熱")
    rerank_indices("台北天氣", ["台北今天是晴天。"], top_k=1)
    print("fastembed embedding and reranker models are ready")


if __name__ == "__main__":
    main()
