"""共用的本地免費 embedding function：包裝 fastembed 的 BAAI/bge-small-zh-v1.5，
供 collector/build_embeddings.py（建索引）和 code/rag.py（查詢）共用同一份實作——
兩邊要用同一個模型算出來的向量才能互相比對，寫兩份容易出現模型/參數不一致的坑。

不用 OpenAI embedding API：使用者在網頁上打字查詢時，每次查詢都要重新算一次
embedding，若每次都打付費 API，訪客越多越花錢；bge-small-zh-v1.5 只有約90MB、
純 CPU 可跑（onnxruntime，不需要 torch），中文語意品質對這個資料集夠用，模型只需
下載一次，之後永久免費、可離線跑。
"""
from pathlib import Path
from typing import List, Union

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# ponytail: fastembed的cache_dir預設落在系統temp目錄(macOS會定期清掉)，不是穩定路徑。
# 2026-10-03發現：temp目錄裡的模型快取被清掉後，下次呼叫會卡在向HuggingFace Hub重新
# 抓檔案(~90MB)，網路一不穩就整個卡住——跟「本地免費模型、不依賴外部」的設計初衷相違背。
# 改成專案內的固定路徑，模型只會真的下載一次，之後永久離線可用。
CACHE_DIR = Path(__file__).resolve().parents[1] / "Data" / "fastembed_cache"

_model = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    return _model


class LocalChineseEmbeddingFunction:
    """符合 chromadb 的 EmbeddingFunction 協定。

    `__call__` 用來 embed 建索引的文件、`embed_query` 用來 embed 使用者查詢——
    bge 系列模型對「查詢」跟「文件」用不同的 instruction prefix 效果才好，
    fastembed 的 passage_embed()/query_embed() 已經內建處理這個差異，不用自己拼字串。
    """

    def __call__(self, input: List[str]) -> List:
        return [v.tolist() for v in _get_model().passage_embed(list(input))]

    def embed_query(self, input: Union[str, List[str]]) -> List:
        texts = [input] if isinstance(input, str) else list(input)
        return [v.tolist() for v in _get_model().query_embed(texts)]

    @staticmethod
    def name() -> str:
        return "local-bge-small-zh"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "LocalChineseEmbeddingFunction":
        return LocalChineseEmbeddingFunction()


def get_embedding_function() -> LocalChineseEmbeddingFunction:
    return LocalChineseEmbeddingFunction()
