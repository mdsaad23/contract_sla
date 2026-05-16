"""
Embedder — wraps fastembed (ONNX Runtime) in a llama-index BaseEmbedding adapter.

Improvements over the previous PyTorch-based HuggingFaceEmbedding:
1. ONNX Runtime is 2-3x faster than PyTorch on CPU for this model size
2. Supports GPU acceleration via DirectML execution provider (AMD/Intel GPU on Windows)
3. Module-level singleton — model is loaded ONCE per process, not twice per contract
4. Batch embedding by default — all chunks in one ONNX call
"""

import os
import threading
from typing import List, ClassVar

import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from fastembed import TextEmbedding

from config import CHROMA_DB_PATH

LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


def _resolve_providers() -> List[str]:
    """Decide which ONNX execution providers to try, in priority order.

    USE_GPU=0 in .env forces CPU only.
    """
    if os.getenv("USE_GPU", "1") == "0":
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        chosen: List[str] = []
        for p in ("DmlExecutionProvider", "CUDAExecutionProvider"):
            if p in available:
                chosen.append(p)
        chosen.append("CPUExecutionProvider")
        return chosen
    except Exception:
        return ["CPUExecutionProvider"]


# ── Singleton fastembed model ─────────────────────────────────────────────────
_MODEL: TextEmbedding = None
_MODEL_LOCK = threading.Lock()
_MODEL_PROVIDER: str = ""


def _get_model() -> TextEmbedding:
    global _MODEL, _MODEL_PROVIDER
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                providers = _resolve_providers()
                _MODEL = TextEmbedding(model_name=LOCAL_EMBED_MODEL, providers=providers)
                _MODEL_PROVIDER = providers[0]
                print(f"[embedder] Loaded {LOCAL_EMBED_MODEL} | provider: {_MODEL_PROVIDER}")
    return _MODEL


def get_active_provider() -> str:
    """Return the highest-priority provider chosen at model load time."""
    _get_model()  # ensure loaded
    return _MODEL_PROVIDER


# ── LlamaIndex BaseEmbedding adapter ───────────────────────────────────────────

class FastEmbedAdapter(BaseEmbedding):
    """Thin llama-index adapter over fastembed. Uses module-level singleton."""

    model_name: str = LOCAL_EMBED_MODEL

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedAdapter"

    def _get_query_embedding(self, query: str) -> List[float]:
        return list(next(_get_model().query_embed([query])))

    def _get_text_embedding(self, text: str) -> List[float]:
        return list(next(_get_model().embed([text])))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [list(e) for e in _get_model().embed(texts)]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


# Singleton adapter — reused across all contracts in the process
_ADAPTER: FastEmbedAdapter = None


def get_embed_model() -> FastEmbedAdapter:
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = FastEmbedAdapter()
        _get_model()  # warm up + log provider
    return _ADAPTER


# ── Per-document vector index ─────────────────────────────────────────────────

def build_per_doc_index(contract_id: str, chunks: List[str]) -> VectorStoreIndex:
    embed_model = get_embed_model()

    chroma_client   = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection_name = f"contract_{contract_id[:40]}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection      = chroma_client.get_or_create_collection(collection_name)
    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = [Document(text=chunk) for chunk in chunks]
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
