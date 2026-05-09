"""
LlamaIndex RAG index builder.
Creates a VectorStoreIndex backed by ChromaDB using bge-m3 embeddings.
"""
from functools import lru_cache
from typing import TYPE_CHECKING

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.openai import utils as _openai_utils
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

# LlamaIndex validates model names against OpenAI's catalogue.
# Register DeepSeek model names so the context-window lookup succeeds.
_openai_utils.ALL_AVAILABLE_MODELS.setdefault("deepseek-chat", 131072)
_openai_utils.ALL_AVAILABLE_MODELS.setdefault("deepseek-reasoner", 131072)
_openai_utils.CHAT_MODELS.setdefault("deepseek-chat", 131072)
_openai_utils.CHAT_MODELS.setdefault("deepseek-reasoner", 131072)

from src.config import settings

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex as VectorStoreIndexType


def _configure_llama_settings() -> None:
    """Configure LlamaIndex global settings (embedding model + LLM)."""
    # Use the same bge-m3 model as the ingestion pipeline
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            cache_folder="/app/.cache/sentence_transformers",
        )
        Settings.embed_model = embed_model
    except Exception as e:
        logger.warning(f"Could not load HuggingFace embedding model for LlamaIndex: {e}")

    # LLM for LlamaIndex query synthesis (DeepSeek V3, OpenAI-compatible)
    Settings.llm = LlamaOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_base_url,
        temperature=0,
        max_tokens=4096,
    )
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap


@lru_cache(maxsize=1)
def get_vector_index() -> "VectorStoreIndexType":
    """
    Get or create the LlamaIndex VectorStoreIndex backed by ChromaDB.
    Cached as singleton — safe because ChromaDB is persistent.
    """
    _configure_llama_settings()

    chroma_client = chromadb.HttpClient(
        host=settings.chromadb_host,
        port=settings.chromadb_port,
    )
    collection = chroma_client.get_or_create_collection(
        name=settings.chromadb_collection,
        metadata={"hnsw:space": "cosine"},
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    logger.info(
        f"LlamaIndex VectorStoreIndex ready "
        f"({collection.count()} documents in ChromaDB)"
    )
    return index


def invalidate_index_cache() -> None:
    """Force reload of the index on next access (call after bulk ingestion)."""
    get_vector_index.cache_clear()
