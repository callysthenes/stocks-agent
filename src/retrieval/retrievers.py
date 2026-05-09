"""
Custom retrievers for hybrid search combining ChromaDB vector search
with MariaDB metadata filtering.
"""
from typing import Any, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from loguru import logger


def build_retriever(
    index: VectorStoreIndex,
    similarity_top_k: int = 8,
    channel_id: str | None = None,
) -> VectorIndexRetriever:
    """
    Build a vector retriever with optional metadata filtering.

    Args:
        index: LlamaIndex VectorStoreIndex backed by ChromaDB.
        similarity_top_k: Number of top similar nodes to retrieve.
        channel_id: Filter to specific channel (matches ChromaDB metadata).

    Returns:
        Configured VectorIndexRetriever.
    """
    filters = None
    if channel_id:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="channel_id", value=channel_id)]
        )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
        filters=filters,
    )
    return retriever


def retrieve_for_ticker(
    index: VectorStoreIndex,
    ticker_symbol: str,
    n_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve transcript chunks most relevant to a specific ticker symbol.
    Uses both direct ticker search and semantic similarity.
    """
    query = f"análisis de {ticker_symbol} recomendación inversión acción precio"

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=n_results,
    )

    query_bundle = QueryBundle(query_str=query)
    nodes: list[NodeWithScore] = retriever.retrieve(query_bundle)

    results = []
    for node in nodes:
        meta = node.metadata or {}
        results.append({
            "video_id": meta.get("video_id", ""),
            "channel_name": meta.get("channel_name", ""),
            "video_title": meta.get("video_title", ""),
            "published_at": meta.get("published_at", ""),
            "score": node.score or 0.0,
            "text": node.text or "",
        })

    logger.debug(f"Retrieved {len(results)} chunks for ticker {ticker_symbol}")
    return results
