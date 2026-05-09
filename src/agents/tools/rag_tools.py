"""RAG tools — wraps LlamaIndex query engine for agent use."""
from typing import Any

from loguru import logger

from src.retrieval.query_engine import query_transcripts


def search_transcripts(query: str, channel_id: str | None = None, top_k: int = 8) -> dict[str, Any]:
    """
    Search video transcripts using semantic similarity.

    Args:
        query: Natural language query in Spanish.
        channel_id: Optionally restrict search to a specific channel.
        top_k: Number of results to return.

    Returns:
        Dict with 'answer' and 'sources'.
    """
    logger.debug(f"RAG query: '{query[:80]}' (channel={channel_id}, top_k={top_k})")
    return query_transcripts(query=query, similarity_top_k=top_k, channel_id=channel_id)


def search_ticker_mentions(ticker_symbol: str, top_k: int = 10) -> dict[str, Any]:
    """Search specifically for mentions of a ticker symbol in transcripts."""
    query = (
        f"análisis opinión recomendación {ticker_symbol} "
        f"acción precio target comprar vender"
    )
    return query_transcripts(query=query, similarity_top_k=top_k)
