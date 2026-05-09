"""
ChromaDB client — collection management and vector operations.
"""
from typing import Any

import chromadb
from chromadb import Collection
from loguru import logger

from src.config import settings

_client: chromadb.HttpClient | None = None
_collection: Collection | None = None


def get_chroma_client() -> chromadb.HttpClient:
    """Get or create ChromaDB HTTP client (singleton)."""
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
        )
        logger.info(f"ChromaDB client connected to {settings.chromadb_host}:{settings.chromadb_port}")
    return _client


def get_collection() -> Collection:
    """Get or create the main video_transcripts collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.chromadb_collection,
            metadata={
                "hnsw:space": "cosine",
                "description": "Video transcript chunks with bge-m3 embeddings",
            },
        )
        logger.info(f"ChromaDB collection '{settings.chromadb_collection}' ready ({_collection.count()} docs)")
    return _collection


def upsert_chunks(
    chunk_ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """Upsert transcript chunks with embeddings into ChromaDB."""
    collection = get_collection()
    collection.upsert(
        ids=chunk_ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.debug(f"Upserted {len(chunk_ids)} chunks to ChromaDB")


def query_similar(
    query_embedding: list[float],
    n_results: int = 10,
    where: dict | None = None,
) -> dict[str, Any]:
    """Query ChromaDB for similar chunks using a pre-computed embedding."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return results


def delete_video_chunks(video_id: str) -> None:
    """Delete all chunks belonging to a video (e.g., on reprocessing)."""
    collection = get_collection()
    collection.delete(where={"video_id": video_id})
    logger.debug(f"Deleted ChromaDB chunks for video {video_id}")


def get_collection_stats() -> dict[str, Any]:
    """Return basic stats about the collection."""
    collection = get_collection()
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }


def check_chroma_connection() -> bool:
    """Health check for ChromaDB connectivity."""
    try:
        client = get_chroma_client()
        client.heartbeat()
        return True
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        return False
