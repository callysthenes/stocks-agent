"""
Embedding generator using sentence-transformers with BAAI/bge-m3.
bge-m3 is the best multilingual model for Spanish financial text (100+ languages).
Model is loaded once per worker process and cached as a module-level singleton.
"""
from typing import TYPE_CHECKING

from loguru import logger

from src.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_embed_model: "SentenceTransformer | None" = None


def get_embedding_model() -> "SentenceTransformer":
    """Load and cache the embedding model (singleton per process)."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run this in the worker container (Dockerfile.worker)."
            ) from e

        logger.info(f"Loading embedding model '{settings.embedding_model}'...")
        _embed_model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
            cache_folder="/app/.cache/sentence_transformers",
        )
        dim = _embed_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dim={dim}, device={settings.embedding_device})")
    return _embed_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using bge-m3.

    bge-m3 does NOT require instruction prefixes (unlike e5 models).
    Normalizes embeddings for cosine similarity.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
        show_progress_bar=len(texts) > 10,
        convert_to_numpy=True,
    )

    logger.debug(f"Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """
    Generate a single embedding for a query string.
    Same normalization as embed_texts for consistent cosine similarity.
    """
    results = embed_texts([query])
    return results[0] if results else []


def get_embedding_dim() -> int:
    """Return the embedding dimensionality."""
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension() or 1024


def unload_model() -> None:
    """Explicitly unload the model to free GPU memory."""
    global _embed_model
    if _embed_model is not None:
        del _embed_model
        _embed_model = None
        logger.info("Embedding model unloaded")
