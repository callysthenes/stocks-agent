"""
Langfuse observability setup.
Provides callback handlers for LangChain/LangGraph and LlamaIndex.
"""
from functools import lru_cache
from typing import Any

from loguru import logger

from src.config import settings


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Get Langfuse client (singleton). Returns None if not configured."""
    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.debug("Langfuse not configured — observability disabled")
        return None
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info(f"Langfuse client initialized (host={settings.langfuse_host})")
        return client
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        return None


def get_langfuse_callback():
    """Get Langfuse LangChain callback handler. Returns None if not configured."""
    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return handler
    except Exception as e:
        logger.debug(f"Could not create Langfuse callback handler: {e}")
        return None


def trace_event(
    name: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict | None = None,
) -> None:
    """Create a manual Langfuse trace event for non-LangChain operations."""
    client = get_langfuse_client()
    if not client:
        return
    try:
        trace = client.trace(name=name, metadata=metadata or {})
        trace.generation(
            name=name,
            input=str(input_data)[:1000] if input_data else None,
            output=str(output_data)[:1000] if output_data else None,
        )
    except Exception as e:
        logger.debug(f"Langfuse trace error (non-critical): {e}")


def flush_langfuse() -> None:
    """Flush pending Langfuse events (call before process exit)."""
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass
