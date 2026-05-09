"""
LlamaIndex query engine — builds retriever-augmented query engines
for querying transcript content about specific tickers or topics.
"""
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from loguru import logger

from src.retrieval.index_builder import get_vector_index
from src.retrieval.retrievers import build_retriever

# Spanish-language QA prompt for the response synthesizer
_QA_PROMPT_TEMPLATE = PromptTemplate(
    """Eres un analista financiero senior experto en mercados bursátiles.
Usa ÚNICAMENTE la información de los fragmentos de vídeo proporcionados para responder.
Si la información no está en los fragmentos, dilo claramente.

Contexto de vídeos de YouTube sobre inversión:
---------------------
{context_str}
---------------------

Pregunta: {query_str}

Responde en español, de forma precisa y estructurada. 
Si se mencionan tickers concretos, inclúyelos en tu respuesta.
Cita el canal y el vídeo cuando sea posible.

Respuesta:"""
)

_REFINE_PROMPT_TEMPLATE = PromptTemplate(
    """Eres un analista financiero senior. Refina la respuesta existente usando nuevo contexto adicional.

Respuesta existente: {existing_answer}

Nuevo contexto:
---------------------
{context_msg}
---------------------

Refina la respuesta (o mantenla si el nuevo contexto no aporta información adicional):"""
)


def build_query_engine(
    similarity_top_k: int = 8,
    channel_id: str | None = None,
    ticker_filter: str | None = None,
) -> RetrieverQueryEngine:
    """
    Build a LlamaIndex query engine with optional filtering.

    Args:
        similarity_top_k: Number of chunks to retrieve.
        channel_id: Filter results to a specific channel (optional).
        ticker_filter: Not used at query engine level — filtering happens in retriever.

    Returns:
        A configured RetrieverQueryEngine.
    """
    index = get_vector_index()
    retriever = build_retriever(
        index=index,
        similarity_top_k=similarity_top_k,
        channel_id=channel_id,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=_QA_PROMPT_TEMPLATE,
        refine_template=_REFINE_PROMPT_TEMPLATE,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )


def query_transcripts(
    query: str,
    similarity_top_k: int = 8,
    channel_id: str | None = None,
) -> dict:
    """
    Execute a RAG query against the transcript database.

    Returns:
        dict with 'answer', 'sources', and 'node_count'.
    """
    try:
        engine = build_query_engine(
            similarity_top_k=similarity_top_k,
            channel_id=channel_id,
        )
        response = engine.query(query)

        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                meta = node.metadata or {}
                sources.append({
                    "video_id": meta.get("video_id", ""),
                    "channel_name": meta.get("channel_name", ""),
                    "video_title": meta.get("video_title", ""),
                    "published_at": meta.get("published_at", ""),
                    "score": node.score or 0.0,
                    "snippet": node.text[:300] if node.text else "",
                })

        return {
            "answer": str(response),
            "sources": sources,
            "node_count": len(sources),
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "answer": f"Error executing query: {e}",
            "sources": [],
            "node_count": 0,
        }
