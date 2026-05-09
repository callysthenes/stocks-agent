"""
Research Agent — searches YouTube transcript database via LlamaIndex RAG.
Finds what channels have said about specific tickers, predictions, context.
"""
import re

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from loguru import logger

from src.agents.prompts import RESEARCH_AGENT_SYSTEM
from src.agents.state import AgentState
from src.agents.tools.prediction_tools import (
    get_active_predictions_for_ticker,
    get_prediction_history_for_ticker,
)
from src.agents.tools.rag_tools import search_ticker_mentions, search_transcripts
from src.config import settings

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.1,
            max_tokens=3000,
        )
    return _llm


def research_node(state: AgentState) -> Command:
    """
    Research Agent node.
    Searches transcripts and prediction DB, synthesizes findings.
    """
    query = state.get("user_query", "")
    tickers = state.get("ticker_symbols", [])

    logger.info(f"Research Agent: query='{query[:60]}' tickers={tickers}")

    # ── Step 1: Extract tickers from query if not already identified ──────────
    if not tickers:
        tickers = _extract_tickers_from_query(query)

    research_results = []

    # ── Step 2: RAG search for each ticker ───────────────────────────────────
    for ticker in tickers[:5]:  # Limit to 5 tickers per query
        rag_result = search_ticker_mentions(ticker, top_k=8)
        active_preds = get_active_predictions_for_ticker(ticker)
        historical_preds = get_prediction_history_for_ticker(ticker, limit=5)

        research_results.append({
            "ticker": ticker,
            "rag_answer": rag_result.get("answer", ""),
            "rag_sources": rag_result.get("sources", []),
            "active_predictions": active_preds,
            "historical_predictions": historical_preds,
        })

    # ── Step 3: General context if no specific tickers ────────────────────────
    if not tickers and query:
        general_result = search_transcripts(query, top_k=10)
        research_results.append({
            "ticker": "general",
            "rag_answer": general_result.get("answer", ""),
            "rag_sources": general_result.get("sources", []),
            "active_predictions": [],
            "historical_predictions": [],
        })

    # ── Step 4: Synthesize with LLM ───────────────────────────────────────────
    research_summary = _synthesize_research(query, research_results, tickers)

    response_msg = AIMessage(
        content=f"[Research Agent]\n{research_summary}"
    )

    return Command(
        goto="supervisor",
        update={
            "messages": [response_msg],
            "ticker_symbols": tickers,
            "research_results": research_results,
        },
    )


def _extract_tickers_from_query(query: str) -> list[str]:
    """
    Extract likely ticker symbols from a user query.

    For video-alert queries the caller embeds a structured section
    "Tickers mencionados: <list|ninguno identificado>" — we parse that
    section first so we don't accidentally match words from an ALL-CAPS
    YouTube title.
    """
    # ── Parse structured ticker section if present ────────────────────────────
    if "Tickers mencionados:" in query:
        section = query.split("Tickers mencionados:", 1)[1].split(".")[0].strip()
        if "ninguno" in section.lower():
            return []
        # Pull only the real ticker symbols from this section
        pattern = r"\b([A-Z]{1,6}(?:\.[A-Z]{1,3})?(?:-[A-Z]{2,4})?)\b"
        candidates = re.findall(pattern, section)
        stop_words = {"CEO", "CFO", "ETF", "USA", "EUR", "USD", "AI", "OK",
                      "AND", "OR", "NOT", "THE", "FOR", "WITH"}
        return [c for c in candidates if c not in stop_words and len(c) >= 2]

    # ── Fallback: regex extraction for free-form interactive queries ──────────
    pattern = r"\b([A-Z]{2,5}(?:\.[A-Z]{1,3})?(?:-[A-Z]{2,4})?)\b"
    candidates = re.findall(pattern, query)
    # Extended stop-word list to reduce false positives
    stop_words = {
        "CEO", "CFO", "ETF", "USA", "EUR", "USD", "AI", "OK", "AND", "OR",
        "NOT", "THE", "FOR", "WITH", "FROM", "INTO", "OVER", "UNDER", "IRAN",
        "WAR", "IS", "AT", "IN", "BY", "ON", "TO", "OF", "AS", "AN",
        "EU", "UK", "UN", "NATO", "GDP", "IPO", "PE", "EPS", "ROE", "ROI",
        "YTD", "YOY", "QOQ", "MOM", "ATH", "ATL", "HODL", "DCA",
    }
    return [c for c in candidates if c not in stop_words]


def _synthesize_research(
    query: str, results: list[dict], tickers: list[str]
) -> str:
    """Use LLM to synthesize research results into a coherent summary."""
    if not results:
        return "No se encontró información relevante en la base de datos de transcripciones."

    context_parts = []
    for r in results:
        ticker = r.get("ticker", "")
        rag_answer = r.get("rag_answer", "")
        active_preds = r.get("active_predictions", [])
        hist_preds = r.get("historical_predictions", [])

        part = f"**{ticker}:**\n{rag_answer}\n"
        if active_preds:
            part += f"Predicciones activas: {len(active_preds)} pendientes de evaluación\n"
        if hist_preds:
            accurate = sum(1 for p in hist_preds if p.get("was_accurate"))
            part += f"Historial: {accurate}/{len(hist_preds)} predicciones correctas\n"
        context_parts.append(part)

    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": RESEARCH_AGENT_SYSTEM},
        {"role": "user", "content": f"Consulta del usuario: {query}\n\nResultados encontrados:\n{context}"},
    ]

    try:
        response = get_llm().invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Research synthesis error: {e}")
        return context
