"""
User Agent — formats the final report and delivers it via Telegram.
"""
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command
from loguru import logger

from src.agents.prompts import USER_AGENT_SYSTEM
from src.agents.state import AgentState
from src.config import settings

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.2,
            max_tokens=3000,
        )
    return _llm


def user_agent_node(state: AgentState) -> Command:
    """
    User Agent node — formats and delivers the final response.
    Always terminates the graph (goes to END).
    """
    query = state.get("user_query", "")
    analysis_results = state.get("analysis_results", {})
    research_results = state.get("research_results", [])
    tickers = state.get("ticker_symbols", [])
    video_id = state.get("video_id")  # set only for automated video-alert runs

    # Compile all available content
    content_parts = []

    # Analysis text (primary content)
    if analysis_results.get("analysis_text"):
        content_parts.append(analysis_results["analysis_text"])

    # Research summaries (supplementary)
    for r in research_results:
        if r.get("rag_answer") and r["ticker"] != "general":
            if not any(r["ticker"] in p for p in content_parts):
                content_parts.append(
                    f"\n**Contexto de canales para {r['ticker']}:**\n{r['rag_answer']}"
                )

    full_content = "\n\n".join(content_parts) if content_parts else "No se encontró información suficiente."

    # ── Choose format based on context ───────────────────────────────────────
    dashboard_url = settings.dashboard_url
    tickers_str = ", ".join(tickers) if tickers else "N/A"

    if video_id:
        # Automated video-alert: use the structured video-report format
        format_instruction = (
            "Usa el formato de alerta de vídeo:\n"
            "📹 *Nuevo Vídeo Analizado*\nCanal: ...\nVídeo: ...\nTickers: ...\n[resumen]\n"
            f"🔗 {dashboard_url}"
        )
    elif tickers:
        # Ticker analysis: structured financial report
        format_instruction = (
            f"Usa el formato de análisis de ticker para {tickers_str}. "
            "Incluye análisis técnico, fundamental, y qué dicen los canales. "
            f"Termina con el link {dashboard_url} y el disclaimer."
        )
    else:
        # Conversational / general query — plain answer, no video/ticker template
        format_instruction = (
            "El usuario hizo una consulta general (no hay ticker específico). "
            "Responde directamente a la pregunta de forma conversacional, clara y concisa. "
            "No uses plantillas de informes de vídeo ni de ticker. "
            f"Puedes mencionar el dashboard al final: {dashboard_url}"
        )

    format_prompt = (
        f"Consulta del usuario: {query}\n"
        f"Tickers analizados: {tickers_str}\n\n"
        f"Instrucción de formato: {format_instruction}\n\n"
        f"Contenido a formatear:\n{full_content[:6000]}"
    )

    messages = [
        {"role": "system", "content": USER_AGENT_SYSTEM},
        {"role": "user", "content": format_prompt},
    ]

    try:
        response = get_llm().invoke(messages)
        final_report = response.content
    except Exception as e:
        logger.error(f"User Agent formatting error: {e}")
        final_report = full_content[:4000]

    # Ensure disclaimer is present
    if "asesoramiento financiero" not in final_report.lower():
        final_report += "\n\n⚠️ *Este análisis es únicamente informativo y no constituye asesoramiento financiero.*"

    # Truncate for Telegram limit
    if len(final_report) > 4096:
        final_report = final_report[:4000] + f"\n\n... [Continúa en {dashboard_url}]"

    logger.info(f"User Agent: final report ready ({len(final_report)} chars)")

    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=final_report)],
            "final_report": final_report,
        },
    )
