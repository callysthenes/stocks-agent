"""
Analysis Agent — performs financial analysis using yfinance + pandas-ta.
Technical, fundamental, news analysis + prediction tracking.
"""
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from loguru import logger

from src.agents.prompts import ANALYSIS_AGENT_SYSTEM
from src.agents.state import AgentState
from src.agents.tools.financial_tools import (
    get_fundamentals,
    get_recent_news,
    get_technical_indicators,
)
from src.agents.tools.prediction_tools import (
    get_active_predictions_for_ticker,
    get_channel_accuracy_summary,
)
from src.config import settings

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            max_tokens=4096,
        )
    return _llm


def analysis_node(state: AgentState) -> Command:
    """
    Analysis Agent node.
    Performs full financial analysis for identified tickers.
    """
    tickers = state.get("ticker_symbols", [])
    query = state.get("user_query", "")
    research_results = state.get("research_results", [])

    if not tickers:
        logger.warning("Analysis Agent: no tickers to analyze")
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content="[Analysis Agent] No hay tickers identificados para analizar.")],
                "analysis_results": {},
            },
        )

    logger.info(f"Analysis Agent: analyzing {tickers}")

    # ── Gather financial data for each ticker ─────────────────────────────────
    ticker_data: dict[str, dict] = {}
    for ticker in tickers[:5]:  # Max 5 tickers
        technical = get_technical_indicators(ticker)
        fundamentals = get_fundamentals(ticker)
        news = get_recent_news(ticker, max_items=5)
        active_predictions = get_active_predictions_for_ticker(ticker)

        ticker_data[ticker] = {
            "technical": technical,
            "fundamentals": fundamentals,
            "news": news,
            "active_predictions": active_predictions,
        }

    # Channel accuracy summary for context
    channel_accuracy = get_channel_accuracy_summary()

    # ── Research context from previous agent ──────────────────────────────────
    research_context = ""
    for r in research_results:
        if r.get("rag_answer"):
            research_context += f"\n**Contexto de vídeos ({r['ticker']}):**\n{r['rag_answer']}\n"

    # ── Synthesize analysis with DeepSeek ────────────────────────────────────
    analysis_text = _generate_analysis(
        query=query,
        ticker_data=ticker_data,
        research_context=research_context,
        channel_accuracy=channel_accuracy,
    )

    analysis_results = {
        "ticker_data": ticker_data,
        "channel_accuracy": channel_accuracy,
        "analysis_text": analysis_text,
    }

    return Command(
        goto="supervisor",
        update={
            "messages": [AIMessage(content=f"[Analysis Agent]\n{analysis_text}")],
            "analysis_results": analysis_results,
        },
    )


def _generate_analysis(
    query: str,
    ticker_data: dict,
    research_context: str,
    channel_accuracy: list[dict],
) -> str:
    """Generate the full financial analysis report using DeepSeek."""

    # Build structured data prompt
    data_parts = []
    for ticker, data in ticker_data.items():
        tech = data.get("technical", {})
        fund = data.get("fundamentals", {})
        news = data.get("news", [])
        active_preds = data.get("active_predictions", [])

        part = f"""
### {ticker} — {fund.get('company_name', ticker)}

**Datos técnicos:**
- Precio actual: {tech.get('current_price', 'N/A')} {fund.get('currency', '')}
- Cambio 1 día: {tech.get('price_change_1d_pct', 'N/A')}%
- Cambio 1 mes: {tech.get('price_change_1mo_pct', 'N/A')}%
- Cambio 1 año: {tech.get('price_change_1y_pct', 'N/A')}%
- RSI (14): {tech.get('rsi_14', 'N/A')}
- MACD: {tech.get('macd', 'N/A')} (histograma: {tech.get('macd_histogram', 'N/A')})
- SMA 20/50/200: {tech.get('sma_20', 'N/A')} / {tech.get('sma_50', 'N/A')} / {tech.get('sma_200', 'N/A')}
- Tendencia: {tech.get('trend', 'N/A')}
- Máx/Mín 52 semanas: {tech.get('52w_high', 'N/A')} / {tech.get('52w_low', 'N/A')}

**Datos fundamentales:**
- Sector: {fund.get('sector', 'N/A')} | Industria: {fund.get('industry', 'N/A')}
- P/E: {fund.get('pe_ratio', 'N/A')} | P/E forward: {fund.get('forward_pe', 'N/A')}
- P/B: {fund.get('price_to_book', 'N/A')} | P/S: {fund.get('price_to_sales', 'N/A')}
- Margen neto: {fund.get('profit_margin', 'N/A')} | ROE: {fund.get('return_on_equity', 'N/A')}
- Deuda/Equity: {fund.get('debt_to_equity', 'N/A')}
- Precio objetivo analistas: {fund.get('analyst_target_price', 'N/A')}
- Recomendación: {fund.get('analyst_recommendation', 'N/A')}

**Noticias recientes:**
{chr(10).join(f"- [{n['published_at']}] {n['title']}" for n in news) if news else "Sin noticias recientes"}

**Predicciones activas de los canales:** {len(active_preds)} pendientes
"""
        data_parts.append(part)

    channel_acc_text = ""
    if channel_accuracy:
        channel_acc_text = "\n**Precisión de los canales:**\n"
        for ch in channel_accuracy[:5]:
            acc = ch.get("accuracy_pct")
            channel_acc_text += (
                f"- {ch['channel_name']}: {acc:.1f}% "
                f"({ch['accurate_predictions']}/{ch['total_predictions']})\n"
                if acc else f"- {ch['channel_name']}: Sin datos suficientes\n"
            )

    prompt = f"""El usuario pregunta: "{query}"

Datos financieros disponibles:
{"".join(data_parts)}

{research_context}

{channel_acc_text}

Genera un análisis completo y estructurado como analista senior. 
Incluye análisis técnico, fundamental, contexto de los canales de YouTube, y una conclusión.
Añade el disclaimer al final."""

    messages = [
        {"role": "system", "content": ANALYSIS_AGENT_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        response = get_llm().invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Analysis generation error: {e}")
        return f"Error generando análisis: {e}\n\nDatos disponibles:\n{''.join(data_parts)}"
