"""
LangGraph Supervisor — routes between research, analysis, and user agents.
Implements the supervisor pattern with circuit breaker (max iterations).
"""
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from loguru import logger

from src.agents.prompts import SUPERVISOR_SYSTEM
from src.agents.state import AgentState
from src.config import settings

MAX_ITERATIONS = 6

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            max_tokens=256,
        )
    return _llm


def supervisor_node(state: AgentState) -> Command:
    """
    Supervisor node — decides which agent to invoke next based on current state.
    Implements circuit breaker to prevent infinite loops.
    """
    iteration = state.get("iteration_count", 0)

    # Circuit breaker: if we've iterated too many times, force user_agent
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"Supervisor circuit breaker triggered at iteration {iteration}")
        return Command(
            goto="user_agent",
            update={"iteration_count": iteration + 1, "route_to": "user_agent"},
        )

    has_research = bool(state.get("research_results"))
    has_analysis = bool(state.get("analysis_results"))
    tickers = state.get("ticker_symbols", [])

    system_prompt = SUPERVISOR_SYSTEM.format(
        ticker_symbols=", ".join(tickers) if tickers else "ninguno identificado aún",
        has_research="sí" if has_research else "no",
        has_analysis="sí" if has_analysis else "no",
        iteration=iteration + 1,
        max_iterations=MAX_ITERATIONS,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.type if m.type != "ai" else "assistant", "content": m.content}
          for m in state["messages"][-10:]],  # Last 10 messages to stay within context
    ]

    try:
        response = get_llm().invoke(messages)
        next_agent = response.content.strip().lower().replace('"', "").replace("'", "")

        # Normalize the response to a valid agent name
        if "research" in next_agent:
            next_agent = "research_agent"
        elif "analysis" in next_agent or "analisis" in next_agent:
            next_agent = "analysis_agent"
        else:
            next_agent = "user_agent"

        logger.info(f"Supervisor routing to: {next_agent} (iteration {iteration + 1})")

        return Command(
            goto=next_agent,
            update={
                "route_to": next_agent,
                "iteration_count": iteration + 1,
            },
        )
    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        return Command(
            goto="user_agent",
            update={"route_to": "user_agent", "iteration_count": iteration + 1},
        )
