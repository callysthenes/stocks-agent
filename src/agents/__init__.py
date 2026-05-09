"""
LangGraph multi-agent graph construction.
Wires together supervisor, research, analysis, and user agents.
"""
from functools import lru_cache

from langgraph.graph import START, StateGraph

from src.agents.analysis_agent import analysis_node
from src.agents.research_agent import research_node
from src.agents.state import AgentState
from src.agents.supervisor import supervisor_node
from src.agents.user_agent import user_agent_node
from src.observability.langfuse_setup import get_langfuse_callback


def build_graph():
    """Build and compile the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("analysis_agent", analysis_node)
    graph.add_node("user_agent", user_agent_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Edges: all agents return to supervisor (or END via user_agent) ────────
    # Routing is handled by Command(goto=...) inside each node
    # LangGraph follows the Command.goto field automatically

    return graph.compile()


@lru_cache(maxsize=1)
def get_agent_graph():
    """Get the compiled agent graph (singleton)."""
    return build_graph()


async def run_agent(
    query: str,
    send_telegram: bool = False,
    video_id: str | None = None,
) -> dict:
    """
    Run the full agent pipeline for a user query.

    Args:
        query: User question or task in Spanish.
        send_telegram: Whether the User Agent should deliver via Telegram.
        video_id: Optional video ID if triggered by a video alert.

    Returns:
        Final agent state dict.
    """
    from langchain_core.messages import HumanMessage

    graph = get_agent_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "ticker_symbols": [],
        "research_results": [],
        "analysis_results": {},
        "final_report": None,
        "route_to": "supervisor",
        "send_telegram": send_telegram,
        "video_id": video_id,
        "iteration_count": 0,
    }

    config: dict = {}
    callback = get_langfuse_callback()
    if callback:
        config["callbacks"] = [callback]

    result = await graph.ainvoke(initial_state, config=config)
    return result
