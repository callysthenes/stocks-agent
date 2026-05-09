"""
LangGraph agent state definitions.
Shared state object passed between all nodes in the graph.
"""
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state for the multi-agent graph."""
    # Conversation messages (append-only via add_messages)
    messages: Annotated[list[BaseMessage], add_messages]
    # Original user query
    user_query: str
    # Tickers identified in the query or extracted from research
    ticker_symbols: list[str]
    # Results from Research Agent (list of RAG result dicts)
    research_results: list[dict[str, Any]]
    # Results from Analysis Agent (dict with technical/fundamental/news)
    analysis_results: dict[str, Any]
    # Final formatted report (set by User Agent)
    final_report: str | None
    # Control flow: which agent to route to next
    route_to: str
    # Whether to send via Telegram
    send_telegram: bool
    # Optional: video_id if triggered by a specific video
    video_id: str | None
    # Number of supervisor iterations (circuit breaker)
    iteration_count: int
