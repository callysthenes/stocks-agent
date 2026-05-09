"""FastAPI agent router — direct LLM query endpoint."""
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents import run_agent

router = APIRouter()


class AgentQuery(BaseModel):
    query: str
    send_telegram: bool = False


@router.post("/query")
async def query_agent(payload: AgentQuery):
    """Run a query through the multi-agent pipeline."""
    if not payload.query.strip():
        raise HTTPException(400, detail="Query cannot be empty")
    if len(payload.query) > 2000:
        raise HTTPException(400, detail="Query too long (max 2000 chars)")

    result = await run_agent(
        query=payload.query,
        send_telegram=payload.send_telegram,
    )
    return {
        "final_report": result.get("final_report"),
        "ticker_symbols": result.get("ticker_symbols", []),
        "iteration_count": result.get("iteration_count", 0),
    }
