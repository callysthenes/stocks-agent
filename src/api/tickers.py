"""FastAPI tickers router — financial analysis endpoint."""
from fastapi import APIRouter, HTTPException

from src.agents.tools.financial_tools import (
    get_fundamentals,
    get_recent_news,
    get_technical_indicators,
)
from src.agents.tools.prediction_tools import (
    get_active_predictions_for_ticker,
    get_prediction_history_for_ticker,
)
from src.storage import repository
from src.storage.mariadb_client import get_sync_db

router = APIRouter()


@router.get("/{symbol}")
async def get_ticker_analysis(symbol: str):
    """Full analysis for a ticker: technical + fundamental + news + channel mentions."""
    symbol = symbol.upper()
    technical = get_technical_indicators(symbol)
    if "error" in technical:
        raise HTTPException(404, detail=f"No data found for {symbol}")

    return {
        "ticker": symbol,
        "technical": technical,
        "fundamentals": get_fundamentals(symbol),
        "news": get_recent_news(symbol, max_items=5),
        "active_predictions": get_active_predictions_for_ticker(symbol),
        "prediction_history": get_prediction_history_for_ticker(symbol, limit=10),
        "channel_mentions": _get_channel_mentions(symbol),
    }


def _get_channel_mentions(ticker_symbol: str) -> list[dict]:
    with get_sync_db() as db:
        mentions = repository.get_ticker_mentions(db, ticker_symbol, limit=20)
        return [
            {
                "video_id": m.video_id,
                "sentiment": m.sentiment,
                "context_snippet": m.context_snippet,
                "created_at": str(m.created_at),
                "exchange": m.exchange,
            }
            for m in mentions
        ]
