"""
Financial analysis tools for the Analysis Agent.
Wraps yfinance + pandas-ta to provide structured financial data.
"""
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import pandas_ta as ta
    _PANDAS_TA_AVAILABLE = True
except ImportError:
    _PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available — technical indicators will be limited")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=False)
def get_price_history(ticker_symbol: str, period: str = "6mo") -> pd.DataFrame | None:
    """Fetch OHLCV price history from yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period, auto_adjust=True)
        if hist.empty:
            logger.warning(f"No price history for {ticker_symbol}")
            return None
        return hist
    except Exception as e:
        logger.error(f"Failed to fetch price history for {ticker_symbol}: {e}")
        return None


def get_technical_indicators(ticker_symbol: str) -> dict[str, Any]:
    """
    Calculate key technical indicators for a ticker.
    Returns a structured dict with all indicators.
    """
    hist = get_price_history(ticker_symbol, period="1y")
    if hist is None or hist.empty or len(hist) < 20:
        return {"error": f"Insufficient data for {ticker_symbol}"}

    result: dict[str, Any] = {
        "ticker": ticker_symbol,
        "current_price": round(float(hist["Close"].iloc[-1]), 4),
        "price_change_1d_pct": round(
            float((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100), 2
        ),
        "price_change_1mo_pct": round(
            float((hist["Close"].iloc[-1] - hist["Close"].iloc[-22]) / hist["Close"].iloc[-22] * 100), 2
        ) if len(hist) >= 22 else None,
        "price_change_1y_pct": round(
            float((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100), 2
        ),
        "volume_avg_20d": int(hist["Volume"].tail(20).mean()),
        "52w_high": round(float(hist["High"].tail(252).max()), 4),
        "52w_low": round(float(hist["Low"].tail(252).min()), 4),
    }

    if _PANDAS_TA_AVAILABLE:
        try:
            # RSI
            hist.ta.rsi(length=14, append=True)
            rsi_col = [c for c in hist.columns if "RSI" in c]
            if rsi_col:
                result["rsi_14"] = round(float(hist[rsi_col[0]].iloc[-1]), 2)

            # MACD
            hist.ta.macd(fast=12, slow=26, signal=9, append=True)
            macd_col = [c for c in hist.columns if "MACD_12_26_9" == c]
            macdh_col = [c for c in hist.columns if "MACDh_12_26_9" == c]
            if macd_col:
                result["macd"] = round(float(hist[macd_col[0]].iloc[-1]), 4)
            if macdh_col:
                result["macd_histogram"] = round(float(hist[macdh_col[0]].iloc[-1]), 4)

            # Moving averages
            for period in [20, 50, 200]:
                if len(hist) >= period:
                    hist.ta.sma(length=period, append=True)
                    col = f"SMA_{period}"
                    if col in hist.columns:
                        result[f"sma_{period}"] = round(float(hist[col].iloc[-1]), 4)

            # Bollinger Bands
            hist.ta.bbands(length=20, append=True)
            bb_upper = [c for c in hist.columns if "BBU_20" in c]
            bb_lower = [c for c in hist.columns if "BBL_20" in c]
            if bb_upper:
                result["bb_upper_20"] = round(float(hist[bb_upper[0]].iloc[-1]), 4)
            if bb_lower:
                result["bb_lower_20"] = round(float(hist[bb_lower[0]].iloc[-1]), 4)

            # Trend direction
            price = result["current_price"]
            sma50 = result.get("sma_50")
            sma200 = result.get("sma_200")
            if sma50 and sma200:
                if price > sma50 > sma200:
                    result["trend"] = "bullish"
                elif price < sma50 < sma200:
                    result["trend"] = "bearish"
                else:
                    result["trend"] = "mixed"

        except Exception as e:
            logger.warning(f"Technical indicator calculation error for {ticker_symbol}: {e}")

    return result


def get_fundamentals(ticker_symbol: str) -> dict[str, Any]:
    """Fetch fundamental/valuation data from yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}

        return {
            "ticker": ticker_symbol,
            "company_name": info.get("longName") or info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            # Valuation
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            # Growth
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "earnings_growth_yoy": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            # Financial health
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "free_cashflow": info.get("freeCashflow"),
            "return_on_equity": info.get("returnOnEquity"),
            # Dividends
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            # Analyst
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
        }
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker_symbol}: {e}")
        return {"error": str(e)}


def get_recent_news(ticker_symbol: str, max_items: int = 5) -> list[dict[str, Any]]:
    """Fetch recent news headlines for a ticker from yfinance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        news_items = ticker.news or []
        result = []
        for item in news_items[:max_items]:
            pub_time = item.get("providerPublishTime")
            pub_str = (
                datetime.fromtimestamp(pub_time, tz=timezone.utc).strftime("%Y-%m-%d")
                if pub_time
                else ""
            )
            result.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published_at": pub_str,
            })
        return result
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker_symbol}: {e}")
        return []


def get_current_price(ticker_symbol: str) -> float | None:
    """Get the most recent closing price for a ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        fast_info = ticker.fast_info
        price = fast_info.last_price or fast_info.previous_close
        return float(price) if price else None
    except Exception:
        return None
