"""
Dashboard Page 3 — Ticker Analysis: per-ticker deep dive with charts.
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Ticker Analysis — StocksAgent", page_icon="🔍", layout="wide")

from dashboard.auth import require_login  # noqa: E402
require_login()

st.title("🔍 Análisis de Ticker")

ticker_input = st.text_input("Introduce el símbolo del ticker (ej: AAPL, SAN.MC, MSFT)", "").strip().upper()

if not ticker_input:
    st.info("Introduce un ticker para comenzar el análisis.")
    st.markdown("""
**Ejemplos:**
- Bolsa de Madrid: `SAN.MC` (Santander), `ITX.MC` (Inditex), `TEF.MC` (Telefónica)
- NYSE/NASDAQ: `AAPL`, `TSLA`, `MSFT`, `NVDA`
- Cripto: `BTC-USD`, `ETH-USD`
""")
    st.stop()

st.subheader(f"📊 Análisis: {ticker_input}")

with st.spinner(f"Cargando datos para {ticker_input}..."):
    try:
        from src.agents.tools.financial_tools import (
            get_price_history, get_technical_indicators,
            get_fundamentals, get_recent_news
        )
        from src.agents.tools.prediction_tools import (
            get_active_predictions_for_ticker, get_prediction_history_for_ticker
        )
        from src.storage.mariadb_client import get_sync_db
        from src.storage import repository

        hist = get_price_history(ticker_input, period="1y")
        technical = get_technical_indicators(ticker_input)
        fundamentals = get_fundamentals(ticker_input)
        news = get_recent_news(ticker_input, max_items=5)
        active_preds = get_active_predictions_for_ticker(ticker_input)
        hist_preds = get_prediction_history_for_ticker(ticker_input, limit=10)

        with get_sync_db() as db:
            channel_mentions = repository.get_ticker_mentions(db, ticker_input, limit=20)

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

# ── Price metrics ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Precio Actual", f"{technical.get('current_price', 'N/A')} {fundamentals.get('currency', '')}")
col2.metric("Cambio 1D", f"{technical.get('price_change_1d_pct', 'N/A')}%")
col3.metric("RSI (14)", technical.get("rsi_14", "N/A"))
col4.metric("Tendencia", technical.get("trend", "N/A").capitalize() if technical.get("trend") else "N/A")
col5.metric("Recomendación", fundamentals.get("analyst_recommendation", "N/A"))

st.divider()

# ── OHLCV Chart ───────────────────────────────────────────────────────────────
if hist is not None and not hist.empty:
    st.subheader("Gráfico de Precios (1 año)")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="OHLCV",
    ))

    # Add SMA overlays
    for period, color in [(20, "#3498db"), (50, "#e67e22"), (200, "#e74c3c")]:
        if f"SMA_{period}" in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist[f"SMA_{period}"],
                name=f"SMA {period}", line=dict(color=color, width=1),
            ))

    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Technical + Fundamental side by side ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Análisis Técnico")
    tech_data = {
        "SMA 20": technical.get("sma_20"),
        "SMA 50": technical.get("sma_50"),
        "SMA 200": technical.get("sma_200"),
        "RSI (14)": technical.get("rsi_14"),
        "MACD": technical.get("macd"),
        "MACD Histograma": technical.get("macd_histogram"),
        "BB Superior": technical.get("bb_upper_20"),
        "BB Inferior": technical.get("bb_lower_20"),
        "Máx 52 sem.": technical.get("52w_high"),
        "Mín 52 sem.": technical.get("52w_low"),
    }
    df_tech = pd.DataFrame(
        [(k, v) for k, v in tech_data.items() if v is not None],
        columns=["Indicador", "Valor"]
    )
    st.dataframe(df_tech, use_container_width=True, hide_index=True)

with col2:
    st.subheader("💼 Análisis Fundamental")
    fund_data = {
        "Empresa": fundamentals.get("company_name"),
        "Sector": fundamentals.get("sector"),
        "P/E Trailing": fundamentals.get("pe_ratio"),
        "P/E Forward": fundamentals.get("forward_pe"),
        "P/B": fundamentals.get("price_to_book"),
        "P/S": fundamentals.get("price_to_sales"),
        "EV/EBITDA": fundamentals.get("ev_to_ebitda"),
        "Margen neto": fundamentals.get("profit_margin"),
        "ROE": fundamentals.get("return_on_equity"),
        "Deuda/Equity": fundamentals.get("debt_to_equity"),
        "Yield dividendo": fundamentals.get("dividend_yield"),
        "Precio objetivo": fundamentals.get("analyst_target_price"),
    }
    df_fund = pd.DataFrame(
        [(k, v) for k, v in fund_data.items() if v is not None],
        columns=["Métrica", "Valor"]
    )
    st.dataframe(df_fund, use_container_width=True, hide_index=True)

st.divider()

# ── News ──────────────────────────────────────────────────────────────────────
st.subheader("📰 Noticias Recientes")
if news:
    for n in news:
        st.markdown(f"- [{n['published_at']}] **[{n['title']}]({n['link']})** — {n['publisher']}")
else:
    st.info("No hay noticias recientes disponibles.")

st.divider()

# ── Channel mentions ──────────────────────────────────────────────────────────
st.subheader("📺 Menciones en Canales de YouTube")
if channel_mentions:
    sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
    for m in channel_mentions[:10]:
        emoji = sentiment_emoji.get(str(m.sentiment), "⚪")
        st.markdown(f"{emoji} *{str(m.created_at)[:10]}* — {m.context_snippet or 'Sin contexto'}")
else:
    st.info("No hay menciones de este ticker en los vídeos procesados.")

# ── Predictions ───────────────────────────────────────────────────────────────
if active_preds or hist_preds:
    st.divider()
    st.subheader("🎯 Predicciones")

    tab1, tab2 = st.tabs(["Activas", "Historial"])
    with tab1:
        if active_preds:
            st.dataframe(pd.DataFrame(active_preds), use_container_width=True)
        else:
            st.info("No hay predicciones activas para este ticker.")

    with tab2:
        if hist_preds:
            st.dataframe(pd.DataFrame(hist_preds), use_container_width=True)
        else:
            st.info("Sin historial de predicciones.")
