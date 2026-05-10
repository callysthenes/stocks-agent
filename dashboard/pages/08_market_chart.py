"""
Dashboard Page 8 — Market Chart.
Professional TradingView-style OHLCV candlestick chart with:
  - 10 timeframe buttons (1m → Max)
  - Volume subplot
  - MA 20 / 50 / 200 overlays
  - Bollinger Bands overlay
  - Auto dark theme with weekend gap removal
  - Live price header
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Market Chart — StocksAgent",
    page_icon="📈",
    layout="wide",
)

from dashboard.auth import require_login  # noqa: E402
require_login()

# ── Palette (TradingView Dark) ────────────────────────────────────────────────
BG          = "#131722"
BG_PANEL    = "#1E2130"
GRID        = "#1E2130"
UP          = "#26a69a"
DOWN        = "#ef5350"
MA_COLORS   = {"20": "#F6C90E", "50": "#FF8C42", "200": "#9B59B6"}

# ── Timeframe definitions ─────────────────────────────────────────────────────
# Each key is a unique data combination: candle-interval × data-range
TIMEFRAMES: dict[str, dict] = OrderedDict([
    ("1m",  {"interval": "1m",  "period": "2d",  "label": "1m",  "intraday": True}),
    ("5m",  {"interval": "5m",  "period": "5d",  "label": "5m",  "intraday": True}),
    ("30m", {"interval": "30m", "period": "1mo", "label": "30m", "intraday": True}),
    ("1h",  {"interval": "1h",  "period": "3mo", "label": "1h",  "intraday": True}),
    ("1d",  {"interval": "1d",  "period": "6mo", "label": "1D",  "intraday": False}),
    ("1wk", {"interval": "1wk", "period": "2y",  "label": "1W",  "intraday": False}),
    ("1mo", {"interval": "1mo", "period": "5y",  "label": "1M",  "intraday": False}),
    ("1y",  {"interval": "1d",  "period": "1y",  "label": "1Y",  "intraday": False}),
    ("5y",  {"interval": "1wk", "period": "5y",  "label": "5Y",  "intraday": False}),
    ("max", {"interval": "1mo", "period": "max", "label": "Max", "intraday": False}),
])

# ── Popular tickers ───────────────────────────────────────────────────────────
POPULAR: dict[str, list[str]] = {
    "Ibex 35": ["SAN.MC", "TEF.MC", "IBE.MC", "ITX.MC", "CABK.MC", "BBVA.MC",
                "REP.MC", "ACS.MC", "ACX.MC", "GRF.MC"],
    "Wall St.": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD"],
    "Índices":  ["^IBEX", "^GSPC", "^IXIC", "^DJI", "^FTSE", "^GDAXI"],
    "ETF / FX": ["SPY", "QQQ", "GLD", "BTC-USD", "EURUSD=X"],
}

# ── Session-state defaults ─────────────────────────────────────────────────────
if "chart_ticker" not in st.session_state:
    st.session_state.chart_ticker = "SAN.MC"
if "chart_tf" not in st.session_state:
    st.session_state.chart_tf = "1d"

# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    """Download OHLCV from Yahoo Finance (cached 60 s)."""
    import yfinance as yf
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None

        # Flatten MultiIndex columns (yfinance ≥ 0.2.46 returns them for single tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Normalise the date column name across intraday/daily
        for col_name in ("Datetime", "Date", "index"):
            if col_name in df.columns:
                df = df.rename(columns={col_name: "dt"})
                break

        df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_localize(None)
        return df.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_live(ticker: str) -> dict:
    """Fetch live quote metadata."""
    import yfinance as yf
    try:
        fi = yf.Ticker(ticker).fast_info
        return {
            "price":      float(getattr(fi, "last_price",    None) or 0) or None,
            "prev_close": float(getattr(fi, "previous_close", None) or 0) or None,
            "currency":   getattr(fi, "currency", ""),
        }
    except Exception:
        return {}


# ── Chart builder ──────────────────────────────────────────────────────────────

def _build_chart(
    df: pd.DataFrame,
    ticker: str,
    tf_key: str,
    show_ma: bool,
    show_bb: bool,
    show_vol: bool,
) -> go.Figure:
    tf = TIMEFRAMES[tf_key]
    rows = 2 if show_vol else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.72, 0.28] if show_vol else [1.0],
    )

    # ── Candlestick ────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df["dt"],
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=ticker,
        increasing=dict(line=dict(color=UP,   width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
        whiskerwidth=0.6,
        hoverlabel=dict(bgcolor=BG_PANEL),
    ), row=1, col=1)

    close = df["Close"]

    # ── Moving averages ────────────────────────────────────────────────────────
    if show_ma:
        for p, color in MA_COLORS.items():
            period_int = int(p)
            if len(df) >= period_int:
                ma = close.rolling(period_int).mean()
                fig.add_trace(go.Scatter(
                    x=df["dt"], y=ma, name=f"MA {p}",
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"MA{p}: %{{y:.4f}}<extra></extra>",
                ), row=1, col=1)

    # ── Bollinger Bands (20, 2σ) ───────────────────────────────────────────────
    if show_bb and len(df) >= 20:
        roll = close.rolling(20)
        mid  = roll.mean()
        std  = roll.std()
        upper, lower = mid + 2 * std, mid - 2 * std

        fig.add_trace(go.Scatter(
            x=df["dt"], y=upper, name="BB+2σ",
            line=dict(color="#888", width=1, dash="dot"), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["dt"], y=lower, name="BB−2σ",
            line=dict(color="#888", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(136,136,136,0.08)",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["dt"], y=mid, name="BB Mid",
            line=dict(color="#AAAAAA", width=1, dash="dash"),
        ), row=1, col=1)

    # ── Volume ─────────────────────────────────────────────────────────────────
    if show_vol and "Volume" in df.columns:
        vol_colors = [
            UP if c >= o else DOWN
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df["dt"], y=df["Volume"],
            name="Vol", marker_color=vol_colors, marker_opacity=0.65,
            showlegend=False,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────────
    axis_style = dict(
        gridcolor=GRID, showgrid=True, zeroline=False,
        tickfont=dict(size=11, color="#9B9EA4"),
        linecolor="#2A2E3B",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=11, color="#CCC"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=640,
        margin=dict(l=5, r=5, t=40, b=5),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG_PANEL, font_size=12),
    )
    # Apply axis styles to all axes
    for axis in ("xaxis", "yaxis", "xaxis2", "yaxis2"):
        fig.update_layout(**{axis: axis_style})

    # Remove weekend gaps on daily+ charts
    if not tf["intraday"]:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ── Sidebar: quick-select popular tickers ─────────────────────────────────────
with st.sidebar:
    st.markdown("### 📌 Tickers rápidos")
    for group, tickers in POPULAR.items():
        with st.expander(group, expanded=False):
            cols = st.columns(2)
            for i, t in enumerate(tickers):
                if cols[i % 2].button(t, key=f"qs_{t}", use_container_width=True):
                    st.session_state.chart_ticker = t
                    st.session_state.chart_tf = "1d"
                    st.cache_data.clear()
                    st.rerun()
    st.divider()
    st.caption("Fuente: Yahoo Finance · yfinance")

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📈 Gráfico de Mercado")

# ── Ticker input + live price ─────────────────────────────────────────────────
col_input, col_price, col_btn = st.columns([2, 5, 1])

with col_input:
    raw = st.text_input(
        "ticker_input", label_visibility="collapsed",
        value=st.session_state.chart_ticker,
        placeholder="Ej: SAN.MC  AAPL  ^IBEX  BTC-USD",
    ).upper().strip()
    if raw and raw != st.session_state.chart_ticker:
        st.session_state.chart_ticker = raw
        st.cache_data.clear()
        st.rerun()

ticker = st.session_state.chart_ticker
live   = fetch_live(ticker)
price  = live.get("price")
prev   = live.get("prev_close")
cur    = live.get("currency", "")

with col_price:
    if price and prev and prev != 0:
        chg     = (price - prev) / prev * 100
        chg_abs = price - prev
        arrow   = "▲" if chg >= 0 else "▼"
        color   = UP if chg >= 0 else DOWN
        st.markdown(
            f"<span style='font-size:1.9rem;font-weight:700;color:#E8E8E8'>{price:,.4f}</span>"
            f"&nbsp;<span style='font-size:0.9rem;color:#888'>{cur}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='font-size:1.1rem;font-weight:600;color:{color}'>"
            f"{arrow} {chg_abs:+.4f} ({chg:+.2f}%)</span>",
            unsafe_allow_html=True,
        )
    elif price:
        st.markdown(
            f"<span style='font-size:1.9rem;font-weight:700'>{price:,.4f} {cur}</span>",
            unsafe_allow_html=True,
        )

with col_btn:
    if st.button("🔄", help="Refrescar datos en tiempo real", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Timeframe pills ───────────────────────────────────────────────────────────
tf_keys   = list(TIMEFRAMES.keys())
tf_labels = [TIMEFRAMES[k]["label"] for k in tf_keys]

# st.pills was introduced in Streamlit 1.37
selected_label = st.pills(
    "Timeframe",
    options=tf_labels,
    default=TIMEFRAMES[st.session_state.chart_tf]["label"],
    label_visibility="collapsed",
    key="tf_pills",
)

# Map label back to key
selected_tf = next(
    (k for k, v in TIMEFRAMES.items() if v["label"] == selected_label),
    st.session_state.chart_tf,
)
if selected_tf != st.session_state.chart_tf:
    st.session_state.chart_tf = selected_tf
    st.rerun()

# ── Overlay toggles ───────────────────────────────────────────────────────────
ov1, ov2, ov3, _spacer = st.columns([1, 1, 1, 6])
with ov1:
    show_ma  = st.toggle("MA 20/50/200", value=True)
with ov2:
    show_bb  = st.toggle("Bollinger", value=False)
with ov3:
    show_vol = st.toggle("Volumen", value=True)

# ── Fetch & render chart ──────────────────────────────────────────────────────
tf = TIMEFRAMES[st.session_state.chart_tf]

with st.spinner(f"Cargando {ticker}  ·  intervalo {tf['label']} …"):
    df = fetch_ohlcv(ticker, tf["interval"], tf["period"])

if df is None or df.empty:
    st.error(
        f"No hay datos para **{ticker}** con intervalo **{tf['label']}**. "
        "Comprueba el ticker (usa el sufijo correcto: `.MC` para BME, `.PA` para París, etc.)"
    )
    st.stop()

# ── OHLCV summary bar ─────────────────────────────────────────────────────────
last  = df.iloc[-1]
first = df.iloc[0]
pchg  = (float(last["Close"]) - float(first["Open"])) / float(first["Open"]) * 100

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Apertura",          f"{float(last['Open']):,.4f}")
m2.metric("Máximo",            f"{float(last['High']):,.4f}")
m3.metric("Mínimo",            f"{float(last['Low']):,.4f}")
m4.metric("Cierre",            f"{float(last['Close']):,.4f}")
if "Volume" in df.columns and pd.notna(last["Volume"]):
    vol = int(last["Volume"])
    m5.metric("Volumen",       f"{vol:,}")
else:
    m5.metric("Volumen",       "N/A")
m6.metric(f"Δ período ({tf['label']})", f"{pchg:+.2f}%")

# ── Main chart ────────────────────────────────────────────────────────────────
fig = _build_chart(df, ticker, st.session_state.chart_tf, show_ma, show_bb, show_vol)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": "png", "filename": f"{ticker}_{tf['label']}",
            "width": 1600, "height": 900,
        },
    },
)

st.caption(
    f"📡 Yahoo Finance via yfinance · {len(df):,} velas · "
    f"intervalo {tf['interval']} · período {tf['period']} · "
    f"caché 60 s · scroll para zoom · arrastra para desplazar"
)
