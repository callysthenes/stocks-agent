"""
Dashboard Page 7 — Predictions vs Actual: visualise how channel predictions
have performed against real market movements.
"""
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Pred vs Actual — StocksAgent", page_icon="📉", layout="wide"
)

from dashboard.auth import require_login  # noqa: E402
require_login()

st.title("📉 Predicciones vs Mercado Real")
st.caption("Contraste entre las predicciones de los canales y los movimientos reales del mercado")

try:
    from src.storage.mariadb_client import get_sync_db
    from src.models import Channel, Prediction, GroundTruth
    from sqlalchemy import select, desc
    from sqlalchemy.orm import selectinload

    # ── Load evaluated predictions with ground-truth ──────────────────────────
    with get_sync_db() as db:
        channel_map = {
            c.id: c.name
            for c in db.scalars(select(Channel)).all()
        }

        rows = db.scalars(
            select(Prediction)
            .options(selectinload(Prediction.ground_truth))
            .where(Prediction.is_evaluated == True)  # noqa: E712
            .order_by(desc(Prediction.prediction_date))
            .limit(500)
        ).all()

        records = [
            {
                "Canal": channel_map.get(p.channel_id, p.channel_id[:8]),
                "Ticker": p.ticker_symbol,
                "Dirección Predicha": str(p.predicted_direction),
                "Dirección Real": str(p.ground_truth.actual_direction) if p.ground_truth else None,
                "Cambio Real %": float(p.ground_truth.actual_change_pct)
                    if p.ground_truth and p.ground_truth.actual_change_pct else None,
                "Precio Inicio": float(p.price_at_prediction) if p.price_at_prediction else None,
                "Precio Eval": float(p.ground_truth.price_at_evaluation)
                    if p.ground_truth and p.ground_truth.price_at_evaluation else None,
                "Precio Objetivo": float(p.target_price) if p.target_price else None,
                "Correcta": p.ground_truth.is_accurate if p.ground_truth else None,
                "Fecha": p.prediction_date,
                "Tipo": str(p.prediction_type),
                "Confianza": p.confidence_score,
            }
            for p in rows
        ]

    if not records:
        st.info(
            "No hay predicciones evaluadas todavía. "
            "El agente de evaluación corre diariamente a las 22:00 UTC."
        )
        st.stop()

    df = pd.DataFrame(records).dropna(subset=["Cambio Real %"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Correcta Texto"] = df["Correcta"].map(
        {True: "✅ Correcta", False: "❌ Incorrecta", None: "❓ Sin datos"}
    )

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total = len(df)
    correct = df["Correcta"].sum() if "Correcta" in df else 0
    pct = correct / total * 100 if total else 0
    avg_change = df["Cambio Real %"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicciones evaluadas", total)
    k2.metric("Tasa de acierto global", f"{pct:.1f}%")
    k3.metric("Cambio real medio", f"{avg_change:+.2f}%")
    k4.metric("Canales analizados", df["Canal"].nunique())

    st.divider()

    # ── Filter controls ───────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_channels = st.multiselect(
            "Filtrar por canal", sorted(df["Canal"].unique()), default=[]
        )
    with col_f2:
        sel_tickers = st.multiselect(
            "Filtrar por ticker", sorted(df["Ticker"].unique()), default=[]
        )
    with col_f3:
        date_range = st.date_input(
            "Rango de fechas",
            value=(
                df["Fecha"].min().date() if not df["Fecha"].isna().all() else None,
                df["Fecha"].max().date() if not df["Fecha"].isna().all() else None,
            ),
        )

    dff = df.copy()
    if sel_channels:
        dff = dff[dff["Canal"].isin(sel_channels)]
    if sel_tickers:
        dff = dff[dff["Ticker"].isin(sel_tickers)]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        dff = dff[
            (dff["Fecha"] >= pd.Timestamp(start_d))
            & (dff["Fecha"] <= pd.Timestamp(end_d))
        ]

    if dff.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        st.stop()

    st.divider()

    # ── Chart 1: Scatter — Cambio real % por predicción con color de acierto ─
    st.subheader("🎯 Distribución de cambio real (por predicción)")
    fig_scatter = px.strip(
        dff,
        x="Canal",
        y="Cambio Real %",
        color="Correcta Texto",
        hover_data=["Ticker", "Dirección Predicha", "Dirección Real", "Fecha"],
        color_discrete_map={
            "✅ Correcta": "#2ecc71",
            "❌ Incorrecta": "#e74c3c",
            "❓ Sin datos": "#95a5a6",
        },
        stripmode="overlay",
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.update_layout(height=420, xaxis_title="", yaxis_title="Cambio real (%)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Chart 2: Accuracy over time (rolling 30-day window) ──────────────────
    st.subheader("📈 Evolución de la precisión en el tiempo")
    df_time = dff.dropna(subset=["Fecha", "Correcta"]).copy()
    df_time = df_time.sort_values("Fecha")
    if len(df_time) >= 5:
        df_time["Correcta_num"] = df_time["Correcta"].astype(int)
        df_time["Precisión (30d rolling)"] = (
            df_time["Correcta_num"].rolling(window=min(30, len(df_time)), min_periods=1).mean() * 100
        )
        fig_time = px.line(
            df_time,
            x="Fecha",
            y="Precisión (30d rolling)",
            color="Canal",
            markers=True,
        )
        fig_time.add_hline(y=50, line_dash="dot", line_color="orange",
                           annotation_text="50% baseline")
        fig_time.update_layout(height=380, yaxis_range=[0, 105], yaxis_title="Precisión %")
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Necesita al menos 5 predicciones evaluadas para mostrar la evolución temporal.")

    st.divider()

    # ── Chart 3: Predicted price vs actual price (waterfall per ticker) ───────
    st.subheader("💹 Precio predicho vs real por ticker")
    df_price = dff.dropna(subset=["Precio Inicio", "Precio Eval"]).copy()
    if not df_price.empty:
        ticker_sel = st.selectbox(
            "Selecciona ticker",
            sorted(df_price["Ticker"].unique()),
            key="ticker_waterfall",
        )
        df_t = df_price[df_price["Ticker"] == ticker_sel].sort_values("Fecha")

        fig_candle = go.Figure()
        fig_candle.add_trace(go.Scatter(
            x=df_t["Fecha"],
            y=df_t["Precio Inicio"],
            mode="markers+lines",
            name="Precio al predecir",
            line=dict(color="#3498db", dash="dot"),
            marker=dict(size=8),
        ))
        fig_candle.add_trace(go.Scatter(
            x=df_t["Fecha"],
            y=df_t["Precio Eval"],
            mode="markers+lines",
            name="Precio real (eval)",
            line=dict(color="#2ecc71"),
            marker=dict(size=8),
        ))
        if df_t["Precio Objetivo"].notna().any():
            fig_candle.add_trace(go.Scatter(
                x=df_t["Fecha"],
                y=df_t["Precio Objetivo"],
                mode="markers",
                name="Precio objetivo",
                marker=dict(symbol="star", size=14, color="#f39c12"),
            ))
        fig_candle.update_layout(
            height=380,
            xaxis_title="Fecha predicción",
            yaxis_title="Precio",
            title=f"{ticker_sel} — precio al predecir vs al evaluar",
        )
        st.plotly_chart(fig_candle, use_container_width=True)
    else:
        st.info("No hay datos de precio disponibles para el scatter de precios.")

    st.divider()

    # ── Chart 4: Accuracy heatmap by channel × direction ─────────────────────
    st.subheader("🗂️ Heatmap de aciertos: Canal × Dirección predicha")
    df_heat = dff.dropna(subset=["Correcta"]).copy()
    if not df_heat.empty:
        pivot = (
            df_heat.groupby(["Canal", "Dirección Predicha"])["Correcta"]
            .mean()
            .mul(100)
            .round(1)
            .unstack(fill_value=0)
        )
        fig_heat = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            aspect="auto",
            labels=dict(color="Precisión %"),
        )
        fig_heat.update_layout(height=max(250, len(pivot) * 60))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("📋 Datos detallados"):
        st.dataframe(
            dff[[
                "Fecha", "Canal", "Ticker", "Tipo",
                "Dirección Predicha", "Dirección Real",
                "Cambio Real %", "Precio Inicio", "Precio Eval",
                "Precio Objetivo", "Confianza", "Correcta Texto",
            ]].sort_values("Fecha", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.exception(e)
