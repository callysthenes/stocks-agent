"""
Dashboard Page 2 — Predictions: tracking and accuracy visualization.
"""
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Predictions — StocksAgent", page_icon="🎯", layout="wide")
st.title("🎯 Seguimiento de Predicciones")

try:
    from src.storage.mariadb_client import get_sync_db
    from src.models import Prediction, GroundTruth, Channel
    from sqlalchemy import select, desc, func

    with get_sync_db() as db:
        predictions = db.scalars(
            select(Prediction).order_by(desc(Prediction.created_at)).limit(200)
        ).all()

        # Accuracy stats
        evaluated = [p for p in predictions if p.is_evaluated and p.ground_truth]
        accurate = [p for p in evaluated if p.ground_truth.is_accurate]

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predicciones", len(predictions))
    col2.metric("Evaluadas", len(evaluated))
    col3.metric("Correctas", len(accurate))
    col4.metric(
        "Precisión Global",
        f"{len(accurate)/len(evaluated)*100:.1f}%" if evaluated else "N/A"
    )

    st.divider()

    if evaluated:
        # ── Accuracy by direction ─────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precisión por Dirección")
            direction_stats = {}
            for p in evaluated:
                d = str(p.predicted_direction)
                direction_stats.setdefault(d, {"correct": 0, "total": 0})
                direction_stats[d]["total"] += 1
                if p.ground_truth.is_accurate:
                    direction_stats[d]["correct"] += 1

            df_dir = pd.DataFrame([
                {"Dirección": d, "Precisión": v["correct"]/v["total"]*100, "Total": v["total"]}
                for d, v in direction_stats.items()
            ])
            fig = px.bar(df_dir, x="Dirección", y="Precisión", color="Dirección",
                        text=df_dir["Precisión"].apply(lambda x: f"{x:.0f}%"))
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cambio Real (evaluadas)")
            change_data = [
                float(p.ground_truth.actual_change_pct)
                for p in evaluated
                if p.ground_truth.actual_change_pct
            ]
            if change_data:
                fig2 = px.histogram(
                    pd.DataFrame({"Cambio %": change_data}),
                    x="Cambio %", nbins=30,
                    color_discrete_sequence=["#3498db"]
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Predictions table ─────────────────────────────────────────────────────
    st.subheader("Todas las Predicciones")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        ticker_filter = st.text_input("Filtrar por ticker", "").upper()
    with filter_col2:
        status_filter = st.selectbox("Estado", ["Todas", "Pendientes", "Evaluadas"])

    filtered = predictions
    if ticker_filter:
        filtered = [p for p in filtered if ticker_filter in p.ticker_symbol]
    if status_filter == "Pendientes":
        filtered = [p for p in filtered if not p.is_evaluated]
    elif status_filter == "Evaluadas":
        filtered = [p for p in filtered if p.is_evaluated]

    if filtered:
        rows = []
        for p in filtered[:100]:
            gt = p.ground_truth
            rows.append({
                "Ticker": p.ticker_symbol,
                "Canal": p.channel.name if p.channel else "Unknown",
                "Tipo": p.prediction_type,
                "Dirección": p.predicted_direction,
                "Precio Inicio": float(p.price_at_prediction) if p.price_at_prediction else None,
                "Precio Objetivo": float(p.target_price) if p.target_price else None,
                "Horizonte (d)": p.timeframe_days,
                "Fecha": p.prediction_date,
                "Evaluada": "✅" if p.is_evaluated else "⏳",
                "Correcta": ("✅" if gt.is_accurate else "❌") if gt else "—",
                "Cambio Real %": float(gt.actual_change_pct) if gt and gt.actual_change_pct else None,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No hay predicciones que coincidan con el filtro.")

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
