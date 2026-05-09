"""
Dashboard Page 4 — Channel Accuracy: comparative performance between channels.
"""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Channel Accuracy — StocksAgent", page_icon="🏆", layout="wide")

from dashboard.auth import require_login  # noqa: E402
require_login()

st.title("🏆 Precisión de Canales")
st.caption("Comparativa de la fiabilidad de predicciones por canal de YouTube")

try:
    from src.storage.mariadb_client import get_sync_db
    from src.models import Channel, ChannelAccuracy, Prediction, GroundTruth
    from sqlalchemy import select, desc

    with get_sync_db() as db:
        accuracy_rows = db.scalars(
            select(ChannelAccuracy).order_by(desc(ChannelAccuracy.accuracy_pct))
        ).all()

        channel_names = {
            c.id: c.name
            for c in db.scalars(select(Channel)).all()
        }

    if not accuracy_rows:
        st.info(
            "Aún no hay datos de precisión disponibles. "
            "Los datos aparecerán conforme se evalúen las predicciones de los canales."
        )
        st.stop()

    # ── Accuracy leaderboard ──────────────────────────────────────────────────
    st.subheader("🥇 Clasificación de Canales")
    rows = []
    for r in accuracy_rows:
        name = channel_names.get(r.channel_id, r.channel_id)
        acc = float(r.accuracy_pct) if r.accuracy_pct else None
        rows.append({
            "Canal": name,
            "Total Predicciones": r.total_predictions,
            "Correctas": r.accurate_predictions,
            "Precisión %": acc,
            "Retorno Medio %": float(r.avg_return_pct) if r.avg_return_pct else None,
        })

    df = pd.DataFrame(rows)

    # Color code by accuracy
    def color_accuracy(val):
        if val is None:
            return ""
        if val >= 60:
            return "background-color: #d5f5e3"
        elif val >= 40:
            return "background-color: #fef9e7"
        else:
            return "background-color: #fadbd8"

    st.dataframe(
        df.style.applymap(color_accuracy, subset=["Precisión %"]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Bar chart comparison ───────────────────────────────────────────────────
    df_chart = df.dropna(subset=["Precisión %"])
    if not df_chart.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Precisión por Canal")
            fig = px.bar(
                df_chart.sort_values("Precisión %", ascending=True),
                x="Precisión %",
                y="Canal",
                orientation="h",
                color="Precisión %",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                text=df_chart.sort_values("Precisión %")["Precisión %"].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_layout(yaxis_title="", height=max(300, len(df_chart) * 50))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Volumen de Predicciones")
            fig2 = px.bar(
                df_chart.sort_values("Total Predicciones", ascending=True),
                x="Total Predicciones",
                y="Canal",
                orientation="h",
                color="Total Predicciones",
                color_continuous_scale="Blues",
            )
            fig2.update_layout(yaxis_title="", height=max(300, len(df_chart) * 50))
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Channel detail drill-down ─────────────────────────────────────────────
    st.subheader("Detalle por Canal")
    selected_channel = st.selectbox("Selecciona un canal", list(channel_names.values()))
    channel_id = next((cid for cid, name in channel_names.items() if name == selected_channel), None)

    if channel_id:
        with get_sync_db() as db:
            channel_preds = db.scalars(
                select(Prediction)
                .where(Prediction.channel_id == channel_id, Prediction.is_evaluated == True)  # noqa: E712
                .order_by(desc(Prediction.prediction_date))
                .limit(50)
            ).all()

        if channel_preds:
            pred_rows = []
            for p in channel_preds:
                gt = p.ground_truth
                pred_rows.append({
                    "Ticker": p.ticker_symbol,
                    "Tipo": p.prediction_type,
                    "Dirección": p.predicted_direction,
                    "Fecha": p.prediction_date,
                    "Precio Inicio": float(p.price_at_prediction) if p.price_at_prediction else None,
                    "Correcta": "✅" if gt.is_accurate else "❌" if gt else "—",
                    "Cambio Real %": float(gt.actual_change_pct) if gt and gt.actual_change_pct else None,
                })
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No hay predicciones evaluadas para este canal.")

except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.exception(e)
