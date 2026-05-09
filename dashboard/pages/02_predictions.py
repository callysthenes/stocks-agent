"""
Dashboard Page 2 — Predictions: tracking and accuracy visualization.
"""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Predictions — StocksAgent", page_icon="🎯", layout="wide")

from dashboard.auth import require_login  # noqa: E402
require_login()

st.title("🎯 Seguimiento de Predicciones")

try:
    from src.storage.mariadb_client import get_sync_db
    from src.models import Prediction, GroundTruth, Channel
    from sqlalchemy import select, desc
    from sqlalchemy.orm import selectinload

    with get_sync_db() as db:
        predictions = db.scalars(
            select(Prediction)
            .options(
                selectinload(Prediction.ground_truth),
                selectinload(Prediction.channel),
            )
            .order_by(desc(Prediction.created_at))
            .limit(200)
        ).all()

        # Convert to plain dicts while session is open to avoid lazy-load issues
        pred_rows = []
        for p in predictions:
            gt = p.ground_truth
            pred_rows.append({
                "id": p.id,
                "ticker_symbol": p.ticker_symbol,
                "channel_name": p.channel.name if p.channel else "Unknown",
                "prediction_type": str(p.prediction_type),
                "recommendation": getattr(p, "recommendation", None),
                "predicted_direction": str(p.predicted_direction),
                "price_at_prediction": float(p.price_at_prediction) if p.price_at_prediction else None,
                "entry_price": float(p.entry_price) if getattr(p, "entry_price", None) else None,
                "target_price": float(p.target_price) if p.target_price else None,
                "stop_loss": float(p.stop_loss) if getattr(p, "stop_loss", None) else None,
                "is_long_term": getattr(p, "is_long_term", None),
                "confidence_score": getattr(p, "confidence_score", None),
                "analyst_reasoning": getattr(p, "analyst_reasoning", None),
                "timeframe_days": p.timeframe_days,
                "prediction_date": p.prediction_date,
                "is_evaluated": p.is_evaluated,
                "gt_is_accurate": gt.is_accurate if gt else None,
                "gt_actual_change_pct": float(gt.actual_change_pct) if gt and gt.actual_change_pct else None,
                "gt_actual_direction": str(gt.actual_direction) if gt and gt.actual_direction else None,
            })

    evaluated = [r for r in pred_rows if r["is_evaluated"] and r["gt_is_accurate"] is not None]
    accurate = [r for r in evaluated if r["gt_is_accurate"]]

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predicciones", len(pred_rows))
    col2.metric("Evaluadas", len(evaluated))
    col3.metric("Correctas", len(accurate))
    col4.metric(
        "Precisión Global",
        f"{len(accurate)/len(evaluated)*100:.1f}%" if evaluated else "N/A"
    )

    st.divider()

    if evaluated:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Precisión por Dirección")
            direction_stats: dict = {}
            for r in evaluated:
                d = r["predicted_direction"]
                direction_stats.setdefault(d, {"correct": 0, "total": 0})
                direction_stats[d]["total"] += 1
                if r["gt_is_accurate"]:
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
            st.subheader("Cambio Real % (evaluadas)")
            change_data = [r["gt_actual_change_pct"] for r in evaluated if r["gt_actual_change_pct"] is not None]
            if change_data:
                fig2 = px.histogram(
                    pd.DataFrame({"Cambio %": change_data}),
                    x="Cambio %", nbins=30, color_discrete_sequence=["#3498db"]
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Predictions table ─────────────────────────────────────────────────────
    st.subheader("Todas las Predicciones")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        ticker_filter = st.text_input("Filtrar por ticker", "").upper()
    with filter_col2:
        status_filter = st.selectbox("Estado", ["Todas", "Pendientes", "Evaluadas"])
    with filter_col3:
        direction_filter = st.selectbox("Dirección", ["Todas", "up", "down", "neutral"])

    filtered = pred_rows
    if ticker_filter:
        filtered = [r for r in filtered if ticker_filter in r["ticker_symbol"]]
    if status_filter == "Pendientes":
        filtered = [r for r in filtered if not r["is_evaluated"]]
    elif status_filter == "Evaluadas":
        filtered = [r for r in filtered if r["is_evaluated"]]
    if direction_filter != "Todas":
        filtered = [r for r in filtered if r["predicted_direction"] == direction_filter]

    if filtered:
        display = []
        for r in filtered[:150]:
            display.append({
                "Ticker": r["ticker_symbol"],
                "Canal": r["channel_name"],
                "Recomendación": r["recommendation"] or r["prediction_type"],
                "Dirección": r["predicted_direction"],
                "Largo Plazo": "✅" if r["is_long_term"] else ("❌" if r["is_long_term"] is False else "—"),
                "P. Entrada": r["entry_price"],
                "P. Objetivo": r["target_price"],
                "Stop Loss": r["stop_loss"],
                "P. al Anuncio": r["price_at_prediction"],
                "Horizonte (d)": r["timeframe_days"],
                "Confianza": f"{r['confidence_score']:.0%}" if r["confidence_score"] else "—",
                "Fecha": r["prediction_date"],
                "Razonamiento": (r["analyst_reasoning"] or "")[:80],
                "Evaluada": "✅" if r["is_evaluated"] else "⏳",
                "Correcta": ("✅" if r["gt_is_accurate"] else "❌") if r["gt_is_accurate"] is not None else "—",
                "Cambio Real %": r["gt_actual_change_pct"],
            })
        st.dataframe(pd.DataFrame(display), use_container_width=True)
    else:
        st.info("No hay predicciones que coincidan con el filtro.")

except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)
