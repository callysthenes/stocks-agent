"""
Dashboard Page 1 — Overview: channels, recent videos, pipeline status.
"""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Overview — StocksAgent", page_icon="📊", layout="wide")
st.title("📊 Overview")

try:
    from src.storage.mariadb_client import get_sync_db
    from src.models import Video, Channel, ProcessingStatus, TickerMention
    from sqlalchemy import func, select, desc
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    with get_sync_db() as db:
        # Channel stats
        channels = db.scalars(select(Channel).where(Channel.is_active == True)).all()  # noqa: E712

        # Recent videos
        recent_videos = db.scalars(
            select(Video).order_by(desc(Video.created_at)).limit(20)
        ).all()

        # Status breakdown
        status_counts = db.execute(
            select(Video.processing_status, func.count(Video.id).label("count"))
            .group_by(Video.processing_status)
        ).all()

        # Top tickers
        top_tickers = db.execute(
            select(TickerMention.ticker_symbol, func.count(TickerMention.id).label("count"))
            .group_by(TickerMention.ticker_symbol)
            .order_by(desc("count"))
            .limit(15)
        ).all()

    # ── Channels ──────────────────────────────────────────────────────────────
    st.subheader("📺 Canales Monitorizados")
    if channels:
        ch_data = [
            {"Canal": c.name, "URL": c.url, "Activo": "✅" if c.is_active else "❌",
             "Último Check": c.last_checked_at or "Nunca"}
            for c in channels
        ]
        st.dataframe(pd.DataFrame(ch_data), use_container_width=True)
    else:
        st.info("No hay canales configurados. Añade canales via API: POST /api/v1/channels")

    st.divider()

    # ── Pipeline Status ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pipeline Status")
        if status_counts:
            df_status = pd.DataFrame(
                [(s.processing_status, s.count) for s in status_counts],
                columns=["Estado", "Vídeos"]
            )
            color_map = {
                "completed": "#2ecc71",
                "failed": "#e74c3c",
                "pending": "#95a5a6",
                "transcribing": "#3498db",
                "embedding": "#9b59b6",
                "fetching": "#f39c12",
            }
            fig = px.pie(
                df_status, names="Estado", values="Vídeos",
                color="Estado", color_discrete_map=color_map,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Tickers Mencionados")
        if top_tickers:
            df_tickers = pd.DataFrame(
                [(t.ticker_symbol, t.count) for t in top_tickers],
                columns=["Ticker", "Menciones"]
            )
            fig2 = px.bar(
                df_tickers, x="Ticker", y="Menciones",
                color="Menciones", color_continuous_scale="viridis",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Recent Videos ─────────────────────────────────────────────────────────
    st.subheader("🎬 Vídeos Recientes")
    if recent_videos:
        status_emoji = {
            "completed": "✅", "failed": "❌", "pending": "⏳",
            "transcribing": "🎙️", "embedding": "🔢", "fetching": "📥",
        }
        video_data = []
        for v in recent_videos:
            st.markdown(
                f"{status_emoji.get(str(v.processing_status), '?')} "
                f"**[{v.title[:60]}]({v.youtube_url})** — "
                f"{v.channel.name if v.channel else 'Unknown'} | "
                f"{v.published_at or 'N/A'}"
            )
    else:
        st.info("No hay vídeos procesados aún.")

except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.exception(e)
