"""
StocksAgent Dashboard — Streamlit entry point.
"""
import os
import sys

import streamlit as st

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="StocksAgent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation header
with st.sidebar:
    st.title("📈 StocksAgent")
    st.caption("AI Stock Analysis from Spanish YouTube Channels")
    st.divider()
    st.markdown("**Navegación**")

# Main landing page
st.title("📈 StocksAgent — Panel de Control")
st.caption("Análisis de acciones basado en canales de YouTube en español")

st.divider()

col1, col2, col3, col4 = st.columns(4)

try:
    from src.storage.mariadb_client import get_sync_db
    from src.models import Video, ProcessingStatus, Channel, Prediction
    from sqlalchemy import func, select

    with get_sync_db() as db:
        total_channels = db.scalar(select(func.count(Channel.id))) or 0
        total_videos = db.scalar(select(func.count(Video.id))) or 0
        completed_videos = db.scalar(
            select(func.count(Video.id)).where(Video.processing_status == ProcessingStatus.completed)
        ) or 0
        total_predictions = db.scalar(select(func.count(Prediction.id))) or 0

    col1.metric("Canales", total_channels)
    col2.metric("Vídeos Procesados", completed_videos, f"{total_videos} total")
    col3.metric("Predicciones", total_predictions)
    col4.metric("Tasa Éxito", f"{(completed_videos/total_videos*100):.0f}%" if total_videos else "N/A")

except Exception as e:
    st.error(f"Error conectando a la base de datos: {e}")
    col1.metric("Canales", "—")
    col2.metric("Vídeos", "—")
    col3.metric("Predicciones", "—")
    col4.metric("Tasa Éxito", "—")

st.divider()
st.info("Usa el menú de la izquierda para navegar entre secciones.")

st.markdown("""
### Secciones disponibles

| Página | Descripción |
|--------|-------------|
| 📊 Overview | Resumen de canales, vídeos y estado del pipeline |
| 🎯 Predictions | Seguimiento de predicciones y precisión por canal |
| 🔍 Ticker Analysis | Análisis técnico, fundamental y contexto de canales |
| 🏆 Channel Accuracy | Comparativa de precisión entre canales |
| 💬 Agent Chat | Chat interactivo con el agente de IA |

---
⚠️ *Este dashboard es únicamente informativo y no constituye asesoramiento financiero.*
""")
