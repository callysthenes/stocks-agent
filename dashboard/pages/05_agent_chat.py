"""
Dashboard Page 5 — Agent Chat: interactive chat with the AI agent via REST API.
"""
import os
import sys

import httpx
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

API_BASE = "http://api:8090"
AGENT_ENDPOINT = f"{API_BASE}/api/v1/agent/query"
TIMEOUT = 120  # seconds — agent can take a while

st.set_page_config(page_title="Agent Chat — StocksAgent", page_icon="💬", layout="wide")

from dashboard.auth import require_login  # noqa: E402
require_login()

st.title("💬 Chat con el Agente")
st.caption("Consulta directamente al sistema multi-agente de análisis financiero")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example queries
with st.expander("💡 Ejemplos de consultas"):
    examples = [
        "¿Qué dicen los canales sobre Apple (AAPL)?",
        "Analiza Santander (SAN.MC) técnica y fundamentalmente",
        "¿Cuál es la precisión de las predicciones de los canales?",
        "¿Qué acciones del sector tecnológico se han mencionado recientemente?",
        "Compara Tesla vs Nvidia según los vídeos de los últimos meses",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

st.divider()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tickers"):
            st.caption(f"Tickers analizados: {', '.join(msg['tickers'])}")

# Chat input
user_input = st.chat_input("Escribe tu consulta sobre acciones...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analizando... (puede tardar 20-60 segundos según la complejidad)"):
            try:
                resp = httpx.post(
                    AGENT_ENDPOINT,
                    json={"query": user_input, "send_telegram": False},
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()

                response = data.get("final_report") or "No se pudo generar una respuesta."
                tickers = data.get("ticker_symbols", [])

                st.markdown(response)
                if tickers:
                    st.divider()
                    st.caption(f"Tickers analizados: {', '.join(tickers)}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "tickers": tickers,
                })

            except httpx.TimeoutException:
                error_msg = "Timeout: el agente tardó demasiado. Intenta una consulta más sencilla."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except httpx.HTTPStatusError as e:
                error_msg = f"Error del API ({e.response.status_code}): {e.response.text}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"Error inesperado: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear button
if st.session_state.messages:
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

st.divider()
st.caption("⚠️ *Este análisis es únicamente informativo y no constituye asesoramiento financiero.*")
