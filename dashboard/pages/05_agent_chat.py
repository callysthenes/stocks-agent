"""
Dashboard Page 5 — Agent Chat: interactive chat with the AI agent.
"""
import asyncio
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Agent Chat — StocksAgent", page_icon="💬", layout="wide")
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

st.divider()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Escribe tu consulta sobre acciones...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Analizando... (puede tardar 20-60 segundos según la complejidad)"):
            try:
                from src.agents import run_agent

                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        run_agent(query=user_input, send_telegram=False)
                    )
                finally:
                    loop.close()

                response = result.get("final_report") or "No se pudo generar una respuesta."
                tickers = result.get("ticker_symbols", [])

                st.markdown(response)

                if tickers:
                    st.divider()
                    st.caption(f"Tickers analizados: {', '.join(tickers)}")

                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"Error en el agente: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear button
if st.session_state.messages:
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

st.divider()
st.caption("⚠️ *Este análisis es únicamente informativo y no constituye asesoramiento financiero.*")
