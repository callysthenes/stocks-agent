"""
Dashboard Page 6 — DB Explorer: browse raw MariaDB tables and ChromaDB stats.
"""
import os
import sys

import httpx
import pandas as pd
import streamlit as st
from sqlalchemy import inspect, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CHROMA_BASE = "http://chromadb:8000"
CHROMA_TENANT = "default_tenant"
CHROMA_DB = "default_database"
CHROMA_API = f"{CHROMA_BASE}/api/v2/tenants/{CHROMA_TENANT}/databases/{CHROMA_DB}"

st.set_page_config(page_title="DB Explorer — StocksAgent", page_icon="🗄️", layout="wide")

from dashboard.auth import require_login  # noqa: E402
require_login()

st.title("🗄️ DB Explorer")
st.caption("Navega por las tablas de MariaDB y las colecciones de ChromaDB")

tab_mariadb, tab_chroma = st.tabs(["🐬 MariaDB", "🔮 ChromaDB"])

# ── MariaDB ───────────────────────────────────────────────────────────────────
with tab_mariadb:
    try:
        from src.storage.mariadb_client import sync_engine

        inspector = inspect(sync_engine)
        table_names = inspector.get_table_names()

        if not table_names:
            st.info("No se encontraron tablas en la base de datos.")
        else:
            col_sel, col_limit = st.columns([3, 1])
            with col_sel:
                selected_table = st.selectbox("Tabla", sorted(table_names))
            with col_limit:
                row_limit = st.number_input("Filas máx.", min_value=10, max_value=5000, value=200, step=50)

            if selected_table:
                # Column info
                with st.expander("Columnas", expanded=False):
                    cols = inspector.get_columns(selected_table)
                    col_df = pd.DataFrame([
                        {"Columna": c["name"], "Tipo": str(c["type"]), "Nullable": c.get("nullable", True)}
                        for c in cols
                    ])
                    st.dataframe(col_df, use_container_width=True)

                # Row count
                with sync_engine.connect() as conn:
                    total_rows = conn.execute(
                        text(f"SELECT COUNT(*) FROM `{selected_table}`")  # noqa: S608
                    ).scalar()
                st.caption(f"Total filas: **{total_rows:,}** — mostrando primeras {row_limit}")

                # Data
                with sync_engine.connect() as conn:
                    df = pd.read_sql(
                        text(f"SELECT * FROM `{selected_table}` LIMIT :lim"),  # noqa: S608
                        conn,
                        params={"lim": row_limit},
                    )
                st.dataframe(df, use_container_width=True)

                # Raw SQL
                with st.expander("Ejecutar SQL personalizado", expanded=False):
                    st.warning("Solo lectura — instrucciones SELECT únicamente.")
                    raw_sql = st.text_area("SQL", value=f"SELECT * FROM `{selected_table}` LIMIT 50")
                    if st.button("Ejecutar"):
                        sql_lower = raw_sql.strip().lower()
                        if not sql_lower.startswith("select"):
                            st.error("Solo se permiten consultas SELECT.")
                        else:
                            try:
                                with sync_engine.connect() as conn:
                                    result_df = pd.read_sql(text(raw_sql), conn)
                                st.dataframe(result_df, use_container_width=True)
                                st.caption(f"{len(result_df)} filas devueltas")
                            except Exception as sql_err:
                                st.error(f"Error SQL: {sql_err}")

    except Exception as e:
        st.error(f"Error conectando a MariaDB: {e}")
        st.exception(e)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
with tab_chroma:
    try:
        resp = httpx.get(f"{CHROMA_BASE}/api/v2/heartbeat", timeout=5)
        resp.raise_for_status()
        st.success("ChromaDB conectado")
    except Exception as e:
        st.error(f"ChromaDB no disponible: {e}")
        st.stop()

    # List collections
    try:
        col_resp = httpx.get(f"{CHROMA_API}/collections", timeout=10)
        col_resp.raise_for_status()
        collections = col_resp.json()

        if not collections:
            st.info("No hay colecciones en ChromaDB.")
        else:
            coll_names = [c["name"] for c in collections]
            selected_coll = st.selectbox("Colección", coll_names)

            coll_obj = next((c for c in collections if c["name"] == selected_coll), None)
            if coll_obj:
                coll_id = coll_obj["id"]

                count_resp = httpx.get(f"{CHROMA_API}/collections/{coll_id}/count", timeout=10)
                count_resp.raise_for_status()
                doc_count = count_resp.json()

                col1, col2, col3 = st.columns(3)
                col1.metric("Colección", selected_coll)
                col2.metric("Documentos", f"{doc_count:,}")
                col3.metric("ID", coll_id[:8] + "…")

                with st.expander("Metadatos de la colección", expanded=False):
                    st.json(coll_obj)

                st.subheader("Muestra de documentos")
                n_sample = st.slider("Número de documentos", min_value=1, max_value=50, value=10)

                query_resp = httpx.post(
                    f"{CHROMA_API}/collections/{coll_id}/get",
                    json={"limit": n_sample, "include": ["documents", "metadatas"]},
                    timeout=15,
                )
                query_resp.raise_for_status()
                data = query_resp.json()

                ids = data.get("ids", [])
                docs = data.get("documents", [])
                metas = data.get("metadatas", [])

                if ids:
                    rows = []
                    for i, doc_id in enumerate(ids):
                        row = {"ID": doc_id}
                        if metas and i < len(metas):
                            row.update(metas[i] or {})
                        if docs and i < len(docs):
                            row["Texto (preview)"] = (docs[i] or "")[:200]
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("La colección está vacía.")

    except Exception as e:
        st.error(f"Error consultando ChromaDB: {e}")
        st.exception(e)
