"""
Dashboard authentication — login / register backed by MariaDB.

Uses PBKDF2-HMAC-SHA256 for password hashing (stdlib only, no extra deps).
The `dashboard_users` table is created on first call if it does not exist.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from base64 import b64decode, b64encode
from datetime import datetime
from typing import TypedDict

import streamlit as st
from sqlalchemy import text


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _db():
    from src.storage.mariadb_client import get_sync_db
    return get_sync_db()


def _ensure_table() -> None:
    """Create dashboard_users table if it does not exist (idempotent)."""
    with _db() as db:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS dashboard_users (
                id            VARCHAR(36)  NOT NULL PRIMARY KEY,
                username      VARCHAR(64)  NOT NULL UNIQUE,
                email         VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
                is_admin      BOOLEAN      NOT NULL DEFAULT FALSE,
                created_at    DATETIME     NOT NULL,
                updated_at    DATETIME     NOT NULL
            ) ENGINE=InnoDB
              DEFAULT CHARSET=utf8mb4
              COLLATE=utf8mb4_unicode_ci
        """))
        db.commit()


# ─── Password hashing (PBKDF2-HMAC-SHA256, 600 k iterations) ─────────────────

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(32)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return b64encode(salt + dk).decode("ascii")


def verify_password(password: str, stored: str) -> bool:
    try:
        raw = b64decode(stored.encode("ascii"))
        salt, stored_dk = raw[:32], raw[32:]
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
        return hmac.compare_digest(dk, stored_dk)
    except Exception:
        return False


# ─── User DB operations ───────────────────────────────────────────────────────

class UserRow(TypedDict):
    id: str
    username: str
    email: str
    password_hash: str
    is_active: bool
    is_admin: bool


def get_user(username: str) -> UserRow | None:
    with _db() as db:
        row = db.execute(
            text(
                "SELECT id, username, email, password_hash, is_active, is_admin "
                "FROM dashboard_users WHERE username = :u"
            ),
            {"u": username},
        ).fetchone()
    return dict(row._mapping) if row else None


def username_exists(username: str) -> bool:
    with _db() as db:
        n = db.execute(
            text("SELECT COUNT(*) FROM dashboard_users WHERE username = :u"),
            {"u": username},
        ).scalar()
    return (n or 0) > 0


def email_exists(email: str) -> bool:
    with _db() as db:
        n = db.execute(
            text("SELECT COUNT(*) FROM dashboard_users WHERE email = :e"),
            {"e": email},
        ).scalar()
    return (n or 0) > 0


def count_users() -> int:
    with _db() as db:
        n = db.execute(text("SELECT COUNT(*) FROM dashboard_users")).scalar()
    return n or 0


def create_user(
    username: str,
    email: str,
    password: str,
    is_admin: bool = False,
) -> tuple[bool, str]:
    """
    Create a new user. Returns (success, error_message).
    First user is automatically promoted to admin.
    """
    if len(username) < 3:
        return False, "El nombre de usuario debe tener al menos 3 caracteres."
    if len(password) < 8:
        return False, "La contraseña debe tener al menos 8 caracteres."
    if "@" not in email:
        return False, "Email no válido."
    if username_exists(username):
        return False, "Ese nombre de usuario ya está en uso."
    if email_exists(email):
        return False, "Ese email ya está registrado."

    first_user = count_users() == 0
    now = datetime.utcnow()
    try:
        with _db() as db:
            db.execute(
                text("""
                    INSERT INTO dashboard_users
                        (id, username, email, password_hash, is_active, is_admin, created_at, updated_at)
                    VALUES
                        (:id, :username, :email, :pw_hash, TRUE, :is_admin, :now, :now)
                """),
                {
                    "id": str(uuid.uuid4()),
                    "username": username,
                    "email": email,
                    "pw_hash": hash_password(password),
                    "is_admin": first_user or is_admin,
                    "now": now,
                },
            )
            db.commit()
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ─── Session helpers ──────────────────────────────────────────────────────────

def _set_session(user: UserRow) -> None:
    st.session_state["authenticated"] = True
    st.session_state["username"] = user["username"]
    st.session_state["is_admin"] = user["is_admin"]


def logout() -> None:
    for key in ("authenticated", "username", "is_admin"):
        st.session_state.pop(key, None)
    st.rerun()


def is_logged_in() -> bool:
    return bool(st.session_state.get("authenticated"))


def current_user() -> str:
    return st.session_state.get("username", "")


# ─── Auth gate ────────────────────────────────────────────────────────────────

def require_login() -> None:
    """
    Call at the top of every page.
    Shows login / register UI and calls st.stop() if the user is not logged in.
    """
    _ensure_table()
    if is_logged_in():
        _render_sidebar_user()
        return

    _render_auth_page()
    st.stop()


# ─── UI helpers ───────────────────────────────────────────────────────────────

def _render_sidebar_user() -> None:
    with st.sidebar:
        st.divider()
        st.caption(f"Conectado como **{current_user()}**")
        if st.button("Cerrar sesión", use_container_width=True):
            logout()


def _render_auth_page() -> None:
    # Centre the form with columns
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.title("📈 StocksAgent")
        st.caption("Análisis de acciones con IA — acceso restringido")
        st.divider()

        tab_login, tab_register = st.tabs(["Iniciar sesión", "Registrarse"])

        with tab_login:
            _login_form()

        with tab_register:
            _register_form()


def _login_form() -> None:
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Completa todos los campos.")
            return
        user = get_user(username)
        if user is None or not verify_password(password, user["password_hash"]):
            st.error("Usuario o contraseña incorrectos.")
            return
        if not user["is_active"]:
            st.error("Cuenta desactivada. Contacta con el administrador.")
            return
        _set_session(user)
        st.rerun()


def _register_form() -> None:
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Usuario (mín. 3 caracteres)")
        email = st.text_input("Email")
        password = st.text_input("Contraseña (mín. 8 caracteres)", type="password")
        password2 = st.text_input("Repite la contraseña", type="password")
        submitted = st.form_submit_button("Crear cuenta", use_container_width=True)

    if submitted:
        if not all([username, email, password, password2]):
            st.error("Completa todos los campos.")
            return
        if password != password2:
            st.error("Las contraseñas no coinciden.")
            return
        ok, err = create_user(username, email, password)
        if not ok:
            st.error(err)
            return
        st.success("¡Cuenta creada! Ya puedes iniciar sesión.")
