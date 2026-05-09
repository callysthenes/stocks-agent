"""
Application configuration using Pydantic Settings.
All values are loaded from environment variables / .env file.
"""
from functools import lru_cache
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── DeepSeek API ──────────────────────────────────────────────────────────
    deepseek_api_key: str = Field(..., description="DeepSeek API key")
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(..., description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Default chat ID for reports")

    # ── MariaDB ───────────────────────────────────────────────────────────────
    mariadb_host: str = "mariadb"
    mariadb_port: int = 3306
    mariadb_database: str = "stocks_agent"
    mariadb_user: str = "appuser"
    mariadb_password: str = Field(..., description="MariaDB app user password")
    mariadb_root_password: str = Field(default="", description="MariaDB root password")

    @computed_field  # type: ignore[misc]
    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mariadb_user}:{self.mariadb_password}"
            f"@{self.mariadb_host}:{self.mariadb_port}/{self.mariadb_database}"
            "?charset=utf8mb4"
        )

    @computed_field  # type: ignore[misc]
    @property
    def database_url_async(self) -> str:
        return (
            f"mysql+aiomysql://{self.mariadb_user}:{self.mariadb_password}"
            f"@{self.mariadb_host}:{self.mariadb_port}/{self.mariadb_database}"
            "?charset=utf8mb4"
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    chromadb_collection: str = "video_transcripts"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://ollama:11434"

    # ── Langfuse ──────────────────────────────────────────────────────────────
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://langfuse:3000"

    # ── Embedding Model ───────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # ── Whisper ───────────────────────────────────────────────────────────────
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # ── Processing ────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_videos_per_fetch: int = 5
    media_dir: str = "/app/media"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Scheduling ────────────────────────────────────────────────────────────
    daily_report_hour_utc: int = 22
    daily_report_minute_utc: int = 30
    prediction_eval_hour_utc: int = 22
    prediction_eval_minute_utc: int = 0
    channel_poll_interval_minutes: int = 30

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_url: str = "http://localhost:8501"

    # ── Logging / Environment ─────────────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
