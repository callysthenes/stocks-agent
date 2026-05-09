"""Test configuration and shared fixtures."""
import os

import pytest

# Set test environment before any imports
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0000000000:test")
os.environ.setdefault("MARIADB_PASSWORD", "test")
os.environ.setdefault("MARIADB_HOST", "localhost")
os.environ.setdefault("CHROMADB_HOST", "localhost")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("WHISPER_DEVICE", "cpu")


@pytest.fixture
def sample_transcript() -> str:
    return (
        "Hola a todos, bienvenidos al canal. Hoy vamos a hablar de Apple, "
        "cuyo ticker es AAPL en el Nasdaq. Creo que Apple es una empresa muy sólida "
        "y que el precio puede llegar a los 200 dólares en los próximos 6 meses. "
        "También quiero mencionar a Santander, SAN en la bolsa de Madrid, que me parece "
        "muy interesante para el largo plazo. Por otro lado, Tesla TSLA me genera dudas."
    )


@pytest.fixture
def sample_ticker_mentions() -> list[dict]:
    return [
        {
            "ticker_symbol": "AAPL",
            "company_name": "Apple Inc",
            "exchange": "NASDAQ",
            "sentiment": "bullish",
            "context_snippet": "Apple es una empresa muy sólida",
            "confidence": 0.95,
        },
        {
            "ticker_symbol": "SAN.MC",
            "company_name": "Banco Santander",
            "exchange": "BME",
            "sentiment": "bullish",
            "context_snippet": "Santander me parece muy interesante",
            "confidence": 0.85,
        },
    ]
