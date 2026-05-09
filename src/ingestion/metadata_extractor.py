"""
Metadata extractor — uses DeepSeek V3 (JSON mode) to extract:
  - Ticker mentions with sentiment from transcript text
  - Explicit buy/sell/hold predictions
"""
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

# ── LLM client ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        max_tokens=4096,
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Eres un experto analista financiero especializado en los mercados bursátiles español, europeo y norteamericano.
Tu tarea es analizar transcripciones de vídeos de YouTube sobre inversión en bolsa y extraer información estructurada.
Responde SIEMPRE en formato JSON válido y NADA MÁS. No incluyas explicaciones ni texto adicional fuera del JSON."""

_TICKER_EXTRACTION_PROMPT = """Analiza la siguiente transcripción de un vídeo de inversión en bolsa en español y extrae TODOS los activos financieros mencionados (acciones, ETFs, índices, criptomonedas).

Para cada uno devuelve:
- ticker_symbol: símbolo bursátil (ej: AAPL, SAN.MC, MSFT, BTC-USD). Si no sabes el ticker exacto, inferlo del nombre de la empresa.
- company_name: nombre completo de la empresa o activo
- exchange: mercado (NYSE, NASDAQ, BME, LSE, EURONEXT, CRYPTO, etc.)
- sentiment: sentimiento del presentador → "bullish", "bearish" o "neutral"
- context_snippet: fragmento de texto donde se menciona (máx 200 caracteres)
- confidence: confianza 0.0-1.0 en la extracción

Transcripción:
{transcript}

Responde SOLO con un JSON array. Si no hay tickers, devuelve [].
Ejemplo:
[{{"ticker_symbol": "SAN.MC", "company_name": "Banco Santander", "exchange": "BME", "sentiment": "bullish", "context_snippet": "Santander me parece una empresa muy sólida...", "confidence": 0.9}}]"""

_PREDICTION_EXTRACTION_PROMPT = """De la siguiente transcripción, extrae ÚNICAMENTE las predicciones o recomendaciones EXPLÍCITAS que hace el presentador sobre los activos: {tickers}.

Una predicción explícita es cuando el presentador dice claramente que compraría, vendería, o que espera que el precio suba/baje.

Para cada predicción devuelve:
- ticker_symbol: símbolo del activo
- prediction_type: "buy", "sell", "hold" o "watch"
- predicted_direction: "up", "down" o "neutral"
- target_price: precio objetivo numérico o null
- timeframe_days: horizonte temporal en días o null (ej: "en 6 meses" → 180)
- context_snippet: cita exacta del presentador (máx 300 caracteres)

Transcripción:
{transcript}

Responde SOLO con un JSON array. Si no hay predicciones explícitas, devuelve [].
Ejemplo:
[{{"ticker_symbol": "AAPL", "prediction_type": "buy", "predicted_direction": "up", "target_price": 200.0, "timeframe_days": 90, "context_snippet": "Yo compraría Apple por debajo de 180..."}}]"""


# ── Extractor class ───────────────────────────────────────────────────────────

class MetadataExtractor:
    """Extracts structured financial metadata from Spanish transcripts."""

    def __init__(self) -> None:
        self._llm = _get_llm()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        reraise=True,
    )
    def extract_tickers(self, transcript: str) -> list[dict[str, Any]]:
        """
        Extract ticker mentions with sentiment from a transcript.
        Splits long transcripts into segments to stay within token limits.
        """
        # Use first 6000 words to stay within context window
        truncated = " ".join(transcript.split()[:6000])

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_TICKER_EXTRACTION_PROMPT.format(transcript=truncated)),
        ]
        try:
            response = self._llm.invoke(messages)
            return self._parse_json_list(response.content)
        except Exception as e:
            logger.error(f"Ticker extraction failed: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        reraise=True,
    )
    def extract_predictions(
        self, transcript: str, tickers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Extract explicit predictions for the given tickers.
        """
        if not tickers:
            return []

        ticker_list = ", ".join(
            t["ticker_symbol"] for t in tickers if t.get("ticker_symbol")
        )
        truncated = " ".join(transcript.split()[:6000])

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=_PREDICTION_EXTRACTION_PROMPT.format(
                    tickers=ticker_list, transcript=truncated
                )
            ),
        ]
        try:
            response = self._llm.invoke(messages)
            return self._parse_json_list(response.content)
        except Exception as e:
            logger.error(f"Prediction extraction failed: {e}")
            return []

    @staticmethod
    def _parse_json_list(content: str) -> list[dict[str, Any]]:
        """Parse JSON array from LLM response, handling markdown code blocks."""
        content = content.strip()

        # Strip markdown code block if present
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        # Find the first JSON array in the response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            content = match.group()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}\nContent: {content[:200]}")
            return []
