"""
Metadata extractor — uses DeepSeek V3 (JSON mode) to extract:
  - Ticker mentions with sentiment from transcript text
  - Explicit buy/sell/hold predictions with price levels and reasoning
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

Para cada uno devuelve un objeto con EXACTAMENTE estos campos:
- ticker_symbol: símbolo bursátil (ej: AAPL, SAN.MC, MSFT, BTC-USD, ^IBEX). Si no conoces el ticker exacto, infierelo del nombre.
- company_name: nombre completo de la empresa o activo
- exchange: mercado (NYSE, NASDAQ, BME, LSE, EURONEXT, XETRA, CRYPTO, INDEX, etc.)
- sector: sector empresarial (Technology, Financials, Energy, Healthcare, etc.) o null si no aplica
- currency: divisa principal (EUR, USD, GBP, etc.) o null si no sabes
- country: país de origen (Spain, USA, Germany, etc.) o null
- is_index: true si es un índice (IBEX35, S&P500, etc.), false en caso contrario
- is_etf: true si es un ETF, false en caso contrario
- is_crypto: true si es criptomoneda, false en caso contrario
- sentiment: sentimiento del presentador → "bullish", "bearish" o "neutral"
- mention_count: número aproximado de veces que se menciona en el fragmento (entero ≥ 1)
- context_snippet: fragmento donde se menciona con mayor detalle (máx 200 caracteres)
- confidence: confianza 0.0-1.0 en la extracción del ticker

Transcripción:
{transcript}

Responde SOLO con un JSON array. Si no hay tickers, devuelve [].
Ejemplo:
[{{"ticker_symbol": "SAN.MC", "company_name": "Banco Santander", "exchange": "BME", "sector": "Financials", "currency": "EUR", "country": "Spain", "is_index": false, "is_etf": false, "is_crypto": false, "sentiment": "bullish", "mention_count": 3, "context_snippet": "Santander me parece una empresa muy sólida...", "confidence": 0.9}}]"""

_PREDICTION_EXTRACTION_PROMPT = """De la siguiente transcripción, extrae ÚNICAMENTE las predicciones o recomendaciones EXPLÍCITAS que hace el presentador sobre los activos: {tickers}.

Una predicción explícita es cuando el presentador dice claramente que compraría, vendería, acumularía, o que espera que el precio suba/baje/se mantenga.

Para cada predicción devuelve un objeto con EXACTAMENTE estos campos:
- ticker_symbol: símbolo del activo (de la lista proporcionada)
- prediction_type: "buy", "sell", "hold" o "watch"
- recommendation: recomendación granular → "buy", "accumulate", "hold", "reduce", "sell" o "avoid"
- predicted_direction: "up", "down" o "neutral"
- is_long_term: true si el horizonte es > 6 meses o el presentador habla de largo plazo, false en caso contrario
- price_at_prediction: precio actual del activo mencionado en el vídeo o null
- entry_price: precio de entrada recomendado por el presentador o null
- target_price: precio objetivo mencionado o null
- stop_loss: nivel de stop loss mencionado o null
- timeframe_days: horizonte temporal en días (ej: "en 6 meses" → 180, "a largo plazo" → 365) o null
- confidence_score: confianza 0.0-1.0 del presentador en su predicción según el contexto
- analyst_reasoning: razón principal de la recomendación en ≤ 200 caracteres
- context_snippet: cita literal del presentador (máx 300 caracteres)

Transcripción:
{transcript}

Responde SOLO con un JSON array. Si no hay predicciones explícitas, devuelve [].
Ejemplo:
[{{"ticker_symbol": "AAPL", "prediction_type": "buy", "recommendation": "accumulate", "predicted_direction": "up", "is_long_term": false, "price_at_prediction": 175.0, "entry_price": 170.0, "target_price": 200.0, "stop_loss": 160.0, "timeframe_days": 90, "confidence_score": 0.8, "analyst_reasoning": "Corrección técnica a soporte clave con catalizadores macro favorables", "context_snippet": "Yo compraría Apple por debajo de 170 con objetivo en 200..."}}]"""


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
        """Extract ticker mentions with sentiment and metadata from a transcript."""
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
        """Extract explicit predictions with price levels and reasoning."""
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
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
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
