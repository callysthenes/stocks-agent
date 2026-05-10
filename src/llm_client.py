"""
Rate-limit-aware LLM client with DeepSeek V3 primary + Z.AI GLM-4 fallback.

Strategy
--------
- Track calls-per-minute for each provider in Redis (60-second sliding window).
- On a 429 / RateLimitError from DeepSeek, saturate its counter and immediately
  re-try the request via Z.AI.
- If *both* providers are exhausted, wait 30 s with jitter then retry DeepSeek.
- Conservative RPM caps (well below documented limits) avoid surprises.
"""
from __future__ import annotations

import random
import time
from typing import Any

import redis as redis_lib
from loguru import logger
from openai import OpenAI, RateLimitError

from src.config import settings

# ── Rate-limit caps (requests per minute per provider) ────────────────────────
_RATE_LIMITS: dict[str, int] = {
    "deepseek": 55,    # DeepSeek free/standard: ~60 RPM; keep headroom
    "zhipuai": 100,    # Z.AI GLM-4-flash: ~120 RPM
}

# ── Retry backoff when both providers are exhausted ───────────────────────────
_EXHAUSTED_WAIT_BASE = 30   # seconds
_EXHAUSTED_WAIT_JITTER = 10  # seconds


class _LLMClient:
    """Singleton lazy-initialised LLM client.  Use the module-level ``llm`` instance."""

    def __init__(self) -> None:
        self._deepseek: OpenAI | None = None
        self._zhipuai: OpenAI | None = None
        self._redis: redis_lib.Redis | None = None  # type: ignore[type-arg]

    # ── Lazy-init properties ──────────────────────────────────────────────────

    @property
    def deepseek(self) -> OpenAI:
        if self._deepseek is None:
            self._deepseek = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=120,
            )
        return self._deepseek

    @property
    def zhipuai(self) -> OpenAI:
        if self._zhipuai is None:
            self._zhipuai = OpenAI(
                api_key=settings.zhipuai_api_key,
                base_url=settings.zhipuai_base_url,
                timeout=120,
            )
        return self._zhipuai

    @property
    def redis(self) -> redis_lib.Redis:  # type: ignore[type-arg]
        if self._redis is None:
            self._redis = redis_lib.Redis.from_url(
                settings.redis_url, decode_responses=True, socket_connect_timeout=3
            )
        return self._redis

    # ── Rate-limit helpers ────────────────────────────────────────────────────

    def _is_saturated(self, provider: str) -> bool:
        try:
            count = int(self.redis.get(f"llm:rate:{provider}") or 0)
            return count >= _RATE_LIMITS[provider]
        except Exception:
            return False  # Redis down → optimistically proceed

    def _record(self, provider: str) -> None:
        try:
            pipe = self.redis.pipeline()
            pipe.incr(f"llm:rate:{provider}")
            pipe.expire(f"llm:rate:{provider}", 60)
            pipe.execute()
        except Exception:
            pass

    def _saturate(self, provider: str, ttl: int = 60) -> None:
        """Pin counter at limit so future calls skip this provider."""
        try:
            self.redis.set(
                f"llm:rate:{provider}",
                _RATE_LIMITS[provider] * 10,
                ex=ttl,
            )
        except Exception:
            pass

    # ── Core chat method ─────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat completion.  Returns the content string.

        Routing priority:
          1. DeepSeek (if not saturated)
          2. Z.AI GLM (if configured and not saturated)
          3. Wait + retry DeepSeek
        """
        # ── Try DeepSeek ──────────────────────────────────────────────────────
        if not self._is_saturated("deepseek"):
            try:
                resp = self.deepseek.chat.completions.create(
                    model=model or settings.deepseek_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self._record("deepseek")
                content = resp.choices[0].message.content or ""
                logger.debug(f"[LLM] deepseek → {len(content)} chars")
                return content
            except RateLimitError as exc:
                logger.warning(f"[LLM] DeepSeek rate-limited: {exc}; switching to Z.AI")
                self._saturate("deepseek", ttl=60)
            except Exception as exc:
                logger.error(f"[LLM] DeepSeek error: {exc}; trying Z.AI")

        # ── Try Z.AI GLM ─────────────────────────────────────────────────────
        if settings.zhipuai_api_key and not self._is_saturated("zhipuai"):
            try:
                resp = self.zhipuai.chat.completions.create(
                    model=settings.zhipuai_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self._record("zhipuai")
                content = resp.choices[0].message.content or ""
                logger.info(f"[LLM] z.ai/{settings.zhipuai_model} → {len(content)} chars")
                return content
            except RateLimitError as exc:
                logger.warning(f"[LLM] Z.AI rate-limited: {exc}")
                self._saturate("zhipuai", ttl=60)
            except Exception as exc:
                logger.error(f"[LLM] Z.AI error: {exc}")

        # ── Both exhausted — wait, then force DeepSeek ────────────────────────
        wait = _EXHAUSTED_WAIT_BASE + random.uniform(0, _EXHAUSTED_WAIT_JITTER)
        logger.warning(f"[LLM] All providers exhausted. Waiting {wait:.1f}s …")
        time.sleep(wait)
        resp = self.deepseek.chat.completions.create(
            model=model or settings.deepseek_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._record("deepseek")
        return resp.choices[0].message.content or ""


# ── Module-level singleton ─────────────────────────────────────────────────────
llm = _LLMClient()
