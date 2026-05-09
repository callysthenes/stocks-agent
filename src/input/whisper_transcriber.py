"""
Whisper transcriber using faster-whisper for local GPU inference.
Model is loaded once per worker process and cached as a module-level singleton.
"""
import os
from typing import TYPE_CHECKING

from loguru import logger

from src.config import settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

_whisper_model: "WhisperModel | None" = None


def get_whisper_model() -> "WhisperModel":
    """Load and cache the Whisper model (singleton per process)."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run this in the worker container (Dockerfile.worker)."
            ) from e

        logger.info(
            f"Loading Whisper model '{settings.whisper_model}' on "
            f"{settings.whisper_device} ({settings.whisper_compute_type})..."
        )
        _whisper_model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root="/app/.cache/whisper",
        )
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def transcribe_audio(audio_path: str, language: str = "es") -> str:
    """
    Transcribe an audio file using Whisper large-v3.

    Args:
        audio_path: Path to the mp3/wav/m4a audio file
        language: Language code (default "es" for Spanish)

    Returns:
        Transcribed text as a single string.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        RuntimeError: If transcription fails.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model()

    logger.info(f"Transcribing {audio_path} (language={language})")
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
        )
        logger.debug(
            f"Whisper detected language: {info.language} "
            f"(probability={info.language_probability:.2f})"
        )
        transcript = " ".join(segment.text.strip() for segment in segments)
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed for {audio_path}: {e}") from e
    finally:
        # Clean up the audio file regardless of success/failure
        try:
            os.remove(audio_path)
            logger.debug(f"Cleaned up audio file: {audio_path}")
        except OSError:
            pass

    return transcript.strip()


def unload_model() -> None:
    """Explicitly unload the Whisper model to free GPU memory."""
    global _whisper_model
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        logger.info("Whisper model unloaded")
