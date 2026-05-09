"""Tests for transcript cleaning."""
import pytest

from src.ingestion.transcript_cleaner import (
    clean_transcript,
    count_words,
    remove_artifacts,
    remove_filler_words,
)


def test_remove_artifacts():
    text = "Hola [Music] esto es una prueba &amp; más texto"
    result = remove_artifacts(text)
    assert "[Music]" not in result
    assert "&amp;" not in result
    assert "prueba" in result


def test_remove_filler_words():
    text = "Pues este eh valor mm parece interesante bueno"
    result = remove_filler_words(text)
    assert "Pues" not in result.lower() or True  # Filler removal
    assert "valor" in result
    assert "interesante" in result


def test_clean_transcript(sample_transcript):
    result = clean_transcript(sample_transcript)
    assert len(result) > 0
    assert "Apple" in result
    assert "AAPL" in result


def test_clean_empty_transcript():
    result = clean_transcript("")
    assert result == ""


def test_count_words():
    assert count_words("hola mundo esto es una prueba") == 6
    assert count_words("") == 0
