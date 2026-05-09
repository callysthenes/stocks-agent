"""Tests for text chunker."""
import pytest

from src.ingestion.chunker import Chunk, chunk_text


def test_chunk_basic():
    text = " ".join(["Esta es una frase de prueba." for _ in range(100)])
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_short_text():
    text = "Texto muy corto."
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_empty_text():
    chunks = chunk_text("")
    assert chunks == []


def test_chunk_overlap():
    """Verify that consecutive chunks share some content (overlap)."""
    text = " ".join([f"Frase número {i}." for i in range(100)])
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=30)
    if len(chunks) >= 2:
        # Some words from the end of chunk 0 should appear at start of chunk 1
        words_end = set(chunks[0].text.split()[-5:])
        words_start = set(chunks[1].text.split()[:10])
        assert len(words_end & words_start) > 0, "Expected overlap between consecutive chunks"


def test_chunk_indices_sequential():
    text = " ".join([f"Frase {i}." for i in range(200)])
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
