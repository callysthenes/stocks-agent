"""Tests for transcript fetcher."""
import pytest
from unittest.mock import MagicMock, patch

from src.input.transcript_fetcher import TranscriptFetcher


def test_fetch_transcript_success():
    """Test successful Spanish caption fetch."""
    fetcher = TranscriptFetcher()

    mock_entries = [{"text": "Hola mundo"}, {"text": "esto es una prueba"}]
    mock_transcript = MagicMock()
    mock_transcript.fetch.return_value = mock_entries

    mock_list = MagicMock()
    mock_list.find_manually_created_transcript.return_value = mock_transcript

    with patch.object(fetcher.api, "list", return_value=mock_list):
        text, lang = fetcher.fetch_transcript("test_video_id")

    assert text == "Hola mundo esto es una prueba"


def test_fetch_transcript_none_when_disabled():
    """Test returns None when transcripts are disabled."""
    from youtube_transcript_api import TranscriptsDisabled
    fetcher = TranscriptFetcher()

    with patch.object(fetcher.api, "list", side_effect=TranscriptsDisabled("test_id")):
        text, lang = fetcher.fetch_transcript("test_video_id")

    assert text is None
    assert lang is None
