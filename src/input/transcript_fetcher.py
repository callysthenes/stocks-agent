"""
YouTube transcript fetcher using youtube-transcript-api.
Tries Spanish captions first, falls back to auto-generated, then translated.
"""
from loguru import logger
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)
from youtube_transcript_api._errors import VideoUnavailable


class TranscriptFetcher:
    """Fetches existing captions from YouTube."""

    # Priority order for Spanish variants
    SPANISH_LANGS = ["es", "es-ES", "es-419", "es-MX", "es-AR"]
    FALLBACK_LANGS = ["en", "en-US", "en-GB"]

    def __init__(self) -> None:
        self.api = YouTubeTranscriptApi()

    def fetch_transcript(self, video_id: str) -> tuple[str, str] | tuple[None, None]:
        """
        Attempt to fetch a Spanish transcript from YouTube.

        Returns:
            (transcript_text, language_code) on success
            (None, None) if no transcript is available
        """
        try:
            transcript_list = self.api.list(video_id)
        except (TranscriptsDisabled, VideoUnavailable) as e:
            logger.debug(f"Transcripts disabled or video unavailable for {video_id}: {e}")
            return None, None
        except Exception as e:
            logger.warning(f"Could not list transcripts for {video_id}: {e}")
            return None, None

        # 1. Try manually-created Spanish
        for lang in self.SPANISH_LANGS:
            try:
                t = transcript_list.find_manually_created_transcript([lang])
                entries = t.fetch()
                return self._join_entries(entries), lang
            except NoTranscriptFound:
                continue
            except Exception:
                continue

        # 2. Try auto-generated Spanish
        for lang in self.SPANISH_LANGS:
            try:
                t = transcript_list.find_generated_transcript([lang])
                entries = t.fetch()
                logger.debug(f"Using auto-generated {lang} captions for {video_id}")
                return self._join_entries(entries), lang
            except NoTranscriptFound:
                continue
            except Exception:
                continue

        # 3. Try translating from English
        for lang in self.FALLBACK_LANGS:
            try:
                t = transcript_list.find_transcript([lang]).translate("es")
                entries = t.fetch()
                logger.debug(f"Using translated (en→es) captions for {video_id}")
                return self._join_entries(entries), "es-translated"
            except (NoTranscriptFound, Exception):
                continue

        logger.debug(f"No usable YouTube captions found for {video_id}")
        return None, None

    @staticmethod
    def _join_entries(entries) -> str:
        """Concatenate transcript entries into a single string."""
        return " ".join(
            str(e.get("text", "") if isinstance(e, dict) else getattr(e, "text", "")).strip()
            for e in entries
            if e
        ).strip()
