"""
Celery transcription tasks — run on the 'gpu' queue (GPU 0).
Handles YouTube caption fetching with Whisper fallback.
"""
from loguru import logger

from src.celery_app import celery_app
from src.models import ProcessingStatus, TranscriptSource
from src.storage import repository
from src.storage.mariadb_client import get_sync_db


@celery_app.task(
    name="src.tasks.transcription_tasks.transcribe_video",
    bind=True,
    max_retries=2,
    queue="gpu",
    time_limit=3600,  # 1 hour max (long videos)
    soft_time_limit=3300,
)
def transcribe_video(self, video_id: str) -> str:
    """
    Transcribe a single video.

    1. Try YouTube captions (fast, no GPU)
    2. Fallback: download audio → Whisper large-v3 (GPU)

    Returns:
        video_id (for use in Celery chain with ingest_video)
    """
    logger.info(f"Transcribing video {video_id}")

    with get_sync_db() as db:
        video = repository.get_video_by_id(db, video_id)
        if not video:
            logger.error(f"Video {video_id} not found")
            return video_id

        if video.processing_status == ProcessingStatus.completed:
            logger.info(f"Video {video_id} already processed, skipping")
            return video_id

        repository.update_video_status(db, video_id, ProcessingStatus.transcribing)
        youtube_video_id = video.youtube_video_id
        youtube_url = video.youtube_url

    # ── Step 1: Try YouTube captions ─────────────────────────────────────────
    try:
        from src.input.transcript_fetcher import TranscriptFetcher
        fetcher = TranscriptFetcher()
        transcript_text, lang = fetcher.fetch_transcript(youtube_video_id)

        if transcript_text and len(transcript_text.split()) > 50:
            logger.info(
                f"Video {video_id}: using YouTube captions "
                f"(lang={lang}, words={len(transcript_text.split())})"
            )
            with get_sync_db() as db:
                repository.save_transcript(
                    db,
                    video_id=video_id,
                    raw_text=transcript_text,
                    source=TranscriptSource.youtube_captions,
                    language=lang or "es",
                    word_count=len(transcript_text.split()),
                )
            return video_id

    except Exception as e:
        logger.warning(f"YouTube caption fetch failed for {video_id}: {e}")

    # ── Step 2: Whisper fallback (GPU) ────────────────────────────────────────
    logger.info(f"Video {video_id}: falling back to Whisper transcription")
    try:
        from src.input.whisper_transcriber import transcribe_audio
        from src.input.youtube_fetcher import YouTubeFetcher

        import asyncio
        loop = asyncio.new_event_loop()
        try:
            fetcher = YouTubeFetcher()
            audio_path = loop.run_until_complete(fetcher.download_audio(youtube_video_id))
        finally:
            loop.close()

        transcript_text = transcribe_audio(audio_path, language="es")

        if not transcript_text or len(transcript_text.split()) < 10:
            raise ValueError(f"Whisper produced empty/short transcript for {video_id}")

        logger.info(
            f"Video {video_id}: Whisper transcription complete "
            f"({len(transcript_text.split())} words)"
        )

        with get_sync_db() as db:
            repository.save_transcript(
                db,
                video_id=video_id,
                raw_text=transcript_text,
                source=TranscriptSource.whisper,
                language="es",
                word_count=len(transcript_text.split()),
            )

    except Exception as e:
        logger.error(f"Whisper transcription failed for {video_id}: {e}")
        with get_sync_db() as db:
            repository.update_video_status(
                db, video_id, ProcessingStatus.failed, error_message=str(e)
            )
        raise self.retry(exc=e)

    return video_id
