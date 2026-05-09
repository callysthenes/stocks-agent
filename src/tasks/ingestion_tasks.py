"""
Celery ingestion tasks — chunk, embed, extract metadata, store.
Embedding generation runs on the 'gpu' queue.
"""
from loguru import logger

from src.celery_app import celery_app
from src.models import ProcessingStatus
from src.storage import repository
from src.storage.mariadb_client import get_sync_db


@celery_app.task(
    name="src.tasks.ingestion_tasks.ingest_video",
    bind=True,
    max_retries=2,
    queue="gpu",
    time_limit=1800,  # 30 min max
)
def ingest_video(self, video_id: str) -> dict:
    """
    Full ingestion pipeline for a transcribed video:
    clean → extract → chunk → embed → store → alert.

    Called as the second step in the Celery chain after transcribe_video.
    """
    logger.info(f"Starting ingestion for video {video_id}")

    with get_sync_db() as db:
        video = repository.get_video_by_id(db, video_id)
        if not video:
            logger.error(f"Video {video_id} not found in DB")
            return {"error": "Video not found"}

        transcript = video.transcript
        if not transcript or not transcript.raw_text:
            logger.error(f"No transcript found for video {video_id}")
            with get_sync_db() as db2:
                repository.update_video_status(
                    db2, video_id, ProcessingStatus.failed,
                    error_message="No transcript available"
                )
            return {"error": "No transcript"}

        raw_text = transcript.raw_text
        source = video.transcript_source

    try:
        from src.ingestion.pipeline import run_ingestion_pipeline
        result = run_ingestion_pipeline(
            video_id=video_id,
            raw_transcript=raw_text,
            transcript_source=source,
        )

        # Trigger per-video alert after successful ingestion
        from src.tasks.report_tasks import send_video_alert
        send_video_alert.apply_async(args=[video_id], countdown=5)

        return result

    except Exception as e:
        logger.error(f"Ingestion failed for video {video_id}: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    name="src.tasks.ingestion_tasks.reprocess_video",
    bind=True,
    max_retries=1,
    queue="gpu",
)
def reprocess_video(self, video_id: str) -> dict:
    """
    Re-run the full pipeline for an existing video (e.g., after model updates).
    Deletes existing ChromaDB chunks before re-ingesting.
    """
    logger.info(f"Reprocessing video {video_id}")

    from src.storage.chroma_client import delete_video_chunks
    delete_video_chunks(video_id)

    with get_sync_db() as db:
        repository.update_video_status(db, video_id, ProcessingStatus.pending)

    # Retrigger the full chain
    from src.tasks.transcription_tasks import transcribe_video
    from celery import chain
    return chain(transcribe_video.s(video_id), ingest_video.s()).apply_async()
