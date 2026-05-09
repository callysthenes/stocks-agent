"""
Celery fetch tasks — poll YouTube channels for new videos.
Runs on the 'io' queue (no GPU required).
"""
from datetime import datetime, timezone

from celery import chain
from loguru import logger

from src.celery_app import celery_app
from src.input.youtube_fetcher import YouTubeFetcher
from src.models import ProcessingStatus
from src.storage import repository
from src.storage.mariadb_client import get_sync_db


@celery_app.task(
    name="src.tasks.fetch_tasks.poll_all_channels",
    bind=True,
    max_retries=3,
    queue="io",
)
def poll_all_channels(self):
    """Poll all active channels for new videos. Triggered by Celery Beat."""
    logger.info("Polling all active channels for new videos")
    fetcher = YouTubeFetcher()

    with get_sync_db() as db:
        channels = repository.get_all_active_channels(db)

    if not channels:
        logger.info("No active channels to poll")
        return {"polled": 0, "new_videos": 0}

    total_new = 0
    for channel in channels:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            new_videos = loop.run_until_complete(
                _fetch_new_videos_for_channel(fetcher, channel)
            )
            loop.close()
            total_new += len(new_videos)
        except Exception as e:
            logger.error(f"Error polling channel {channel.name}: {e}")

    logger.info(f"Poll complete: {len(channels)} channels, {total_new} new videos")
    return {"polled": len(channels), "new_videos": total_new}


@celery_app.task(
    name="src.tasks.fetch_tasks.fetch_channel_backfill",
    bind=True,
    max_retries=3,
    queue="io",
)
def fetch_channel_backfill(self, channel_id: str, limit: int = 5):
    """
    Fetch the last N videos for a channel (initial backfill).
    Triggered manually or on channel creation.
    """
    logger.info(f"Starting backfill for channel {channel_id} (limit={limit})")
    fetcher = YouTubeFetcher()

    with get_sync_db() as db:
        from sqlalchemy import select
        from src.models import Channel
        channel = db.scalar(select(Channel).where(Channel.id == channel_id))
        if not channel:
            logger.error(f"Channel {channel_id} not found")
            return {"error": "Channel not found"}

        channel_url = channel.url
        channel_name = channel.name

    import asyncio
    loop = asyncio.new_event_loop()

    try:
        videos = loop.run_until_complete(
            fetcher.get_channel_videos(channel_url, limit=limit)
        )
    finally:
        loop.close()

    new_count = 0
    with get_sync_db() as db:
        for video_data in videos:
            existing = repository.get_video_by_yt_id(db, video_data["youtube_video_id"])
            if existing:
                continue

            video = repository.create_video(
                db,
                channel_id=channel_id,
                youtube_video_id=video_data["youtube_video_id"],
                title=video_data["title"],
                description=video_data.get("description", ""),
                published_at=video_data.get("published_at", ""),
                duration_seconds=video_data.get("duration_seconds", 0),
                thumbnail_url=video_data.get("thumbnail_url"),
                youtube_url=video_data["youtube_url"],
                processing_status=ProcessingStatus.pending,
            )
            new_count += 1

            # Chain: transcribe → ingest
            from src.tasks.transcription_tasks import transcribe_video
            from src.tasks.ingestion_tasks import ingest_video
            chain(
                transcribe_video.s(video.id),
                ingest_video.s(),
            ).apply_async()

    logger.info(f"Backfill complete for {channel_name}: {new_count} new videos queued")
    return {"channel_id": channel_id, "new_videos": new_count}


async def _fetch_new_videos_for_channel(fetcher: YouTubeFetcher, channel) -> list:
    """Internal helper to fetch and queue new videos for a single channel."""
    from src.tasks.transcription_tasks import transcribe_video
    from src.tasks.ingestion_tasks import ingest_video

    videos_data: list
    if channel.last_checked_at:
        videos_data = await fetcher.get_new_videos_since(
            channel.url, channel.last_checked_at, max_check=20
        )
    else:
        videos_data = await fetcher.get_channel_videos(channel.url, limit=5)

    new_videos = []
    with get_sync_db() as db:
        for video_data in videos_data:
            existing = repository.get_video_by_yt_id(db, video_data["youtube_video_id"])
            if existing:
                continue

            video = repository.create_video(
                db,
                channel_id=channel.id,
                youtube_video_id=video_data["youtube_video_id"],
                title=video_data["title"],
                description=video_data.get("description", ""),
                published_at=video_data.get("published_at", ""),
                duration_seconds=video_data.get("duration_seconds", 0),
                thumbnail_url=video_data.get("thumbnail_url"),
                youtube_url=video_data["youtube_url"],
                processing_status=ProcessingStatus.pending,
            )
            new_videos.append(video)

            # Chain the processing pipeline
            chain(
                transcribe_video.s(video.id),
                ingest_video.s(),
            ).apply_async()

        repository.update_channel_last_checked(db, channel.id)

    logger.info(f"Channel '{channel.name}': {len(new_videos)} new videos queued")
    return new_videos
