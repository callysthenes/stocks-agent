"""FastAPI videos router."""
from fastapi import APIRouter, Query
from sqlalchemy import select

from src.models import Video
from src.storage import repository
from src.storage.mariadb_client import get_sync_db

router = APIRouter()


@router.get("/")
async def list_videos(limit: int = Query(20, le=100), offset: int = 0):
    """List recent videos with pagination."""
    from sqlalchemy import select, desc
    with get_sync_db() as db:
        videos = db.scalars(
            select(Video).order_by(desc(Video.created_at)).offset(offset).limit(limit)
        ).all()
        return [
            {
                "id": v.id,
                "youtube_video_id": v.youtube_video_id,
                "title": v.title,
                "channel_id": v.channel_id,
                "processing_status": v.processing_status,
                "transcript_source": v.transcript_source,
                "published_at": v.published_at,
                "youtube_url": v.youtube_url,
                "duration_seconds": v.duration_seconds,
            }
            for v in videos
        ]


@router.get("/{video_id}")
async def get_video(video_id: str):
    """Get full details for a single video."""
    with get_sync_db() as db:
        video = repository.get_video_by_id(db, video_id)
        if not video:
            from fastapi import HTTPException
            raise HTTPException(404, detail="Video not found")
        return {
            "id": video.id,
            "youtube_video_id": video.youtube_video_id,
            "title": video.title,
            "description": video.description,
            "channel_id": video.channel_id,
            "processing_status": video.processing_status,
            "transcript_source": video.transcript_source,
            "published_at": video.published_at,
            "youtube_url": video.youtube_url,
            "duration_seconds": video.duration_seconds,
            "error_message": video.error_message,
            "ticker_mentions": [
                {
                    "ticker_symbol": m.ticker_symbol,
                    "company_name": m.company_name,
                    "sentiment": m.sentiment,
                    "exchange": m.exchange,
                }
                for m in video.ticker_mentions
            ],
        }
