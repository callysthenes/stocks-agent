"""FastAPI channels router."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from src.input.youtube_fetcher import YouTubeFetcher
from src.models import Channel
from src.storage import repository
from src.storage.mariadb_client import get_sync_db
from src.tasks.fetch_tasks import fetch_channel_backfill

router = APIRouter()


class ChannelCreate(BaseModel):
    url: str


class ChannelResponse(BaseModel):
    id: str
    youtube_channel_id: str
    name: str
    url: str
    is_active: bool
    last_checked_at: str | None
    created_at: str

    class Config:
        from_attributes = True


@router.post("/", response_model=ChannelResponse)
async def add_channel(payload: ChannelCreate):
    """Add a new YouTube channel to monitor."""
    fetcher = YouTubeFetcher()
    channel_info = await fetcher.get_channel_info(payload.url)
    if not channel_info:
        raise HTTPException(status_code=422, detail="Could not fetch channel metadata from YouTube")

    with get_sync_db() as db:
        existing = repository.get_channel_by_yt_id(db, channel_info["youtube_channel_id"])
        if existing:
            raise HTTPException(status_code=409, detail="Channel already exists")

        channel = repository.create_channel(db, **channel_info)
        channel_id = channel.id

    # Trigger backfill asynchronously
    fetch_channel_backfill.apply_async(args=[channel_id, 5])

    with get_sync_db() as db:
        channel = repository.get_channel_by_yt_id(db, channel_info["youtube_channel_id"])
        return ChannelResponse(
            id=channel.id,
            youtube_channel_id=channel.youtube_channel_id,
            name=channel.name,
            url=channel.url,
            is_active=channel.is_active,
            last_checked_at=channel.last_checked_at,
            created_at=str(channel.created_at),
        )


@router.get("/")
async def list_channels():
    """List all monitored channels."""
    with get_sync_db() as db:
        channels = repository.get_all_active_channels(db)
        return [
            {
                "id": c.id,
                "name": c.name,
                "url": c.url,
                "is_active": c.is_active,
                "last_checked_at": c.last_checked_at,
            }
            for c in channels
        ]


@router.post("/{channel_id}/backfill")
async def trigger_backfill(channel_id: str, limit: int = 5):
    """Manually trigger a video backfill for a channel."""
    fetch_channel_backfill.apply_async(args=[channel_id, limit])
    return {"message": "Backfill triggered", "channel_id": channel_id}


@router.delete("/{channel_id}")
async def deactivate_channel(channel_id: str):
    """Deactivate a channel (stop monitoring)."""
    from sqlalchemy import update
    from src.models import Channel as ChannelModel
    with get_sync_db() as db:
        db.execute(update(ChannelModel).where(ChannelModel.id == channel_id).values(is_active=False))
    return {"message": "Channel deactivated"}
