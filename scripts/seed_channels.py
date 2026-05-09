"""
Seed initial YouTube channels from a predefined list.
Usage: python scripts/seed_channels.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

CHANNELS = [
    "https://www.youtube.com/@lamagiadelabolsa",
    # Add more channels here:
    # "https://www.youtube.com/@ChannelName",
]


async def seed():
    from src.input.youtube_fetcher import YouTubeFetcher
    from src.storage import repository
    from src.storage.mariadb_client import get_sync_db, init_db
    from src.tasks.fetch_tasks import fetch_channel_backfill

    # Initialize DB
    await init_db()
    fetcher = YouTubeFetcher()

    for channel_url in CHANNELS:
        print(f"\nProcessing: {channel_url}")
        channel_info = await fetcher.get_channel_info(channel_url)
        if not channel_info:
            print(f"  ERROR: Could not fetch info for {channel_url}")
            continue

        with get_sync_db() as db:
            existing = repository.get_channel_by_yt_id(db, channel_info["youtube_channel_id"])
            if existing:
                print(f"  SKIP: Channel '{channel_info['name']}' already exists")
                continue

            channel = repository.create_channel(db, **channel_info)
            channel_id = channel.id

        print(f"  CREATED: {channel_info['name']} (id={channel_id})")
        print(f"  Triggering backfill (last 5 videos)...")
        fetch_channel_backfill.apply_async(args=[channel_id, 5])
        print(f"  Backfill queued")


if __name__ == "__main__":
    asyncio.run(seed())
