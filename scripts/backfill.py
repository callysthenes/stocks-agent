"""
Manual backfill trigger for existing channels.
Usage: python scripts/backfill.py [channel_id] [--limit=10]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Trigger video backfill for channels")
    parser.add_argument("channel_id", nargs="?", help="Specific channel UUID (default: all)")
    parser.add_argument("--limit", type=int, default=5, help="Videos per channel (default: 5)")
    args = parser.parse_args()

    from src.storage import repository
    from src.storage.mariadb_client import get_sync_db
    from src.tasks.fetch_tasks import fetch_channel_backfill

    with get_sync_db() as db:
        if args.channel_id:
            from sqlalchemy import select
            from src.models import Channel
            channels = [db.scalar(select(Channel).where(Channel.id == args.channel_id))]
            channels = [c for c in channels if c]
        else:
            channels = repository.get_all_active_channels(db)

    if not channels:
        print("No channels found.")
        return

    for channel in channels:
        print(f"Queueing backfill for: {channel.name} (limit={args.limit})")
        fetch_channel_backfill.apply_async(args=[channel.id, args.limit])

    print(f"\nBackfill queued for {len(channels)} channel(s).")
    print("Check the Celery worker logs for progress.")


if __name__ == "__main__":
    main()
