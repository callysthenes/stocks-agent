"""
YouTube channel and video fetcher using yt-dlp.
Handles metadata extraction and audio download for Whisper.
"""
import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import yt_dlp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings


class YouTubeFetcher:
    """Fetches video metadata and audio from YouTube channels."""

    def __init__(self) -> None:
        self._base_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "ignoreerrors": True,
        }

    # ── Channel metadata ──────────────────────────────────────────────────────

    async def get_channel_info(self, channel_url: str) -> dict[str, Any] | None:
        """Fetch channel metadata (name, id, description, thumbnail)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_channel_info_sync, channel_url)

    def _get_channel_info_sync(self, channel_url: str) -> dict[str, Any] | None:
        opts = {**self._base_opts, "playlistend": 1}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                if not info:
                    return None
                return {
                    "youtube_channel_id": info.get("channel_id") or info.get("id", ""),
                    "name": info.get("channel") or info.get("uploader") or info.get("title", ""),
                    "url": channel_url,
                    "description": info.get("description", ""),
                    "thumbnail_url": self._pick_thumbnail(info.get("thumbnails")),
                }
        except Exception as e:
            logger.error(f"Failed to fetch channel info for {channel_url}: {e}")
            return None

    # ── Video listing ─────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    async def get_channel_videos(
        self, channel_url: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Fetch the most recent N videos from a channel."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_channel_videos_sync, channel_url, limit
        )

    def _get_channel_videos_sync(
        self, channel_url: str, limit: int
    ) -> list[dict[str, Any]]:
        opts = {
            **self._base_opts,
            "playlistend": limit,
        }
        # Some channels have no /videos tab (streams-only, etc.) — try tabs in order.
        tabs_to_try = ["/videos", "/streams", ""]
        for tab in tabs_to_try:
            url = f"{channel_url}{tab}"
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    result = ydl.extract_info(url, download=False)
                if not result or "entries" not in result:
                    continue
                entries = [e for e in result["entries"] if e]
                if not entries:
                    continue
                videos = []
                for entry in entries:
                    video_id = entry.get("id") or entry.get("url", "").split("v=")[-1]
                    if not video_id:
                        continue
                    videos.append(self._normalize_video_entry(entry, channel_url))
                if videos:
                    if tab != "/videos":
                        logger.info(f"Channel {channel_url}: used tab '{tab or '(root)'}' to find {len(videos)} videos")
                    return videos
            except Exception as e:
                logger.warning(f"Channel {channel_url} tab '{tab}' failed: {e}")
                continue
        logger.error(f"No videos found for {channel_url} across all tabs")
        return []

    async def get_new_videos_since(
        self, channel_url: str, since_iso: str, max_check: int = 20
    ) -> list[dict[str, Any]]:
        """Return videos published after `since_iso` date (ISO 8601 string)."""
        all_videos = await self.get_channel_videos(channel_url, limit=max_check)
        since_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        new_videos = []
        for video in all_videos:
            pub = video.get("published_at")
            if pub:
                try:
                    pub_dt = datetime.strptime(pub, "%Y%m%d").replace(tzinfo=timezone.utc)
                    if pub_dt > since_dt:
                        new_videos.append(video)
                except ValueError:
                    pass
        return new_videos

    # ── Audio download ────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        reraise=True,
    )
    async def download_audio(self, youtube_video_id: str) -> str:
        """Download audio as mp3 for Whisper transcription. Returns file path."""
        output_path = os.path.join(settings.media_dir, youtube_video_id)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._download_audio_sync, youtube_video_id, output_path
        )

    def _download_audio_sync(self, video_id: str, output_path: str) -> str:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url])
        mp3_path = f"{output_path}.mp3"
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"Audio download failed for {video_id}")
        logger.debug(f"Downloaded audio for {video_id} → {mp3_path}")
        return mp3_path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalize_video_entry(self, entry: dict, channel_url: str) -> dict[str, Any]:
        video_id = entry.get("id", "")
        return {
            "youtube_video_id": video_id,
            "title": entry.get("title", ""),
            "description": entry.get("description", ""),
            "published_at": entry.get("upload_date", ""),
            "duration_seconds": entry.get("duration") or 0,
            "thumbnail_url": self._pick_thumbnail(entry.get("thumbnails")),
            "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
            "channel_url": channel_url,
        }

    @staticmethod
    def _pick_thumbnail(thumbnails: list | None) -> str | None:
        if not thumbnails:
            return None
        # Pick the highest-resolution thumbnail
        sorted_thumbs = sorted(
            [t for t in thumbnails if isinstance(t, dict) and t.get("url")],
            key=lambda t: t.get("width", 0) or 0,
            reverse=True,
        )
        return sorted_thumbs[0]["url"] if sorted_thumbs else None
