"""Celery application configuration."""
from celery import Celery
from celery.schedules import crontab
from src.config import settings

celery_app = Celery(
    "stocks_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.tasks.fetch_tasks",
        "src.tasks.transcription_tasks",
        "src.tasks.ingestion_tasks",
        "src.tasks.analysis_tasks",
        "src.tasks.report_tasks",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task behavior
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,  # Fair dispatch for long GPU tasks
    # Result expiry
    result_expires=86400,  # 24 hours
    # Routing — GPU tasks go to the "gpu" queue, IO to "io"
    task_routes={
        "src.tasks.transcription_tasks.*": {"queue": "gpu"},
        "src.tasks.ingestion_tasks.generate_embeddings": {"queue": "gpu"},
        "src.tasks.fetch_tasks.*": {"queue": "io"},
        "src.tasks.report_tasks.*": {"queue": "default"},
        "src.tasks.analysis_tasks.*": {"queue": "default"},
    },
    # Retry defaults
    task_default_retry_delay=60,
    task_max_retries=3,
    # Beat schedule (periodic tasks)
    beat_schedule={
        "poll-all-channels": {
            "task": "src.tasks.fetch_tasks.poll_all_channels",
            "schedule": crontab(
                minute=f"*/{settings.channel_poll_interval_minutes}"
            ),
            "options": {"queue": "io"},
        },
        "evaluate-predictions": {
            "task": "src.tasks.analysis_tasks.evaluate_pending_predictions",
            "schedule": crontab(
                hour=settings.prediction_eval_hour_utc,
                minute=settings.prediction_eval_minute_utc,
            ),
            "options": {"queue": "default"},
        },
        "send-daily-report": {
            "task": "src.tasks.report_tasks.send_daily_report",
            "schedule": crontab(
                hour=settings.daily_report_hour_utc,
                minute=settings.daily_report_minute_utc,
            ),
            "options": {"queue": "default"},
        },
    },
)
