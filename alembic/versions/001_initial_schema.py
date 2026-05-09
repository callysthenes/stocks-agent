"""Initial database schema migration."""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "channels",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("youtube_channel_id", sa.String(64), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("url", sa.String(512), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("thumbnail_url", sa.String(512), nullable=True),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("last_checked_at", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_channels_youtube_channel_id", "channels", ["youtube_channel_id"])

    op.create_table(
        "videos",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("channel_id", sa.String(36), sa.ForeignKey("channels.id", ondelete="CASCADE"), nullable=False),
        sa.Column("youtube_video_id", sa.String(16), unique=True, nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("published_at", sa.String(32), nullable=True),
        sa.Column("duration_seconds", sa.Integer, nullable=True),
        sa.Column("thumbnail_url", sa.String(512), nullable=True),
        sa.Column("youtube_url", sa.String(512), nullable=False),
        sa.Column("transcript_source", sa.Enum("youtube_captions", "whisper", "none"), nullable=False, default="none"),
        sa.Column("processing_status", sa.Enum("pending", "fetching", "transcribing", "embedding", "completed", "failed"), nullable=False, default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_videos_channel_id", "videos", ["channel_id"])
    op.create_index("ix_videos_processing_status", "videos", ["processing_status"])

    op.create_table(
        "transcripts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("video_id", sa.String(36), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("language", sa.String(10), default="es", nullable=False),
        sa.Column("raw_text", sa.Text(4294967295), nullable=False),
        sa.Column("cleaned_text", sa.Text(4294967295), nullable=True),
        sa.Column("word_count", sa.Integer, nullable=True),
        sa.Column("chunk_count", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )

    op.create_table(
        "ticker_mentions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("video_id", sa.String(36), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker_symbol", sa.String(20), nullable=False),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("exchange", sa.String(50), nullable=True),
        sa.Column("sentiment", sa.Enum("bullish", "bearish", "neutral"), nullable=False),
        sa.Column("context_snippet", sa.Text, nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_ticker_mentions_ticker_symbol", "ticker_mentions", ["ticker_symbol"])
    op.create_index("ix_ticker_mentions_video_id", "ticker_mentions", ["video_id"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("video_id", sa.String(36), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("channel_id", sa.String(36), sa.ForeignKey("channels.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker_symbol", sa.String(20), nullable=False),
        sa.Column("prediction_type", sa.Enum("buy", "sell", "hold", "watch"), nullable=False),
        sa.Column("predicted_direction", sa.Enum("up", "down", "neutral"), nullable=False),
        sa.Column("target_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("timeframe_days", sa.Integer, nullable=True),
        sa.Column("price_at_prediction", sa.Numeric(12, 4), nullable=True),
        sa.Column("prediction_date", sa.String(10), nullable=True),
        sa.Column("context_snippet", sa.Text, nullable=True),
        sa.Column("is_evaluated", sa.Boolean, default=False, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_predictions_ticker_symbol", "predictions", ["ticker_symbol"])
    op.create_index("ix_predictions_is_evaluated", "predictions", ["is_evaluated"])

    op.create_table(
        "ground_truth",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("prediction_id", sa.String(36), sa.ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("price_at_evaluation", sa.Numeric(12, 4), nullable=True),
        sa.Column("actual_change_pct", sa.Numeric(8, 4), nullable=True),
        sa.Column("actual_direction", sa.Enum("up", "down", "neutral"), nullable=True),
        sa.Column("is_accurate", sa.Boolean, nullable=True),
        sa.Column("evaluation_date", sa.String(10), nullable=True),
        sa.Column("notes", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )

    op.create_table(
        "channel_accuracy",
        sa.Column("channel_id", sa.String(36), sa.ForeignKey("channels.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("total_predictions", sa.Integer, default=0),
        sa.Column("accurate_predictions", sa.Integer, default=0),
        sa.Column("accuracy_pct", sa.Numeric(5, 2), nullable=True),
        sa.Column("avg_return_pct", sa.Numeric(8, 4), nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )

    op.create_table(
        "agent_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("report_type", sa.Enum("daily_summary", "video_alert", "ticker_analysis", "user_query"), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("content_markdown", sa.Text(4294967295), nullable=False),
        sa.Column("tickers_analyzed", sa.JSON, nullable=True),
        sa.Column("video_id", sa.String(36), nullable=True),
        sa.Column("telegram_sent", sa.Boolean, default=False),
        sa.Column("telegram_message_id", sa.BigInteger, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )


def downgrade() -> None:
    op.drop_table("agent_reports")
    op.drop_table("channel_accuracy")
    op.drop_table("ground_truth")
    op.drop_table("predictions")
    op.drop_table("ticker_mentions")
    op.drop_table("transcripts")
    op.drop_table("videos")
    op.drop_table("channels")
