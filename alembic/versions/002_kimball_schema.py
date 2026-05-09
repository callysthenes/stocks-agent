"""
002 — Kimball schema: dim_tickers dimension table + enriched prediction/mention columns.

Adds:
  - dim_tickers: normalised ticker/company master dimension
  - predictions: entry_price, stop_loss, recommendation, is_long_term,
                 confidence_score, analyst_reasoning
  - ticker_mentions: dim_ticker_id FK, mention_count
  - agent_reports: channel_id FK, telegram_chat_id
"""
import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── 1. dim_tickers dimension table ────────────────────────────────────────
    op.create_table(
        "dim_tickers",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("ticker_symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("exchange", sa.String(50), nullable=True),
        sa.Column("sector", sa.String(100), nullable=True),
        sa.Column("industry", sa.String(100), nullable=True),
        sa.Column("currency", sa.String(10), nullable=True),
        sa.Column("country", sa.String(50), nullable=True),
        sa.Column("is_index", sa.Boolean, default=False, nullable=False),
        sa.Column("is_etf", sa.Boolean, default=False, nullable=False),
        sa.Column("is_crypto", sa.Boolean, default=False, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_dim_tickers_ticker_symbol", "dim_tickers", ["ticker_symbol"], unique=True)

    # ── 2. predictions — new columns ──────────────────────────────────────────
    op.add_column("predictions", sa.Column(
        "recommendation",
        sa.Enum("buy", "accumulate", "hold", "reduce", "sell", "avoid"),
        nullable=True,
    ))
    op.add_column("predictions", sa.Column("entry_price", sa.Numeric(12, 4), nullable=True))
    op.add_column("predictions", sa.Column("stop_loss", sa.Numeric(12, 4), nullable=True))
    op.add_column("predictions", sa.Column("is_long_term", sa.Boolean, nullable=True))
    op.add_column("predictions", sa.Column("confidence_score", sa.Float, nullable=True))
    op.add_column("predictions", sa.Column("analyst_reasoning", sa.Text, nullable=True))
    op.create_index("ix_predictions_recommendation", "predictions", ["recommendation"])
    # ix_predictions_channel_id already exists from migration 001
    op.create_index("ix_predictions_prediction_date", "predictions", ["prediction_date"])

    # ── 3. ticker_mentions — dim_ticker_id + mention_count ────────────────────
    op.add_column("ticker_mentions", sa.Column(
        "dim_ticker_id",
        sa.String(36),
        sa.ForeignKey("dim_tickers.id", ondelete="SET NULL"),
        nullable=True,
    ))
    op.add_column("ticker_mentions", sa.Column(
        "mention_count", sa.Integer, nullable=False, server_default="1"
    ))
    op.create_index("ix_ticker_mentions_dim_ticker_id", "ticker_mentions", ["dim_ticker_id"])

    # ── 4. agent_reports — channel_id FK + telegram_chat_id ──────────────────
    op.add_column("agent_reports", sa.Column(
        "channel_id",
        sa.String(36),
        sa.ForeignKey("channels.id", ondelete="SET NULL"),
        nullable=True,
    ))
    op.add_column("agent_reports", sa.Column("telegram_chat_id", sa.String(32), nullable=True))
    op.create_index("ix_agent_reports_channel_id", "agent_reports", ["channel_id"])
    op.create_index("ix_agent_reports_video_id", "agent_reports", ["video_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_reports_video_id", "agent_reports")
    op.drop_index("ix_agent_reports_channel_id", "agent_reports")
    op.drop_column("agent_reports", "telegram_chat_id")
    op.drop_column("agent_reports", "channel_id")

    op.drop_index("ix_ticker_mentions_dim_ticker_id", "ticker_mentions")
    op.drop_column("ticker_mentions", "mention_count")
    op.drop_column("ticker_mentions", "dim_ticker_id")

    op.drop_index("ix_predictions_prediction_date", "predictions")
    # ix_predictions_channel_id belongs to migration 001, leave it
    op.drop_index("ix_predictions_recommendation", "predictions")
    op.drop_column("predictions", "analyst_reasoning")
    op.drop_column("predictions", "confidence_score")
    op.drop_column("predictions", "is_long_term")
    op.drop_column("predictions", "stop_loss")
    op.drop_column("predictions", "entry_price")
    op.drop_column("predictions", "recommendation")

    op.drop_index("ix_dim_tickers_ticker_symbol", "dim_tickers")
    op.drop_table("dim_tickers")
