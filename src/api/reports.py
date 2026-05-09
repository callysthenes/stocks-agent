"""FastAPI reports and predictions routers."""
from fastapi import APIRouter, Query
from sqlalchemy import desc, select

from src.models import AgentReport, Prediction
from src.storage.mariadb_client import get_sync_db

router = APIRouter()


@router.get("/")
async def list_reports(limit: int = Query(20, le=100)):
    """List recent AI-generated reports."""
    with get_sync_db() as db:
        reports = db.scalars(
            select(AgentReport).order_by(desc(AgentReport.created_at)).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "report_type": r.report_type,
                "title": r.title,
                "telegram_sent": r.telegram_sent,
                "tickers_analyzed": r.tickers_analyzed,
                "created_at": str(r.created_at),
            }
            for r in reports
        ]


@router.get("/{report_id}")
async def get_report(report_id: str):
    """Get full content of a report."""
    with get_sync_db() as db:
        report = db.scalar(select(AgentReport).where(AgentReport.id == report_id))
        if not report:
            from fastapi import HTTPException
            raise HTTPException(404, "Report not found")
        return {
            "id": report.id,
            "report_type": report.report_type,
            "title": report.title,
            "content_markdown": report.content_markdown,
            "tickers_analyzed": report.tickers_analyzed,
            "telegram_sent": report.telegram_sent,
            "created_at": str(report.created_at),
        }
