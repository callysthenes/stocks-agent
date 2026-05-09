"""
Report generator — formats reports for Telegram and dashboard.
"""
from datetime import date, datetime
from typing import Any

from src.config import settings


def format_video_alert(
    channel_name: str,
    video_title: str,
    youtube_url: str,
    tickers: list[str],
    analysis: str,
) -> str:
    """Format a per-video alert for Telegram."""
    tickers_str = ", ".join(tickers) if tickers else "Ninguno identificado"
    return (
        f"📹 *Nuevo Vídeo Analizado*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📺 Canal: {channel_name}\n"
        f"🎬 Vídeo: {video_title[:80]}\n"
        f"📈 Tickers: {tickers_str}\n\n"
        f"{analysis}\n\n"
        f"🔗 [Ver vídeo]({youtube_url})\n"
        f"📊 Dashboard: {settings.dashboard_url}"
    )


def format_daily_digest(
    processed_videos: int,
    active_tickers: list[str],
    channel_accuracy: list[dict[str, Any]],
    analysis_highlights: str,
    report_date: date | None = None,
) -> str:
    """Format the daily digest report for Telegram."""
    today = (report_date or date.today()).strftime("%d/%m/%Y")
    tickers_str = ", ".join(active_tickers[:10]) if active_tickers else "Ninguno"

    accuracy_lines = []
    for ch in channel_accuracy[:5]:
        acc = ch.get("accuracy_pct")
        name = ch.get("channel_name", "Unknown")
        if acc is not None:
            emoji = "🟢" if acc >= 60 else "🟡" if acc >= 40 else "🔴"
            accuracy_lines.append(f"  {emoji} {name}: {acc:.1f}%")
        else:
            accuracy_lines.append(f"  ⚪ {name}: Sin datos")

    accuracy_section = "\n".join(accuracy_lines) if accuracy_lines else "  Sin datos aún"

    report = (
        f"📈 *Resumen Diario — {today}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📹 Vídeos procesados: {processed_videos}\n"
        f"📊 Tickers activos: {tickers_str}\n\n"
        f"*Precisión de Canales:*\n{accuracy_section}\n\n"
        f"{analysis_highlights[:2000]}\n\n"
        f"📊 Dashboard completo: {settings.dashboard_url}\n\n"
        f"⚠️ _No es asesoramiento financiero._"
    )
    return report[:4096]


def format_ticker_analysis(
    ticker: str,
    technical: dict[str, Any],
    fundamentals: dict[str, Any],
    channel_context: str,
    news: list[dict[str, Any]],
) -> str:
    """Format a single ticker analysis for display."""
    company = fundamentals.get("company_name", ticker)
    price = technical.get("current_price", "N/A")
    change_1d = technical.get("price_change_1d_pct", "N/A")
    rsi = technical.get("rsi_14", "N/A")
    trend = technical.get("trend", "N/A")
    pe = fundamentals.get("pe_ratio", "N/A")
    sector = fundamentals.get("sector", "N/A")

    news_lines = "\n".join(
        f"  • [{n['published_at']}] {n['title'][:80]}" for n in news[:3]
    ) if news else "  Sin noticias recientes"

    return (
        f"📊 *{ticker}* — {company}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Precio: {price} ({change_1d:+.2f}% hoy)\n"
        f"📉 RSI(14): {rsi} | Tendencia: {trend}\n"
        f"💼 P/E: {pe} | Sector: {sector}\n\n"
        f"📰 *Noticias:*\n{news_lines}\n\n"
        f"📺 *Canales:*\n{channel_context[:500]}\n\n"
        f"⚠️ _No es asesoramiento financiero._"
    )
