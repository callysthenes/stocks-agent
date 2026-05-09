# StocksAgent — Architecture & Implementation Plan

## Overview

StocksAgent is a production-grade AI system that monitors Spanish-language YouTube investment channels, extracts stock predictions, performs financial analysis, and delivers curated reports via Telegram and a web dashboard.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DELIVERY LAYER                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐        │
│  │ Telegram Bot  │  │ Streamlit Dash   │  │ FastAPI REST API   │        │
│  └──────┬───────┘  └────────┬─────────┘  └─────────┬──────────┘        │
├─────────┼──────────────────┼───────────────────────┼────────────────────┤
│         │          AGENT / ORCHESTRATION LAYER      │                    │
│         │   ┌──────────────────────────────────┐    │                    │
│         └──►│       LangGraph Supervisor        │◄──┘                    │
│             │  ┌──────────┐ ┌───────────┐ ┌────┴─────┐                  │
│             │  │ Research  │ │ Analysis  │ │  User    │                  │
│             │  │  Agent    │ │  Agent    │ │  Agent   │                  │
│             │  └─────┬────┘ └─────┬─────┘ └──────────┘                  │
│             │      DeepSeek V3 API (OpenAI-compatible)                   │
│             └────────┼────────────┼─────────────────────────────────────┘
│                      │            │                                       │
│             ┌────────▼────────────▼──────────┐                           │
│             │   LlamaIndex RAG Pipeline       │                           │
│             │   (ChromaDB retriever + HybridQ) │                          │
├─────────────┴────────────────────────────────────────────────────────────┤
│                      DATA STORAGE LAYER                                   │
│  ┌──────────────────┐    ┌───────────────────┐   shared UUID PK          │
│  │    ChromaDB       │◄──►│     MariaDB 11    │                           │
│  │  (embeddings +    │    │  (channels, videos,│                           │
│  │   transcript      │    │   predictions,    │                           │
│  │   chunks)         │    │   ground_truth)   │                           │
│  └──────────────────┘    └───────────────────┘                           │
├──────────────────────────────────────────────────────────────────────────┤
│                    DATA INGESTION LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐    │
│  │  Transcript   │  │  Metadata    │  │  Embedding   │  │ Ticker   │    │
│  │  Cleaner      │  │  Extractor   │  │  Generator   │  │ Extractor│    │
│  │  (Spanish NLP)│  │  (DeepSeek) │  │  (bge-m3)    │  │(DeepSeek)│    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘    │
├──────────────────────────────────────────────────────────────────────────┤
│                       INPUT LAYER                                         │
│  ┌──────────────────────┐   ┌──────────────────────┐                     │
│  │  YouTube Fetcher     │   │  Whisper Transcriber  │                     │
│  │  yt-dlp + yt-api     │   │  faster-whisper       │                     │
│  │  (captions → audio)  │   │  large-v3, CUDA GPU0  │                     │
│  └──────────────────────┘   └──────────────────────┘                     │
├──────────────────────────────────────────────────────────────────────────┤
│                     OBSERVABILITY                                         │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │              Langfuse (self-hosted) — traces, evals, costs         │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.11 | Runtime |
| Input | `yt-dlp` | latest | Audio download + metadata |
| Input | `youtube-transcript-api` | latest | Fetch existing captions |
| Transcription | `faster-whisper` | latest | Local Whisper (`large-v3`) on GPU 0 |
| Embeddings | `sentence-transformers` + `bge-m3` | latest | Spanish-optimized embeddings, GPU 0 |
| Vector DB | ChromaDB | 0.5.x | Embedding storage |
| SQL DB | MariaDB | 11.4 | Structured data, predictions, ground truth |
| Task Queue | Celery + Redis | 5.4 / 7.4 | Background processing + scheduling |
| RAG | LlamaIndex | 0.12.x | Retrieval pipeline |
| Agent Orchestration | LangGraph | 0.2.x | Multi-agent supervisor pattern |
| LLM | DeepSeek V3 | - | All agents (OpenAI-compatible API) |
| Financial Data | `yfinance` + `pandas-ta` | latest | Ticker data + technical indicators |
| Dashboard | Streamlit | 1.41+ | Web UI |
| Delivery | `python-telegram-bot` | 21.x | Telegram reports |
| Observability | Langfuse (self-hosted) | 2.x | Traces, evaluations, cost tracking |
| API | FastAPI | 0.115+ | REST endpoints |
| Containerization | Docker Compose | v2 | Multi-service + GPU passthrough |

---

## GPU Allocation

| GPU | Service | VRAM Usage | Notes |
|---|---|---|---|
| GPU 0 (RTX 3060 12GB) | faster-whisper `large-v3` fp16 | ~4.5 GB | Transcription (bursty) |
| GPU 0 (RTX 3060 12GB) | bge-m3 embedding model | ~2.2 GB | Concurrent with idle Whisper |
| GPU 1 (RTX 3060 12GB) | Ollama | Variable | Reserved for local LLM / future use |

Both GPU-bound services run via the `worker` Celery service with `CUDA_VISIBLE_DEVICES=0`. Celery concurrency is set to 2 to prevent OOM — GPU tasks queue behind each other naturally.

---

## Database Schema

### MariaDB

```
channels          — YouTube channels being tracked
videos            — Individual videos, processing status
transcripts       — Raw + cleaned transcripts per video
ticker_mentions   — Stocks mentioned with sentiment + context
predictions       — Explicit channel predictions (buy/sell/hold)
ground_truth      — Actual price outcomes vs predictions
channel_accuracy  — Materialized accuracy summary per channel
agent_reports     — Generated AI reports with Telegram delivery status
```

### ChromaDB Collection: `video_transcripts`

```
id:       {video_uuid}_{chunk_index}
document: Text chunk ~512 tokens (sentence-aware split, 50 token overlap)
embedding: bge-m3 vector (1024 dimensions)
metadata: { video_id, channel_id, channel_name, video_title,
            published_at, chunk_index, timestamp_start, timestamp_end }
```

**Cross-DB linking**: `video_id` in ChromaDB metadata is a UUID FK to `videos.id` in MariaDB.

---

## Multi-Agent Architecture (LangGraph)

```
User Query / Celery Trigger
        │
        ▼
  ┌─────────────┐
  │  SUPERVISOR  │  ← DeepSeek V3, routes based on intent
  └──────┬──────┘
         │ Command(goto=...)
    ┌────┼─────────────────────┐
    │    │                     │
    ▼    ▼                     ▼
┌──────┐ ┌──────────┐  ┌────────────┐
│RSRCH │ │ANALYSIS  │  │   USER     │
│AGENT │ │AGENT     │  │   AGENT    │
│      │ │          │  │            │
│• RAG │ │• yfinance│  │• Formatting│
│• Web │ │• pandas-ta│  │• Telegram  │
│• DB  │ │• DeepSeek│  │• Dashboard │
│  Q   │ │  reason  │  │  update    │
└──────┘ └──────────┘  └────────────┘
```

**Routing logic**: Supervisor reads state, decides: research first → analysis → user (format) → END. Can loop back if more research is needed.

---

## Data Pipeline Flow

### New Video Processing

```
1. Celery Beat: poll_channels() every 30min
   └─ yt-dlp: scan channel for new videos since last_checked
   └─ Create video record (status=pending) in MariaDB
   └─ Chain: transcribe → ingest → alert

2. transcribe_video(video_id)
   ├─ Try youtube-transcript-api (Spanish captions)
   ├─ Fallback: yt-dlp download audio → faster-whisper large-v3
   └─ Store raw transcript, set source

3. ingest_video(video_id)
   ├─ Clean transcript (filler words, ASR artifacts)
   ├─ DeepSeek: extract ticker mentions + sentiment (JSON mode)
   ├─ DeepSeek: extract explicit predictions (JSON mode)
   ├─ Store ticker_mentions + predictions in MariaDB
   ├─ Chunk transcript (512 tokens, 50 overlap, sentence-aware)
   ├─ bge-m3: generate embeddings
   └─ Store chunks + embeddings in ChromaDB

4. send_alert(video_id)
   └─ User Agent: format per-video Telegram alert with tickers + dashboard link
```

### Daily Ground Truth Evaluation

```
Celery Beat: evaluate_predictions() daily at 22:00 UTC
└─ For each unresolved prediction with elapsed timeframe:
   ├─ yfinance: fetch current price
   ├─ Calculate actual_change_pct and actual_direction
   ├─ Store in ground_truth table
   └─ Update channel_accuracy summary

Celery Beat: send_daily_report() daily at 22:30 UTC
└─ Agent pipeline: research + analysis → daily digest → Telegram
```

---

## Security

- All secrets via `.env` file (never committed, see `.env.example`)
- Langfuse and dashboard accessible only via Wireguard VPN
- MariaDB and Redis not exposed on public interfaces (internal Docker network)
- Telegram bot token and DeepSeek API key rotated immediately after any exposure

---

## Implementation Phases

| Phase | Scope |
|---|---|
| 1 | Infrastructure (Docker, DB schema, config) |
| 2 | Input layer (YouTube fetcher, captions, Whisper) |
| 3 | Ingestion (cleaning, chunking, embeddings, extraction) |
| 4 | RAG (LlamaIndex + ChromaDB) |
| 5 | Agents (LangGraph supervisor + 3 agents) |
| 6 | Prediction tracking + ground truth evaluation |
| 7 | Delivery (Telegram bot + report generator) |
| 8 | Dashboard (Streamlit 5 pages) |
| 9 | Observability (Langfuse integration) |
| 10 | Tests, CI, documentation |

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| YouTube IP blocking | Home server IP avoids cloud bans. Exponential backoff. yt-dlp cookies fallback. |
| yfinance API breaks | Redis cache (1h TTL). Alpha Vantage as fallback. Abstract behind interface. |
| DeepSeek downtime | Tenacity retry with backoff. Failed tasks requeued. Ollama GPU1 as fallback LLM. |
| GPU OOM | Celery concurrency=2. GPU tasks never run concurrently (single queue). |
| ChromaDB data loss | Persistent Docker volume. Embeddings are regenerable from stored transcripts. |
| Financial advice liability | Disclaimer on every report: "This is not financial advice." |

---

## Disclaimer

This system is for **educational and research purposes only**. All analysis and predictions are generated by AI and should not be considered financial advice. Always consult a qualified financial advisor before making investment decisions.
