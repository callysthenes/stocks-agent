"""
Semantic text chunker for transcript ingestion.
Splits cleaned transcripts into overlapping chunks suitable for embedding,
respecting sentence boundaries.
"""
import re
from dataclasses import dataclass

from loguru import logger

from src.config import settings


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    text: str
    chunk_index: int
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word for Spanish text."""
    return int(len(text.split()) * 1.33)


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a simple Spanish-aware heuristic.
    Handles abbreviations like 'Sr.', 'Dra.', decimal numbers, etc.
    """
    # Protect common abbreviations from being split
    protected = re.sub(
        r"\b(Sr|Sra|Dr|Dra|Prof|Fig|Ej|Aprox|etc|vs|núm|art|pág)\.",
        r"\1<DOT>",
        text,
    )
    # Protect decimal numbers: 3.14, $1.50
    protected = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", protected)

    # Split on sentence-ending punctuation followed by space + capital
    sentences = re.split(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÜÑ\d\"])", protected)

    # Restore protected dots
    return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.

    Args:
        text: Cleaned transcript text.
        chunk_size: Target chunk size in tokens (default from settings).
        chunk_overlap: Overlap in tokens between consecutive chunks.

    Returns:
        List of Chunk objects with text and metadata.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text:
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        # If a single sentence exceeds chunk_size, split it by words
        if sentence_tokens > chunk_size:
            words = sentence.split()
            sub_chunk_words: list[str] = []
            sub_tokens = 0
            for word in words:
                word_tokens = _estimate_tokens(word)
                if sub_tokens + word_tokens > chunk_size and sub_chunk_words:
                    chunks.append(Chunk(
                        text=" ".join(sub_chunk_words),
                        chunk_index=chunk_index,
                        token_estimate=sub_tokens,
                    ))
                    chunk_index += 1
                    # Overlap: keep last N tokens worth of words
                    overlap_words = sub_chunk_words[-_tokens_to_words(chunk_overlap):]
                    sub_chunk_words = overlap_words + [word]
                    sub_tokens = _estimate_tokens(" ".join(sub_chunk_words))
                else:
                    sub_chunk_words.append(word)
                    sub_tokens += word_tokens
            if sub_chunk_words:
                current_sentences.append(" ".join(sub_chunk_words))
                current_tokens += _estimate_tokens(" ".join(sub_chunk_words))
            continue

        # If adding this sentence would exceed chunk_size, emit current chunk
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunks.append(Chunk(
                text=" ".join(current_sentences),
                chunk_index=chunk_index,
                token_estimate=current_tokens,
            ))
            chunk_index += 1

            # Overlap: keep sentences from the end that fit within chunk_overlap tokens
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for prev_sentence in reversed(current_sentences):
                st = _estimate_tokens(prev_sentence)
                if overlap_tokens + st <= chunk_overlap:
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_tokens += st
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Emit the final chunk
    if current_sentences:
        chunks.append(Chunk(
            text=" ".join(current_sentences),
            chunk_index=chunk_index,
            token_estimate=current_tokens,
        ))

    logger.debug(
        f"Chunked {_estimate_tokens(text)} tokens into {len(chunks)} chunks "
        f"(target={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def _tokens_to_words(token_count: int) -> int:
    """Convert approximate token count to word count."""
    return max(1, int(token_count / 1.33))
