"""
Spanish transcript cleaner.
Removes ASR artifacts, normalizes text, strips filler words,
and prepares transcripts for chunking and embedding.
"""
import re
import unicodedata

from loguru import logger

# Spanish filler words and common ASR artifacts to remove/normalize
_FILLER_WORDS = {
    r"\beh+\b",
    r"\bahh?\b",
    r"\bumm?\b",
    r"\bmmm+\b",
    r"\bpues\b",
    r"\boa+\b",
    r"\bbue+no\b",
    r"\bvenga\b",
    r"\bvale\b",
    r"\bessste\b",
    r"\besssto\b",
    r"\besto+\b",
    r"\byyy+\b",
    r"\bpero\s+bueno\b",
}

_FILLER_RE = re.compile(
    "|".join(_FILLER_WORDS),
    flags=re.IGNORECASE | re.UNICODE,
)

# Patterns for common ASR artifacts
_ARTIFACTS = [
    (re.compile(r"\[[\w\s]+\]"), " "),         # [Music], [Applause] etc.
    (re.compile(r"\([\w\s]+\)"), " "),          # (inaudible) etc.
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&#39;"), "'"),
    (re.compile(r"\s{2,}"), " "),               # Multiple spaces → single
    (re.compile(r"\.{3,}"), "…"),              # Multiple dots → ellipsis
    (re.compile(r"-{2,}"), "—"),               # Multiple dashes → em dash
]

# Sentence boundary regex (simple heuristic for Spanish)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ])")


def normalize_unicode(text: str) -> str:
    """Normalize Unicode (NFC) and strip control characters."""
    text = unicodedata.normalize("NFC", text)
    # Remove control characters except newlines and tabs
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t")
    return text


def remove_artifacts(text: str) -> str:
    """Remove bracketed annotations and HTML entities."""
    for pattern, replacement in _ARTIFACTS:
        text = pattern.sub(replacement, text)
    return text.strip()


def remove_filler_words(text: str) -> str:
    """Remove common Spanish filler words and hesitation sounds."""
    return _FILLER_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_transcript(raw_text: str) -> str:
    """
    Full cleaning pipeline for a raw transcript.

    Steps:
        1. Unicode normalization
        2. Remove ASR artifacts (brackets, HTML entities)
        3. Remove filler words
        4. Normalize whitespace

    Returns:
        Cleaned transcript string.
    """
    if not raw_text:
        return ""

    text = normalize_unicode(raw_text)
    text = remove_artifacts(text)
    text = remove_filler_words(text)
    text = normalize_whitespace(text)

    word_count_before = len(raw_text.split())
    word_count_after = len(text.split())
    logger.debug(
        f"Transcript cleaned: {word_count_before} → {word_count_after} words "
        f"({word_count_before - word_count_after} removed)"
    )
    return text


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split()) if text else 0
