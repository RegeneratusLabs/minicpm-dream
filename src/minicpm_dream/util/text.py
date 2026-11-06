"""Text normalization and similarity helpers (deterministic, no model)."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "as", "by", "from", "has", "have", "had", "do", "does",
    "did", "not", "no", "but", "if", "then", "than", "so", "we", "i", "you",
    "he", "she", "they", "them", "his", "her", "their", "our", "my", "me",
    "user", "users", "themselves", "its", "us",
}


def normalize_text(text: str) -> str:
    """NFKC, collapse whitespace, strip control chars. Keeps case and punctuation."""
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = _WS.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    return text.strip()


def norm_key(text: str) -> str:
    """Case-insensitive, punctuation-free key for duplicate detection."""
    text = normalize_text(text).casefold()
    return " ".join(_TOKEN.findall(text))


_SPELLING = {"favorite": "favourite", "favorites": "favourites"}


def tokens(text: str) -> list[str]:
    folded = normalize_text(text).casefold()
    for source, target in _SPELLING.items():
        folded = folded.replace(source, target)
    return _TOKEN.findall(folded)


def token_set(text: str, *, drop_stopwords: bool = True) -> set[str]:
    toks = tokens(text)
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOP]
    return set(toks)


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(short: str, long: str) -> float:
    """Fraction of the shorter text's tokens present in the longer text."""
    ss, ls = token_set(short), token_set(long)
    if not ss:
        return 0.0
    return len(ss & ls) / len(ss)


def quote_in_source(quote: str, source: str) -> bool:
    """True when the quote appears verbatim after whitespace/case normalization."""
    q = _WS.sub(" ", unicodedata.normalize("NFKC", quote or "")).strip().casefold()
    s = _WS.sub(" ", unicodedata.normalize("NFKC", source or "")).strip().casefold()
    return bool(q) and q in s


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(normalize_text(text)) if s.strip()]


def truncate(text: str, limit: int, suffix: str = " …") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def digits(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


_RETRACTION = re.compile(
    r"(?i)\b(no longer|not anymore|stopped using|stop using|don't use|do not use|"
    r"doesn't use|does not use|never use|cancelled|canceled|not true anymore|"
    r"that's wrong|that is wrong|forget (?:about )?(?:it|that))\b"
)


def is_retraction(text: str) -> bool:
    """A denial/retraction is not a new fact; it needs reconciliation, not storage."""
    return bool(_RETRACTION.search(normalize_text(text)))


_STATUS_ECHO = re.compile(
    r"(?i)^(?:the\s+)?(?:migration|change|update|deployment|task|action|note|status|upgrade)\b"
    r".{0,50}\b(?:recorded|noted|completed|done|applied)[.!]?$"
)
_STATUS_ONLY = re.compile(r"(?i)^(?:noted|recorded|understood|got it|done|okay)[.!]?$")
_NOTHING = re.compile(r"(?i)^(?:no|nothing)\b.{0,40}\b(?:recorded|found|changed|new|open loops)\b[.!]?$")


def is_status_echo(text: str) -> bool:
    """Assistant acknowledgements such as 'PostgreSQL migration recorded.'"""
    clean = normalize_text(text)
    return bool(_STATUS_ECHO.match(clean) or _STATUS_ONLY.match(clean) or _NOTHING.match(clean))


def is_probably_meta(text: str) -> bool:
    """Reject conversational meta-states the extractor sometimes emits."""
    low = normalize_text(text).casefold()
    patterns = (
        r"^(the )?(user|assistant|agent) (is|was|seems|appears) (asking|talking|discussing|trying|in a)",
        r"^(the )?(user|assistant|agent) (asked|said|mentioned|wants to know) (about|that)",
        r"^(this|the) (conversation|session|message|turn) (is|was|contains)",
        r"^(the )?(user|assistant) (is|was) (happy|sad|angry|frustrated|in a good mood|in a bad mood)",
        r"^(there is|there are) no (relevant|useful|durable) (information|fact|memory)",
        r"^(the )?(user|assistant|agent) (mentioned|talked about|asked about|discussed|said|reported|noted)\b",
        r"\bno details (were|was) (provided|given)\b",
        r"\bnot enough (information|context)\b",
        r"\b(is a person the user knows|user mentions)\b",
    )
    return any(re.search(p, low) for p in patterns)
