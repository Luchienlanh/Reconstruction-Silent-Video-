from __future__ import annotations

import re


_APOSTROPHE_RE = re.compile(r"\s*'\s*")
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for LRS-style WER/CER.

    VTP often emits apostrophes as separate tokens, e.g. "doesn ' t".
    We collapse apostrophes and then remove punctuation so "doesn't" and
    "doesn ' t" compare as the same token "doesnt".
    """

    text = text.lower().replace("’", "'").replace("`", "'")
    text = _APOSTROPHE_RE.sub("", text)
    text = _NON_TEXT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def chars(text: str) -> list[str]:
    return list(normalize_text(text).replace(" ", ""))

