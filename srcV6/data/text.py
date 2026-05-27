from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable


BLANK_TOKEN = "<blank>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

_PUNCT_RE = re.compile(r"[^\w\sÀ-ỹà-ỹĐđ]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", " ").lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")


def normalize_text_nodiac(text: str) -> str:
    return normalize_text(strip_diacritics(text))


def tokenize_chars(text: str) -> list[str]:
    normalized = normalize_text(text)
    return list(normalized) if normalized else [UNK_TOKEN]


def tokenize_chars_nodiac(text: str) -> list[str]:
    normalized = normalize_text_nodiac(text)
    return list(normalized) if normalized else [UNK_TOKEN]


def tokenize_syllables_nodiac(text: str) -> list[str]:
    normalized = normalize_text_nodiac(text)
    return normalized.split() if normalized else [UNK_TOKEN]


def tokenize_text_unit(text: str, text_unit: str = "char") -> list[str]:
    if text_unit == "char":
        return tokenize_chars(text)
    if text_unit == "char_nodiac":
        return tokenize_chars_nodiac(text)
    if text_unit == "syllable_nodiac":
        return tokenize_syllables_nodiac(text)
    raise ValueError(f"Unsupported text_unit={text_unit}")


def build_vocab(texts: Iterable[str], min_freq: int = 1, text_unit: str = "char") -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_text_unit(text, text_unit=text_unit))
    chars = [ch for ch, freq in counter.items() if freq >= int(min_freq) and ch not in {BLANK_TOKEN, PAD_TOKEN, UNK_TOKEN}]
    chars.sort(key=lambda ch: (-counter[ch], ch))
    vocab = {BLANK_TOKEN: 0, PAD_TOKEN: 1, UNK_TOKEN: 2}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def build_char_vocab(texts: Iterable[str], min_freq: int = 1) -> dict[str, int]:
    return build_vocab(texts, min_freq=min_freq, text_unit="char")


def ids_to_text(ids: Iterable[int], vocab: dict[str, int]) -> str:
    inv = {idx: tok for tok, idx in vocab.items()}
    chars = []
    for idx in ids:
        tok = inv.get(int(idx), "")
        if tok in {BLANK_TOKEN, PAD_TOKEN, UNK_TOKEN}:
            continue
        chars.append(tok)
    if any(len(ch) > 1 for ch in chars):
        return " ".join(chars).strip()
    return "".join(chars).strip()


def greedy_decode(ids: Iterable[int], vocab: dict[str, int], blank_id: int = 0) -> str:
    collapsed = []
    prev = None
    for raw in ids:
        idx = int(raw)
        if idx != blank_id and idx != prev:
            collapsed.append(idx)
        prev = idx
    return ids_to_text(collapsed, vocab)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = list(normalize_text(reference))
    hyp = list(normalize_text(hypothesis))
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))
