from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Iterable


BLANK_TOKEN = "<blank>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _keep_text_char(ch: str) -> str:
    if ch.isspace():
        return " "
    if unicodedata.category(ch).startswith(("L", "N")):
        return ch
    return " "


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").replace("\ufeff", " ").lower()
    text = "".join(_keep_text_char(ch) for ch in text)
    return " ".join(text.split())


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")


def normalize_text_nodiac(text: str) -> str:
    return normalize_text(strip_diacritics(text))


def tokenize_text_unit(text: str, text_unit: str = "syllable_nodiac") -> list[str]:
    if text_unit == "char":
        normalized = normalize_text(text)
        return list(normalized) if normalized else [UNK_TOKEN]
    if text_unit == "char_nodiac":
        normalized = normalize_text_nodiac(text)
        return list(normalized) if normalized else [UNK_TOKEN]
    if text_unit == "syllable_nodiac":
        normalized = normalize_text_nodiac(text)
        return normalized.split() if normalized else [UNK_TOKEN]
    raise ValueError(f"Unsupported text_unit={text_unit}")


def build_vocab(texts: Iterable[str], min_freq: int = 1, text_unit: str = "syllable_nodiac") -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_text_unit(text, text_unit=text_unit))
    tokens = [tok for tok, freq in counter.items() if freq >= int(min_freq) and tok not in {BLANK_TOKEN, PAD_TOKEN, UNK_TOKEN}]
    tokens.sort(key=lambda tok: (-counter[tok], tok))
    vocab = {BLANK_TOKEN: 0, PAD_TOKEN: 1, UNK_TOKEN: 2}
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def text_to_ids(text: str, vocab: dict[str, int], text_unit: str = "syllable_nodiac") -> list[int]:
    unk = int(vocab.get(UNK_TOKEN, 2))
    return [int(vocab.get(tok, unk)) for tok in tokenize_text_unit(text, text_unit=text_unit)]


def ids_to_text(ids: Iterable[int], vocab: dict[str, int]) -> str:
    inv = {idx: tok for tok, idx in vocab.items()}
    toks = []
    for idx in ids:
        tok = inv.get(int(idx), "")
        if tok in {BLANK_TOKEN, PAD_TOKEN, UNK_TOKEN}:
            continue
        toks.append(tok)
    if any(len(tok) > 1 for tok in toks):
        return " ".join(toks).strip()
    return "".join(toks).strip()


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
    ref = list(normalize_text_nodiac(reference))
    hyp = list(normalize_text_nodiac(hypothesis))
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text_nodiac(reference).split()
    hyp = normalize_text_nodiac(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))

