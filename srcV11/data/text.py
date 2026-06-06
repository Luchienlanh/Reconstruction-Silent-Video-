from __future__ import annotations

import unicodedata
from typing import Iterable


BLANK_TOKEN = "<blank>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_char_en_vocab() -> dict[str, int]:
    tokens = [BLANK_TOKEN, PAD_TOKEN, UNK_TOKEN, " "]
    tokens.extend(chr(ord("a") + i) for i in range(26))
    tokens.extend(str(i) for i in range(10))
    return {token: idx for idx, token in enumerate(tokens)}


CHAR_EN_VOCAB = build_char_en_vocab()


def normalize_english(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").lower().replace("\ufeff", " ")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("'", "").replace("`", "")
    chars = []
    for ch in text:
        if ch.isascii() and (ch.isalpha() or ch.isdigit() or ch.isspace()):
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def text_to_ids(text: str, vocab: dict[str, int] | None = None) -> list[int]:
    vocab = vocab or CHAR_EN_VOCAB
    unk = int(vocab.get(UNK_TOKEN, 2))
    normalized = normalize_english(text)
    if not normalized:
        return []
    return [int(vocab.get(ch, unk)) for ch in normalized]


def ids_to_text(ids: Iterable[int], vocab: dict[str, int] | None = None) -> str:
    vocab = vocab or CHAR_EN_VOCAB
    inv = {idx: token for token, idx in vocab.items()}
    chars = []
    for raw in ids:
        token = inv.get(int(raw), "")
        if token in {BLANK_TOKEN, PAD_TOKEN, UNK_TOKEN}:
            continue
        chars.append(token)
    return " ".join("".join(chars).split())


def greedy_decode(ids: Iterable[int], vocab: dict[str, int] | None = None, blank_id: int = 0) -> str:
    collapsed = []
    prev = None
    for raw in ids:
        idx = int(raw)
        if idx != blank_id and idx != prev:
            collapsed.append(idx)
        prev = idx
    return ids_to_text(collapsed, vocab)


def greedy_decode_with_confidence(
    ids: Iterable[int],
    probs: Iterable[float],
    vocab: dict[str, int] | None = None,
    blank_id: int = 0,
) -> tuple[str, float]:
    collapsed = []
    confs = []
    prev = None
    for raw, prob in zip(ids, probs):
        idx = int(raw)
        if idx != blank_id and idx != prev:
            collapsed.append(idx)
            confs.append(float(prob))
        prev = idx
    confidence = sum(confs) / max(1, len(confs))
    return ids_to_text(collapsed, vocab), float(confidence)


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
    ref = list(normalize_english(reference))
    hyp = list(normalize_english(hypothesis))
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_english(reference).split()
    hyp = normalize_english(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))
