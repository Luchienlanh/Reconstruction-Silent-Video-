from __future__ import annotations

from l2t_arch.text import metric_text


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (0 if ref_item == hyp_item else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def wer_cer(ref_text: str, hyp_text: str) -> dict[str, float | int]:
    ref_norm = metric_text(ref_text)
    hyp_norm = metric_text(hyp_text)
    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    word_edits = edit_distance(ref_words, hyp_words)
    char_edits = edit_distance(ref_chars, hyp_chars)
    return {
        "word_edits": word_edits,
        "words": max(len(ref_words), 1),
        "char_edits": char_edits,
        "chars": max(len(ref_chars), 1),
        "wer": word_edits / max(len(ref_words), 1),
        "cer": char_edits / max(len(ref_chars), 1),
        "exact": int(ref_norm == hyp_norm),
    }

