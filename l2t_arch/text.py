from __future__ import annotations

import re


_APOSTROPHE_RE = re.compile(r"\s*'\s*")
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s']")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, keep_apostrophe: bool = True) -> str:
    text = text.lower().replace("’", "'").replace("`", "'")
    text = _APOSTROPHE_RE.sub("'" if keep_apostrophe else "", text)
    text = _NON_TEXT_RE.sub(" ", text)
    if not keep_apostrophe:
        text = text.replace("'", "")
    return _SPACE_RE.sub(" ", text).strip()


def metric_text(text: str) -> str:
    return normalize_text(text, keep_apostrophe=False)


class CharTokenizer:
    def __init__(self, symbols: str = "abcdefghijklmnopqrstuvwxyz '") -> None:
        self.blank_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.symbols = symbols
        self.id_to_token = ["<blank>", "<bos>", "<eos>", "<unk>"] + list(symbols)
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        normalized = normalize_text(text)
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in normalized]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx in {self.blank_id, self.bos_id}:
                continue
            if idx == self.eos_id:
                break
            if 0 <= idx < len(self.id_to_token):
                token = self.id_to_token[idx]
                if not token.startswith("<"):
                    chars.append(token)
        return _SPACE_RE.sub(" ", "".join(chars)).strip()

    def decode_ctc(self, ids: list[int]) -> str:
        collapsed = []
        prev = None
        for idx in ids:
            if idx == prev:
                continue
            prev = idx
            if idx == self.blank_id:
                continue
            collapsed.append(idx)
        return self.decode(collapsed)

