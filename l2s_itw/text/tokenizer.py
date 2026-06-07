from __future__ import annotations


class CharTokenizer:
    """Small text-unit tokenizer.

    This is deliberately simple for the first pipeline pass. It can be replaced
    by a phoneme/G2P tokenizer without changing the model/data interfaces.
    """

    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, symbols: str, lowercase: bool = True) -> None:
        self.lowercase = lowercase
        unique_symbols = []
        for char in symbols:
            if char not in unique_symbols:
                unique_symbols.append(char)
        self.tokens = [self.pad_token, self.unk_token] + unique_symbols
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def encode(self, text: str) -> list[int]:
        text = self.normalize(text)
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        return ids or [self.unk_id]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if token in {self.pad_token, self.unk_token}:
                continue
            chars.append(token)
        return "".join(chars)
