from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from l2s_itw.data.manifest import read_manifest
from l2s_itw.text import CharTokenizer


class VisualTTSDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CharTokenizer,
        config: dict[str, Any],
        limit: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.rows = read_manifest(self.manifest_path, limit=limit)
        self.tokenizer = tokenizer
        self.data_config = config["data"]
        self.model_config = config["model"]

        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        visual = self._load_tensor(row[self.data_config["visual_feature_key"]], "visual_features")
        mel = self._load_tensor(row[self.data_config["mel_key"]], "mel")

        if visual.ndim != 2:
            raise ValueError(f"visual_features must be [T, D], got {tuple(visual.shape)} for {row.get('id')}")
        if mel.ndim != 2:
            raise ValueError(f"mel must be [T, n_mels], got {tuple(mel.shape)} for {row.get('id')}")

        visual = visual.float()
        mel = mel.float()
        if mel.shape[0] == self.model_config["n_mels"] and mel.shape[1] != self.model_config["n_mels"]:
            mel = mel.transpose(0, 1).contiguous()

        speaker_key = self.data_config["speaker_embedding_key"]
        if speaker_key in row and row[speaker_key]:
            speaker = self._load_tensor(row[speaker_key], "speaker_embedding").float()
        else:
            speaker = torch.zeros(self.model_config["speaker_dim"], dtype=torch.float32)
        speaker = speaker.flatten()

        text_key = self.data_config["text_key"]
        text = str(row.get(text_key, ""))
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        return {
            "id": row.get("id", str(index)),
            "visual": visual,
            "mel": mel,
            "speaker": speaker,
            "tokens": tokens,
            "text": text,
        }

    def _load_tensor(self, path_value: str, dict_key: str) -> torch.Tensor:
        path = Path(path_value)
        if not path.is_absolute():
            path = self.base_dir / path
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if dict_key in obj:
                obj = obj[dict_key]
            elif "tensor" in obj:
                obj = obj["tensor"]
            else:
                available = ", ".join(obj.keys())
                raise KeyError(f"{path} does not contain '{dict_key}' or 'tensor'. Keys: {available}")
        if not torch.is_tensor(obj):
            raise TypeError(f"Expected tensor in {path}, got {type(obj)!r}")
        return obj


def collate_visual_tts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    visuals = [item["visual"] for item in batch]
    mels = [item["mel"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    speakers = [item["speaker"] for item in batch]

    return {
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "visuals": pad_sequence(visuals, batch_first=True),
        "visual_lengths": torch.tensor([x.shape[0] for x in visuals], dtype=torch.long),
        "mels": pad_sequence(mels, batch_first=True),
        "mel_lengths": torch.tensor([x.shape[0] for x in mels], dtype=torch.long),
        "tokens": pad_sequence(tokens, batch_first=True, padding_value=0),
        "text_lengths": torch.tensor([x.shape[0] for x in tokens], dtype=torch.long),
        "speakers": torch.stack(speakers),
    }
