from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .text import build_vocab, normalize_text, normalize_text_nodiac, text_to_ids


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("format") != "avhubert_feature_v1":
        raise ValueError(f"{path} is not an avhubert_feature_v1 file.")
    return item


def split_feature_files(
    data_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    limit_files: int | None = None,
) -> tuple[list[Path], list[Path]]:
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit_files is not None and limit_files > 0:
        files = files[: max(1, min(int(limit_files), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt feature files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 and val_ratio > 0 else 0
    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])
    if not train_files:
        raise RuntimeError("No training files after split.")
    return train_files, val_files


def build_vocab_from_feature_files(files: list[Path], text_unit: str = "syllable_nodiac", min_freq: int = 1) -> dict[str, int]:
    texts = []
    for path in files:
        item = load_feature_cache(path)
        texts.append(str(item.get("transcript_text", "")))
    return build_vocab(texts, min_freq=min_freq, text_unit=text_unit)


class AVFeatureCTCDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        vocab: dict[str, int],
        files: list[str | Path] | None = None,
        text_unit: str = "syllable_nodiac",
        min_input_target_ratio: float = 1.05,
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        resolved = [Path(f) if Path(f).is_absolute() or Path(f).exists() else self.data_dir / Path(f) for f in files]
        self.vocab = vocab
        self.text_unit = text_unit
        self.files = []
        self.skipped = []
        for path in resolved:
            item = load_feature_cache(path)
            ids = text_to_ids(str(item.get("transcript_text", "")), vocab, text_unit=text_unit)
            feature_len = int(item["feature_len"])
            if feature_len >= max(1, int(round(len(ids) * min_input_target_ratio))):
                self.files.append(path)
            else:
                self.skipped.append((str(path), feature_len, len(ids)))
        if not self.files:
            raise RuntimeError("No CTC-usable feature files after input/target length filtering.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = load_feature_cache(path)
        text = str(item.get("transcript_text", ""))
        norm_text = normalize_text_nodiac(text) if self.text_unit.endswith("_nodiac") else normalize_text(text)
        ids = torch.tensor(text_to_ids(text, self.vocab, text_unit=self.text_unit), dtype=torch.long)
        features = item["features"].float()
        return {
            "features": features,
            "feature_len": int(features.shape[0]),
            "target_ids": ids,
            "target_len": int(ids.numel()),
            "transcript_text": norm_text,
            "path": str(path),
            "source_video": item.get("source_video", ""),
        }


def _pad_2d(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, 0, 0, length - x.shape[0]))


def _pad_tokens(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=0)


def collate_feature_ctc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    feature_lengths = torch.tensor([b["feature_len"] for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)
    max_feat = int(feature_lengths.max().item())
    max_target = int(target_lengths.max().item())
    features = torch.stack([_pad_2d(b["features"], max_feat) for b in batch], dim=0)
    targets = torch.stack([_pad_tokens(b["target_ids"], max_target) for b in batch], dim=0)
    feature_mask = torch.arange(max_feat).unsqueeze(0) < feature_lengths.unsqueeze(1)
    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "feature_mask": feature_mask,
        "target_ids": targets,
        "target_lengths": target_lengths,
        "transcript_texts": [b["transcript_text"] for b in batch],
        "paths": [b["path"] for b in batch],
        "source_videos": [b.get("source_video", "") for b in batch],
    }

