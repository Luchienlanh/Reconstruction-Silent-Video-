from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .text import CHAR_EN_VOCAB, normalize_english, text_to_ids


def torch_load_cpu(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    item = torch_load_cpu(path)
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
    if limit_files is not None and int(limit_files) > 0:
        files = files[: max(1, min(int(limit_files), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt feature files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = int(round(len(files) * max(0.0, min(0.9, float(val_ratio)))))
    if len(files) > 1 and val_ratio > 0:
        val_count = max(1, val_count)
    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])
    if not train_files:
        train_files = val_files
        val_files = []
    return train_files, val_files


class AVFeatureTextDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        files: list[str | Path] | None = None,
        vocab: dict[str, int] | None = None,
        min_input_target_ratio: float = 1.05,
        input_length_factor: int = 2,
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        resolved = []
        for file in files:
            path = Path(file)
            resolved.append(path if path.is_absolute() or path.exists() else self.data_dir / path)

        self.vocab = vocab or CHAR_EN_VOCAB
        self.min_input_target_ratio = float(min_input_target_ratio)
        self.input_length_factor = max(1, int(input_length_factor))
        self.files: list[Path] = []
        self.skipped: list[tuple[str, str]] = []
        for path in resolved:
            try:
                item = load_feature_cache(path)
                text = normalize_english(str(item.get("transcript_text", "")))
                ids = text_to_ids(text, self.vocab)
                feature_len = int(item.get("feature_len", item["features"].shape[0]))
                effective_input = feature_len * self.input_length_factor
                required = max(1, int(round(len(ids) * self.min_input_target_ratio)))
                if not text or not ids:
                    self.skipped.append((str(path), "empty_transcript"))
                    continue
                if effective_input < required:
                    self.skipped.append((str(path), f"ctc_too_short:{effective_input}<{required}"))
                    continue
                self.files.append(path)
            except Exception as exc:
                self.skipped.append((str(path), str(exc)))
        if not self.files:
            raise RuntimeError("No CTC-usable AV-HuBERT feature files.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = load_feature_cache(path)
        features = item["features"].float()
        transcript = normalize_english(str(item.get("transcript_text", "")))
        target_ids = torch.tensor(text_to_ids(transcript, self.vocab), dtype=torch.long)
        return {
            "features": features,
            "feature_len": int(features.shape[0]),
            "target_ids": target_ids,
            "target_len": int(target_ids.numel()),
            "transcript_text": transcript,
            "path": str(path),
            "source_video": item.get("source_video", ""),
            "source_cache": item.get("source_cache", ""),
        }


def _pad_2d(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, 0, 0, length - x.shape[0]))


def _pad_1d(x: torch.Tensor, length: int, value: int = 0) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=value)


def collate_feature_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    feature_lengths = torch.tensor([b["feature_len"] for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)
    max_feature = int(feature_lengths.max().item())
    max_target = int(target_lengths.max().item())
    features = torch.stack([_pad_2d(b["features"], max_feature) for b in batch], dim=0)
    targets = torch.stack([_pad_1d(b["target_ids"], max_target, value=0) for b in batch], dim=0)
    feature_mask = torch.arange(max_feature).unsqueeze(0) < feature_lengths.unsqueeze(1)
    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "feature_mask": feature_mask,
        "target_ids": targets,
        "target_lengths": target_lengths,
        "transcript_texts": [b["transcript_text"] for b in batch],
        "paths": [b["path"] for b in batch],
        "source_videos": [b.get("source_video", "") for b in batch],
        "source_caches": [b.get("source_cache", "") for b in batch],
    }
