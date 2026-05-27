from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .text import build_vocab, normalize_text, normalize_text_nodiac, text_to_ids


def load_text_cache(path: str | Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("format") not in {"r2inr_text_v1", "r2inr_v1"}:
        raise ValueError(f"{path} is not an r2inr text cache file.")
    if "landmarks" not in item:
        raise ValueError(f"{path} has no landmarks.")
    return item


def split_cache_files(
    data_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    limit_files: int | None = None,
) -> tuple[list[Path], list[Path]]:
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit_files is not None and limit_files > 0:
        files = files[: max(1, min(int(limit_files), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt files found under {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 and val_ratio > 0 else 0
    val_files = sorted(files[:val_count])
    train_files = sorted(files[val_count:])
    if not train_files:
        raise RuntimeError("No training files after split.")
    return train_files, val_files


def build_vocab_from_files(files: list[Path], text_unit: str = "syllable_nodiac", min_freq: int = 1) -> dict[str, int]:
    texts = []
    for path in files:
        item = load_text_cache(path)
        texts.append(str(item.get("transcript_text", "")))
    return build_vocab(texts, min_freq=min_freq, text_unit=text_unit)


class LandmarkCTCDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        vocab: dict[str, int],
        files: list[str | Path] | None = None,
        frame_stride: int = 1,
        text_unit: str = "syllable_nodiac",
        min_input_target_ratio: float = 1.05,
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        resolved = [Path(f) if Path(f).is_absolute() or Path(f).exists() else self.data_dir / Path(f) for f in files]
        self.vocab = vocab
        self.frame_stride = max(1, int(frame_stride))
        self.text_unit = str(text_unit)
        self.files: list[Path] = []
        self.skipped: list[tuple[str, int, int]] = []
        for path in resolved:
            item = load_text_cache(path)
            ids = text_to_ids(str(item.get("transcript_text", "")), vocab, text_unit=text_unit)
            input_len = max(1, (int(item["landmarks"].shape[0]) + self.frame_stride - 1) // self.frame_stride)
            if input_len >= max(1, int(round(len(ids) * min_input_target_ratio))):
                self.files.append(path)
            else:
                self.skipped.append((str(path), input_len, len(ids)))
        if not self.files:
            raise RuntimeError("No CTC-usable landmark files after input/target length filtering.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = load_text_cache(path)
        stride = self.frame_stride
        landmarks = item["landmarks"][::stride].float()
        if landmarks.shape[-1] < 6:
            landmarks = self._add_derivatives(landmarks[..., :2])
        video_times = item.get("video_times", torch.arange(landmarks.shape[0]).float())[::stride].float()
        mouth_valid = item.get("mouth_valid_mask", torch.ones(landmarks.shape[0], dtype=torch.bool))[::stride].bool()
        raw_text = str(item.get("transcript_text", ""))
        norm_text = normalize_text_nodiac(raw_text) if self.text_unit.endswith("_nodiac") else normalize_text(raw_text)
        ids = torch.tensor(text_to_ids(raw_text, self.vocab, text_unit=self.text_unit), dtype=torch.long)
        return {
            "landmarks": landmarks,
            "video_times": video_times,
            "mouth_valid_mask": mouth_valid,
            "landmark_len": int(landmarks.shape[0]),
            "target_ids": ids,
            "target_len": int(ids.numel()),
            "transcript_text": norm_text,
            "path": str(path),
            "source_video": item.get("source_video", ""),
        }

    @staticmethod
    def _add_derivatives(xy: torch.Tensor) -> torch.Tensor:
        d1 = torch.zeros_like(xy)
        d2 = torch.zeros_like(xy)
        if xy.shape[0] > 1:
            d1[1:] = xy[1:] - xy[:-1]
        if xy.shape[0] > 2:
            d2[1:] = d1[1:] - d1[:-1]
        return torch.cat([xy, d1, d2], dim=-1)


def _pad_landmarks(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, 0, 0, 0, 0, length - x.shape[0]))


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_bool(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=False)


def _pad_tokens(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=0)


def collate_landmark_ctc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([b["landmark_len"] for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    max_target = int(target_lengths.max().item())
    landmarks = torch.stack([_pad_landmarks(b["landmarks"], max_len) for b in batch], dim=0)
    video_times = torch.stack([_pad_1d(b["video_times"], max_len) for b in batch], dim=0)
    mouth_valid = torch.stack([_pad_bool(b["mouth_valid_mask"], max_len) for b in batch], dim=0)
    targets = torch.stack([_pad_tokens(b["target_ids"], max_target) for b in batch], dim=0)
    landmark_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "landmarks": landmarks,
        "video_times": video_times,
        "mouth_valid_mask": mouth_valid,
        "landmark_mask": landmark_mask,
        "landmark_lengths": lengths,
        "target_ids": targets,
        "target_lengths": target_lengths,
        "transcript_texts": [b["transcript_text"] for b in batch],
        "paths": [b["path"] for b in batch],
        "source_videos": [b.get("source_video", "") for b in batch],
    }

