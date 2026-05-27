from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_text_cache(path: str | Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("format") not in {"r2inr_text_v1", "r2inr_v1"}:
        raise ValueError(f"{path} is not an r2inr text cache file.")
    if not str(item.get("transcript_text", "")).strip():
        raise ValueError(f"{path} has no transcript_text.")
    return item


def split_cache_files(data_dir: str | Path, limit_files: int | None = None) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit_files is not None and limit_files > 0:
        files = files[: max(1, min(int(limit_files), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt files found under {data_dir}")
    return files


class VideoTextDataset(Dataset):
    def __init__(self, data_dir: str | Path, files: list[str | Path] | None = None, limit_files: int | None = None):
        self.data_dir = Path(data_dir)
        if files is None:
            files = split_cache_files(self.data_dir, limit_files=limit_files)
        resolved = [Path(f) if Path(f).is_absolute() or Path(f).exists() else self.data_dir / Path(f) for f in files]
        self.files = []
        self.skipped = []
        for path in resolved:
            try:
                load_text_cache(path)
                self.files.append(path)
            except Exception as exc:
                self.skipped.append((str(path), str(exc)))
        if not self.files:
            raise RuntimeError("No text cache files with transcript_text.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = load_text_cache(path)
        return {
            "video": item["video"].float(),
            "video_len": int(item["video_len"]),
            "video_times": item["video_times"].float(),
            "mouth_valid_mask": item["mouth_valid_mask"].bool(),
            "transcript_text": str(item.get("transcript_text", "")),
            "path": str(path),
            "source_video": item.get("source_video", ""),
        }


def _pad_video(video: torch.Tensor, length: int) -> torch.Tensor:
    return video if video.shape[1] == length else F.pad(video, (0, 0, 0, 0, 0, length - video.shape[1]))


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_bool(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=False)


def collate_video_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    video = torch.stack([_pad_video(b["video"], max_len) for b in batch], dim=0)
    video_times = torch.stack([_pad_1d(b["video_times"], max_len) for b in batch], dim=0)
    mouth_valid = torch.stack([_pad_bool(b["mouth_valid_mask"], max_len) for b in batch], dim=0)
    video_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "video": video,
        "video_times": video_times,
        "mouth_valid_mask": mouth_valid,
        "video_mask": video_mask,
        "video_lengths": lengths,
        "transcript_texts": [b["transcript_text"] for b in batch],
        "paths": [b["path"] for b in batch],
        "source_videos": [b.get("source_video", "") for b in batch],
    }


def split_train_val(files: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 and val_ratio > 0 else 0
    return sorted(files[val_count:]), sorted(files[:val_count])

