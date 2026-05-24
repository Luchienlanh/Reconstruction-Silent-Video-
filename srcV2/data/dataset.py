from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _load_cache(path: str | Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") != "r2inr_v1":
        raise ValueError(f"{path} is not an r2inr_v1 cache file.")
    return data


class R2INRDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        files: list[str | Path] | None = None,
        max_frames: int = 0,
        random_crop: bool = False,
        seed: int = 42,
        limit: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        else:
            resolved = []
            for f in files:
                p = Path(f)
                if p.is_absolute() or p.exists():
                    resolved.append(p)
                else:
                    resolved.append(self.data_dir / p)
            files = resolved
        if limit is not None:
            files = list(files)[: max(1, min(int(limit), len(files)))]
        if not files:
            raise RuntimeError(f"No .pt cache files found under {self.data_dir}")
        self.files = list(files)
        self.max_frames = int(max_frames)
        self.random_crop = bool(random_crop)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files)

    def _crop(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.max_frames <= 0 or int(item["video_len"]) <= self.max_frames:
            return item
        v_len = int(item["video_len"])
        if self.random_crop:
            start = self.rng.randint(0, v_len - self.max_frames)
        else:
            start = 0
        end = start + self.max_frames
        t0 = float(item["video_times"][start])
        t1 = float(item["video_times"][end - 1])
        mel_times = item["mel_times"]
        mel_mask = (mel_times >= t0) & (mel_times <= t1)
        if not bool(mel_mask.any()):
            approx = round(self.max_frames * item["sample_rate"] / max(1.0, item["fps"] * item["hop_length"]))
            mel_mask = torch.arange(mel_times.numel()) < max(1, approx)

        item = dict(item)
        item["video"] = item["video"][:, start:end]
        item["landmarks"] = item["landmarks"][start:end]
        item["video_times"] = item["video_times"][start:end]
        item["mouth_valid_mask"] = item["mouth_valid_mask"][start:end]
        item["crop_boxes"] = item["crop_boxes"][start:end]
        item["mel"] = item["mel"][mel_mask]
        item["mel_times"] = item["mel_times"][mel_mask]
        item["video_len"] = int(item["video"].shape[1])
        item["mel_len"] = int(item["mel"].shape[0])
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = self._crop(_load_cache(path))
        return {
            "video": item["video"].float(),
            "landmarks": item["landmarks"].float(),
            "mel": item["mel"].float(),
            "video_times": item["video_times"].float(),
            "mel_times": item["mel_times"].float(),
            "mouth_valid_mask": item["mouth_valid_mask"].bool(),
            "video_len": int(item["video_len"]),
            "mel_len": int(item["mel_len"]),
            "path": str(path),
            "source_video": item.get("source_video", ""),
        }


def _pad_video(video: torch.Tensor, length: int) -> torch.Tensor:
    if video.shape[1] == length:
        return video
    return F.pad(video, (0, 0, 0, 0, 0, length - video.shape[1]))


def _pad_2d(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    return F.pad(x, (0, 0, 0, length - x.shape[0]))


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    return F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_bool(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    return F.pad(x, (0, length - x.shape[0]), value=False)


def collate_r2inr(batch: list[dict[str, Any]]) -> dict[str, Any]:
    v_lens = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
    m_lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    t_video = int(v_lens.max().item())
    t_mel = int(m_lens.max().item())
    paths = [b["path"] for b in batch]
    sources = [b["source_video"] for b in batch]

    video = torch.stack([_pad_video(b["video"], t_video) for b in batch], dim=0)
    landmarks = torch.stack([_pad_2d(b["landmarks"].flatten(1), t_video).view(t_video, b["landmarks"].shape[1], b["landmarks"].shape[2]) for b in batch], dim=0)
    mel = torch.stack([_pad_2d(b["mel"], t_mel) for b in batch], dim=0)
    video_times = torch.stack([_pad_1d(b["video_times"], t_video) for b in batch], dim=0)
    mel_times = torch.stack([_pad_1d(b["mel_times"], t_mel) for b in batch], dim=0)
    mouth_valid = torch.stack([_pad_bool(b["mouth_valid_mask"], t_video) for b in batch], dim=0)
    video_mask = torch.arange(t_video).unsqueeze(0) < v_lens.unsqueeze(1)
    mel_mask = torch.arange(t_mel).unsqueeze(0) < m_lens.unsqueeze(1)

    return {
        "video": video,
        "landmarks": landmarks,
        "mel": mel,
        "video_times": video_times,
        "mel_times": mel_times,
        "mouth_valid_mask": mouth_valid,
        "video_mask": video_mask,
        "mel_mask": mel_mask,
        "video_lengths": v_lens,
        "mel_lengths": m_lens,
        "paths": paths,
        "source_videos": sources,
    }
