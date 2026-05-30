from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_cache(path: str | Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("format") != "r2inr_v1":
        raise ValueError(f"{path} is not an r2inr_v1 cache file.")
    return item


def split_cache_files(
    data_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    limit_files: int | None = None,
) -> tuple[list[Path], list[Path]]:
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit_files is not None:
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


def window_starts(video_len: int, window_frames: int, hop_frames: int) -> list[int]:
    video_len = int(video_len)
    window_frames = max(1, int(window_frames))
    hop_frames = max(1, int(hop_frames))
    if video_len <= window_frames:
        return [0]
    starts = list(range(0, video_len - window_frames + 1, hop_frames))
    last = video_len - window_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def _infer_fps(item: dict[str, Any]) -> float:
    fps = float(item.get("fps") or 0.0)
    if fps > 0:
        return fps
    video_times = item.get("video_times")
    if torch.is_tensor(video_times) and video_times.numel() > 1:
        diffs = video_times[1:] - video_times[:-1]
        valid = diffs[torch.isfinite(diffs) & (diffs > 0)]
        if valid.numel():
            return float(1.0 / valid.median().item())
    return 25.0


def mel_indices_for_video_window(
    item: dict[str, Any],
    start: int,
    end: int,
) -> torch.Tensor:
    mel_times = item["mel_times"].float()
    video_times = item["video_times"].float()
    fps = _infer_fps(item)
    t0 = float(video_times[min(start, video_times.numel() - 1)].item()) if video_times.numel() else 0.0
    duration = max(1, end - start) / max(fps, 1e-6)
    t1 = t0 + duration
    mask = (mel_times >= t0) & (mel_times < t1)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() > 0:
        return idx

    sample_rate = float(item.get("sample_rate") or 16000)
    hop_length = float(item.get("hop_length") or 256)
    approx_len = max(1, int(round(duration * sample_rate / max(hop_length, 1.0))))
    mel_start = int(round(t0 * sample_rate / max(hop_length, 1.0)))
    mel_start = max(0, min(mel_start, max(0, mel_times.numel() - 1)))
    mel_end = max(mel_start + 1, min(mel_start + approx_len, mel_times.numel()))
    return torch.arange(mel_start, mel_end, dtype=torch.long)


def extract_window(
    path: str | Path,
    item: dict[str, Any],
    start: int,
    window_frames: int,
    clip_index: int = 0,
) -> dict[str, Any]:
    video_len = int(item["video_len"])
    start = max(0, min(int(start), max(0, video_len - 1)))
    end = min(video_len, start + int(window_frames))
    mel_idx = mel_indices_for_video_window(item, start, end)
    if mel_idx.numel() <= 0:
        mel_idx = torch.arange(0, max(1, min(1, int(item["mel_len"]))), dtype=torch.long)

    video_times = item["video_times"][start:end].float()
    mel_times = item["mel_times"][mel_idx].float()
    t0 = float(video_times[0].item()) if video_times.numel() else 0.0

    crop_boxes = item.get("crop_boxes")
    if torch.is_tensor(crop_boxes):
        crop_boxes = crop_boxes[start:end]
    else:
        crop_boxes = torch.zeros(end - start, 4)

    return {
        "video": item["video"][:, start:end].float(),
        "landmarks": item["landmarks"][start:end].float(),
        "mel": item["mel"][mel_idx].float(),
        "video_times": video_times - t0,
        "mel_times": mel_times - t0,
        "mouth_valid_mask": item["mouth_valid_mask"][start:end].bool(),
        "crop_boxes": crop_boxes,
        "video_len": int(end - start),
        "mel_len": int(mel_idx.numel()),
        "path": str(path),
        "source_video": item.get("source_video", ""),
        "window_start": int(start),
        "window_end": int(end),
        "mel_indices": mel_idx.long(),
        "clip_index": int(clip_index),
    }


class WindowedMelDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        files: list[str | Path] | None = None,
        window_frames: int = 30,
        hop_frames: int = 10,
        max_windows_per_file: int = 0,
        random_windows_per_file: int = 0,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        resolved = []
        for f in files:
            p = Path(f)
            resolved.append(p if p.is_absolute() or p.exists() else self.data_dir / p)
        if not resolved:
            raise RuntimeError(f"No .pt cache files found under {self.data_dir}")
        self.files = resolved
        self.window_frames = int(window_frames)
        self.hop_frames = int(hop_frames)
        self.seed = int(seed)
        self.max_windows_per_file = int(max_windows_per_file)
        self.random_windows_per_file = int(random_windows_per_file)
        self.index: list[tuple[int, int]] = []
        self.file_starts: list[list[int]] = []
        self._build_index()

    def _build_index(self) -> None:
        rng = random.Random(self.seed)
        for file_idx, path in enumerate(self.files):
            item = load_cache(path)
            starts = window_starts(int(item["video_len"]), self.window_frames, self.hop_frames)
            self.file_starts.append([int(s) for s in starts])
            limit = self.random_windows_per_file or self.max_windows_per_file
            if limit and len(starts) > limit:
                rng.shuffle(starts)
                starts = sorted(starts[: int(limit)])
            self.index.extend((file_idx, int(s)) for s in starts)
        if not self.index:
            raise RuntimeError("No usable windows were created.")

    def resample_windows(self, epoch: int = 0) -> None:
        if self.random_windows_per_file <= 0:
            return
        rng = random.Random(self.seed + int(epoch) * 1000003)
        index: list[tuple[int, int]] = []
        for file_idx, starts in enumerate(self.file_starts):
            chosen = list(starts)
            if len(chosen) > self.random_windows_per_file:
                rng.shuffle(chosen)
                chosen = sorted(chosen[: self.random_windows_per_file])
            index.extend((file_idx, int(s)) for s in chosen)
        if not index:
            raise RuntimeError("No usable windows after resampling.")
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, start = self.index[idx]
        path = self.files[file_idx]
        item = load_cache(path)
        return extract_window(path, item, start, self.window_frames, clip_index=file_idx)


def _pad_video(video: torch.Tensor, length: int) -> torch.Tensor:
    if video.shape[1] == length:
        return video
    return F.pad(video, (0, 0, 0, 0, 0, length - video.shape[1]))


def _pad_2d(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    return F.pad(x, (0, 0, 0, length - x.shape[0]))


def _pad_landmarks(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    return F.pad(x, (0, 0, 0, 0, 0, length - x.shape[0]))


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    return F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_bool(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[0] == length:
        return x
    return F.pad(x, (0, length - x.shape[0]), value=False)


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    video_lengths = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
    mel_lengths = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    max_video = int(video_lengths.max().item())
    max_mel = int(mel_lengths.max().item())

    video = torch.stack([_pad_video(b["video"], max_video) for b in batch], dim=0)
    landmarks = torch.stack([_pad_landmarks(b["landmarks"], max_video) for b in batch], dim=0)
    mel = torch.stack([_pad_2d(b["mel"], max_mel) for b in batch], dim=0)
    video_times = torch.stack([_pad_1d(b["video_times"], max_video) for b in batch], dim=0)
    mel_times = torch.stack([_pad_1d(b["mel_times"], max_mel) for b in batch], dim=0)
    mouth_valid = torch.stack([_pad_bool(b["mouth_valid_mask"], max_video) for b in batch], dim=0)

    video_mask = torch.arange(max_video).unsqueeze(0) < video_lengths.unsqueeze(1)
    mel_mask = torch.arange(max_mel).unsqueeze(0) < mel_lengths.unsqueeze(1)
    return {
        "video": video,
        "landmarks": landmarks,
        "mel": mel,
        "video_times": video_times,
        "mel_times": mel_times,
        "mouth_valid_mask": mouth_valid,
        "video_mask": video_mask,
        "mel_mask": mel_mask,
        "video_lengths": video_lengths,
        "mel_lengths": mel_lengths,
        "paths": [b["path"] for b in batch],
        "source_videos": [b["source_video"] for b in batch],
        "window_starts": torch.tensor([b["window_start"] for b in batch], dtype=torch.long),
        "window_ends": torch.tensor([b["window_end"] for b in batch], dtype=torch.long),
        "clip_indices": torch.tensor([b["clip_index"] for b in batch], dtype=torch.long),
    }
