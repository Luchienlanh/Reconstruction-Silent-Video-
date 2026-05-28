from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .landmark_dataset import load_text_cache, split_cache_files


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


def mel_indices_for_video_window(item: dict[str, Any], start: int, end: int) -> torch.Tensor:
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


def _add_derivatives(xy: torch.Tensor) -> torch.Tensor:
    d1 = torch.zeros_like(xy)
    d2 = torch.zeros_like(xy)
    if xy.shape[0] > 1:
        d1[1:] = xy[1:] - xy[:-1]
    if xy.shape[0] > 2:
        d2[1:] = d1[1:] - d1[:-1]
    return torch.cat([xy, d1, d2], dim=-1)


def _smooth_mel(mel: torch.Tensor, frames: int) -> torch.Tensor:
    frames = int(frames)
    if frames <= 1 or mel.shape[0] <= 2:
        return mel
    if frames % 2 == 0:
        frames += 1
    x = mel.transpose(0, 1).unsqueeze(0)
    x = F.avg_pool1d(x, kernel_size=frames, stride=1, padding=frames // 2)
    return x.squeeze(0).transpose(0, 1).contiguous()


class LandmarkEnvelopeDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        files: list[str | Path] | None = None,
        window_frames: int = 45,
        hop_frames: int = 15,
        max_windows_per_file: int = 0,
        random_windows_per_file: int = 0,
        smooth_target_frames: int = 3,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        self.files = [Path(f) if Path(f).is_absolute() or Path(f).exists() else self.data_dir / Path(f) for f in files]
        self.window_frames = int(window_frames)
        self.hop_frames = int(hop_frames)
        self.max_windows_per_file = int(max_windows_per_file)
        self.random_windows_per_file = int(random_windows_per_file)
        self.smooth_target_frames = int(smooth_target_frames)
        self.seed = int(seed)
        self.index: list[tuple[int, int]] = []
        self.file_starts: list[list[int]] = []
        self._build_index()

    def _build_index(self) -> None:
        rng = random.Random(self.seed)
        for file_idx, path in enumerate(self.files):
            item = load_text_cache(path)
            if "mel" not in item:
                self.file_starts.append([])
                continue
            video_len = int(item.get("video_len") or item["landmarks"].shape[0])
            starts = window_starts(video_len, self.window_frames, self.hop_frames)
            self.file_starts.append([int(s) for s in starts])
            limit = self.random_windows_per_file or self.max_windows_per_file
            if limit and len(starts) > limit:
                rng.shuffle(starts)
                starts = sorted(starts[:limit])
            self.index.extend((file_idx, int(s)) for s in starts)
        if not self.index:
            raise RuntimeError("No usable vocoder windows were created.")

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
            raise RuntimeError("No usable vocoder windows after resampling.")
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, start = self.index[idx]
        path = self.files[file_idx]
        item = load_text_cache(path)
        video_len = int(item.get("video_len") or item["landmarks"].shape[0])
        end = min(video_len, start + self.window_frames)
        mel_idx = mel_indices_for_video_window(item, start, end)

        landmarks = item["landmarks"][start:end].float()
        if landmarks.shape[-1] < 6:
            landmarks = _add_derivatives(landmarks[..., :2])
        video_times = item.get("video_times", torch.arange(video_len).float())[start:end].float()
        mel_times = item["mel_times"][mel_idx].float()
        t0 = float(video_times[0].item()) if video_times.numel() else 0.0
        mel = item["mel"][mel_idx].float()
        target_mel = _smooth_mel(mel, self.smooth_target_frames)
        valid = item.get("mouth_valid_mask", torch.ones(video_len, dtype=torch.bool))[start:end].bool()
        raw_video = item.get("video")
        if torch.is_tensor(raw_video):
            video = raw_video[:, start:end].float()
        else:
            video = torch.zeros(1, end - start, 96, 96, dtype=torch.float32)
        return {
            "video": video,
            "landmarks": landmarks,
            "video_times": video_times - t0,
            "mouth_valid_mask": valid,
            "mel": mel,
            "target_mel": target_mel,
            "mel_times": mel_times - t0,
            "landmark_len": int(landmarks.shape[0]),
            "mel_len": int(mel.shape[0]),
            "path": str(path),
            "source_video": item.get("source_video", ""),
            "sample_rate": int(item.get("sample_rate") or 16000),
            "hop_length": int(item.get("hop_length") or 256),
        }


def _pad_landmarks(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, 0, 0, 0, 0, length - x.shape[0]))


def _pad_video(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[1] == length else F.pad(x, (0, 0, 0, 0, 0, length - x.shape[1]))


def _pad_2d(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, 0, 0, length - x.shape[0]))


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_bool(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=False)


def collate_envelope(batch: list[dict[str, Any]]) -> dict[str, Any]:
    landmark_lengths = torch.tensor([b["landmark_len"] for b in batch], dtype=torch.long)
    mel_lengths = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    max_landmark = int(landmark_lengths.max().item())
    max_mel = int(mel_lengths.max().item())
    video = torch.stack([_pad_video(b["video"], max_landmark) for b in batch], dim=0)
    landmarks = torch.stack([_pad_landmarks(b["landmarks"], max_landmark) for b in batch], dim=0)
    video_times = torch.stack([_pad_1d(b["video_times"], max_landmark) for b in batch], dim=0)
    mouth_valid = torch.stack([_pad_bool(b["mouth_valid_mask"], max_landmark) for b in batch], dim=0)
    mel = torch.stack([_pad_2d(b["mel"], max_mel) for b in batch], dim=0)
    target_mel = torch.stack([_pad_2d(b["target_mel"], max_mel) for b in batch], dim=0)
    mel_times = torch.stack([_pad_1d(b["mel_times"], max_mel) for b in batch], dim=0)
    landmark_mask = torch.arange(max_landmark).unsqueeze(0) < landmark_lengths.unsqueeze(1)
    mel_mask = torch.arange(max_mel).unsqueeze(0) < mel_lengths.unsqueeze(1)
    return {
        "video": video,
        "landmarks": landmarks,
        "video_times": video_times,
        "mouth_valid_mask": mouth_valid,
        "landmark_mask": landmark_mask,
        "landmark_lengths": landmark_lengths,
        "mel": mel,
        "target_mel": target_mel,
        "mel_times": mel_times,
        "mel_mask": mel_mask,
        "mel_lengths": mel_lengths,
        "paths": [b["path"] for b in batch],
        "source_videos": [b.get("source_video", "") for b in batch],
        "sample_rates": torch.tensor([b["sample_rate"] for b in batch], dtype=torch.long),
        "hop_lengths": torch.tensor([b["hop_length"] for b in batch], dtype=torch.long),
    }


__all__ = [
    "LandmarkEnvelopeDataset",
    "collate_envelope",
    "split_cache_files",
]
