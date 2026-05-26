from __future__ import annotations

from collections import OrderedDict
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


def _resolve_files(data_dir: str | Path, files: list[str | Path] | None = None) -> list[Path]:
    root = Path(data_dir)
    if files is None:
        resolved = sorted(root.glob("*.pt"))
    else:
        resolved = []
        for f in files:
            p = Path(f)
            resolved.append(p if p.is_absolute() or p.exists() else root / p)
    if not resolved:
        raise RuntimeError(f"No .pt cache files found under {root}")
    return list(resolved)


class R2INRDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        files: list[str | Path] | None = None,
        max_frames: int = 0,
        random_crop: bool = False,
        seed: int = 42,
        limit: int | None = None,
        audio_target_shift_frames: int = 0,
    ):
        self.data_dir = Path(data_dir)
        files = _resolve_files(self.data_dir, files)
        if limit is not None:
            files = list(files)[: max(1, min(int(limit), len(files)))]
        self.files = list(files)
        self.max_frames = int(max_frames)
        self.random_crop = bool(random_crop)
        self.rng = random.Random(seed)
        self.audio_target_shift_frames = int(audio_target_shift_frames)

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
        mel_idx = torch.nonzero((mel_times >= t0) & (mel_times <= t1), as_tuple=False).flatten()
        if mel_idx.numel() <= 0:
            approx = round(self.max_frames * item["sample_rate"] / max(1.0, item["fps"] * item["hop_length"]))
            mel_idx = torch.arange(mel_times.numel()) < max(1, approx)
            mel_idx = torch.nonzero(mel_idx, as_tuple=False).flatten()
        mel_idx, target_idx = _target_mel_indices(mel_idx, int(item["mel_len"]), self.audio_target_shift_frames)

        item = dict(item)
        item["video"] = item["video"][:, start:end]
        item["landmarks"] = item["landmarks"][start:end]
        item["video_times"] = item["video_times"][start:end]
        item["mouth_valid_mask"] = item["mouth_valid_mask"][start:end]
        item["crop_boxes"] = item["crop_boxes"][start:end]
        item["mel"] = item["mel"][target_idx]
        item["mel_times"] = item["mel_times"][mel_idx]
        item["video_len"] = int(item["video"].shape[1])
        item["mel_len"] = int(item["mel"].shape[0])
        item["_audio_target_shift_applied"] = True
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = self._crop(_load_cache(path))
        if self.audio_target_shift_frames and not item.get("_audio_target_shift_applied"):
            mel_idx = torch.arange(int(item["mel_len"]), dtype=torch.long)
            mel_idx, target_idx = _target_mel_indices(mel_idx, int(item["mel_len"]), self.audio_target_shift_frames)
            item = dict(item)
            item["mel"] = item["mel"][target_idx]
            item["mel_times"] = item["mel_times"][mel_idx]
            item["mel_len"] = int(item["mel"].shape[0])
        crop_boxes = item.get("crop_boxes")
        if torch.is_tensor(crop_boxes):
            crop_boxes = crop_boxes.float()
        return {
            "video": item["video"].float(),
            "landmarks": item["landmarks"].float(),
            "mel": item["mel"].float(),
            "video_times": item["video_times"].float(),
            "mel_times": item["mel_times"].float(),
            "mouth_valid_mask": item["mouth_valid_mask"].bool(),
            "mouth_motion": _mouth_motion_features(item["landmarks"].float(), crop_boxes),
            "video_len": int(item["video_len"]),
            "mel_len": int(item["mel_len"]),
            "path": str(path),
            "source_video": item.get("source_video", ""),
        }


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
    idx = torch.nonzero((mel_times >= t0) & (mel_times < t1), as_tuple=False).flatten()
    if idx.numel() > 0:
        return idx

    sample_rate = float(item.get("sample_rate") or 16000)
    hop_length = float(item.get("hop_length") or 256)
    approx_len = max(1, int(round(duration * sample_rate / max(hop_length, 1.0))))
    mel_start = int(round(t0 * sample_rate / max(hop_length, 1.0)))
    mel_start = max(0, min(mel_start, max(0, mel_times.numel() - 1)))
    mel_end = max(mel_start + 1, min(mel_start + approx_len, mel_times.numel()))
    return torch.arange(mel_start, mel_end, dtype=torch.long)


def _target_mel_indices(mel_idx: torch.Tensor, mel_len: int, audio_target_shift_frames: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    mel_idx = mel_idx.long()
    shift = int(audio_target_shift_frames)
    if mel_idx.numel() == 0 or shift == 0:
        return mel_idx, mel_idx
    target_idx = mel_idx + shift
    valid = (target_idx >= 0) & (target_idx < int(mel_len))
    if bool(valid.any()):
        return mel_idx[valid], target_idx[valid]
    return mel_idx, target_idx.clamp(0, max(0, int(mel_len) - 1))


def _smooth_xy(xy: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if xy.shape[0] < 3:
        return xy
    k = min(int(kernel_size), int(xy.shape[0]))
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return xy
    flat = xy.flatten(1).transpose(0, 1).unsqueeze(0)
    flat = F.pad(flat, (k // 2, k // 2), mode="replicate")
    flat = F.avg_pool1d(flat, kernel_size=k, stride=1)
    return flat.squeeze(0).transpose(0, 1).reshape_as(xy)


def _derivatives(xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d1 = torch.zeros_like(xy)
    d2 = torch.zeros_like(xy)
    if xy.shape[0] > 1:
        d1[1:] = xy[1:] - xy[:-1]
    if xy.shape[0] > 2:
        d2[1:] = d1[1:] - d1[:-1]
    return d1, d2


def _box_motion_features(crop_boxes: torch.Tensor | None, width_height: torch.Tensor) -> torch.Tensor:
    t = int(width_height.shape[0])
    if crop_boxes is None or not torch.is_tensor(crop_boxes) or crop_boxes.numel() == 0:
        return width_height.new_zeros(t, 9)

    boxes = torch.nan_to_num(crop_boxes.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if boxes.shape[0] != t:
        boxes = boxes[:t] if boxes.shape[0] > t else F.pad(boxes, (0, 0, 0, t - boxes.shape[0]))
    if boxes.shape[-1] < 3:
        return width_height.new_zeros(t, 9)

    side = boxes[:, 2:3].clamp_min(1e-4)
    center = boxes[:, :2] + side * 0.5
    side_ref = side.median().clamp_min(1.0)
    center_ref = center.mean(dim=0, keepdim=True)
    center_rel = (center - center_ref) / side_ref
    side_rel = side / side_ref
    side_abs = torch.log1p(side) / 5.0
    raw_width = width_height[:, :1] * side_rel
    raw_height = width_height[:, 1:2] * side_rel
    raw_area = raw_width * raw_height
    raw_open_ratio = raw_height / raw_width.clamp_min(1e-4)

    center_d1, _ = _derivatives(center_rel.unsqueeze(1))
    side_d1, _ = _derivatives(side_rel.unsqueeze(1))
    center_speed = center_d1.squeeze(1).norm(dim=-1, keepdim=True)
    side_speed = side_d1.squeeze(1).abs()
    return torch.cat(
        [raw_width, raw_height, raw_area, raw_open_ratio, center_rel, side_rel, side_abs, center_speed, side_speed],
        dim=-1,
    )


def _mouth_motion_features(landmarks: torch.Tensor, crop_boxes: torch.Tensor | None = None) -> torch.Tensor:
    xy = torch.nan_to_num(landmarks[..., :2].float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_xy = xy.amin(dim=1)
    max_xy = xy.amax(dim=1)
    width_height = max_xy - min_xy
    center = xy.mean(dim=1)
    smooth_xy = _smooth_xy(xy)
    d1, d2 = _derivatives(smooth_xy)
    speed = d1.norm(dim=-1).mean(dim=1, keepdim=True)
    accel = d2.norm(dim=-1).mean(dim=1, keepdim=True)
    mouth_open = xy[..., 1].std(dim=1, unbiased=False, keepdim=True)
    area = width_height[:, :1] * width_height[:, 1:2]
    open_ratio = width_height[:, 1:2] / width_height[:, :1].clamp_min(1e-4)
    box_features = _box_motion_features(crop_boxes, width_height)
    return torch.cat([center, width_height, mouth_open, area, open_ratio, speed, accel, box_features], dim=-1)


def extract_window(path: str | Path, item: dict[str, Any], start: int, window_frames: int, clip_index: int = 0) -> dict[str, Any]:
    video_len = int(item["video_len"])
    start = max(0, min(int(start), max(0, video_len - 1)))
    end = min(video_len, start + int(window_frames))
    mel_idx = mel_indices_for_video_window(item, start, end)
    if mel_idx.numel() <= 0:
        mel_idx = torch.arange(0, max(1, min(1, int(item["mel_len"]))), dtype=torch.long)
    mel_idx, target_mel_idx = _target_mel_indices(
        mel_idx,
        int(item["mel_len"]),
        int(item.get("audio_target_shift_frames", 0)),
    )

    video_times = item["video_times"][start:end].float()
    mel_times = item["mel_times"][mel_idx].float()
    t0 = float(video_times[0].item()) if video_times.numel() else 0.0
    landmarks = item["landmarks"][start:end].float()
    crop_boxes = item.get("crop_boxes")
    if torch.is_tensor(crop_boxes):
        crop_boxes = crop_boxes[start:end].float()

    return {
        "video": item["video"][:, start:end].float(),
        "landmarks": landmarks,
        "mel": item["mel"][target_mel_idx].float(),
        "video_times": video_times - t0,
        "mel_times": mel_times - t0,
        "mouth_valid_mask": item["mouth_valid_mask"][start:end].bool(),
        "mouth_motion": _mouth_motion_features(landmarks, crop_boxes),
        "video_len": int(end - start),
        "mel_len": int(mel_idx.numel()),
        "path": str(path),
        "source_video": item.get("source_video", ""),
        "window_start": int(start),
        "window_end": int(end),
        "mel_indices": mel_idx.long(),
        "target_mel_indices": target_mel_idx.long(),
        "clip_index": int(clip_index),
    }


class WindowedR2INRDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        files: list[str | Path] | None = None,
        window_frames: int = 30,
        hop_frames: int = 10,
        max_windows_per_file: int = 0,
        cache_size: int = 2,
        seed: int = 42,
        limit: int | None = None,
        audio_target_shift_frames: int = 0,
    ):
        self.data_dir = Path(data_dir)
        resolved = _resolve_files(self.data_dir, files)
        if limit is not None:
            resolved = resolved[: max(1, min(int(limit), len(resolved)))]
        self.files = resolved
        self.window_frames = int(window_frames)
        self.hop_frames = int(hop_frames)
        self.cache_size = max(0, int(cache_size))
        self.audio_target_shift_frames = int(audio_target_shift_frames)
        self._cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self.seed = int(seed)
        self.index: list[tuple[int, int]] = []
        self._build_index(max_windows_per_file=max_windows_per_file)

    def _get_item(self, path: Path) -> dict[str, Any]:
        if self.cache_size <= 0:
            return _load_cache(path)
        item = self._cache.get(path)
        if item is not None:
            self._cache.move_to_end(path)
            return item
        item = _load_cache(path)
        self._cache[path] = item
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item

    def _build_index(self, max_windows_per_file: int = 0) -> None:
        rng = random.Random(self.seed)
        for file_idx, path in enumerate(self.files):
            item = self._get_item(path)
            starts = window_starts(int(item["video_len"]), self.window_frames, self.hop_frames)
            if max_windows_per_file and len(starts) > max_windows_per_file:
                rng.shuffle(starts)
                starts = sorted(starts[: int(max_windows_per_file)])
            self.index.extend((file_idx, int(start)) for start in starts)
        if not self.index:
            raise RuntimeError("No usable training windows were created.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, start = self.index[idx]
        path = self.files[file_idx]
        item = self._get_item(path)
        if self.audio_target_shift_frames:
            item = dict(item)
            item["audio_target_shift_frames"] = self.audio_target_shift_frames
        return extract_window(path, item, start, self.window_frames, clip_index=file_idx)


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
    mouth_motion = None
    if "mouth_motion" in batch[0]:
        mouth_motion = torch.stack([_pad_2d(b["mouth_motion"], t_video) for b in batch], dim=0)

    out = {
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
    if mouth_motion is not None:
        out["mouth_motion"] = mouth_motion
        out["window_starts"] = torch.tensor([b.get("window_start", 0) for b in batch], dtype=torch.long)
        out["window_ends"] = torch.tensor([b.get("window_end", b["video_len"]) for b in batch], dtype=torch.long)
        out["clip_indices"] = torch.tensor([b.get("clip_index", 0) for b in batch], dtype=torch.long)
    return out
