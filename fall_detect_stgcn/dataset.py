"""PyTorch dataset for extracted UP-Fall pose sequences."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import activity_to_label
from .pose_features import normalize_pose_sequence, resample_sequence


def list_pose_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.npz"))


def parse_subjects(value: str, minimum: int = 1, maximum: int = 17) -> set[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))

    invalid = [item for item in selected if item < minimum or item > maximum]
    if invalid:
        raise ValueError(f"Values out of range {minimum}-{maximum}: {invalid}")
    return selected


def read_meta(path: Path) -> dict[str, int | str]:
    with np.load(path, allow_pickle=False) as data:
        if {"subject", "activity", "trial", "camera"}.issubset(data.files):
            return {
                "subject": int(data["subject"]),
                "activity": int(data["activity"]),
                "trial": int(data["trial"]),
                "camera": int(data["camera"]),
                "path": str(path),
            }

    pattern = re.compile(r"Subject(\d+).*Activity(\d+).*Trial(\d+).*Camera(\d+)", re.IGNORECASE)
    match = pattern.search(str(path))
    if not match:
        raise ValueError(f"Cannot infer metadata from {path}")
    return {
        "subject": int(match.group(1)),
        "activity": int(match.group(2)),
        "trial": int(match.group(3)),
        "camera": int(match.group(4)),
        "path": str(path),
    }


class UPFallPoseDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        task: str,
        seq_len: int,
        augment: bool = False,
    ) -> None:
        self.files = list(files)
        self.task = task
        self.seq_len = seq_len
        self.augment = augment
        self.metadata = [read_meta(path) for path in self.files]
        self.labels = [activity_to_label(int(meta["activity"]), task) for meta in self.metadata]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            keypoints = data["keypoints"].astype(np.float32)
            mask = data["mask"].astype(np.float32) if "mask" in data.files else None

        features = normalize_pose_sequence(keypoints, mask)
        if self.augment and len(features) > self.seq_len:
            # Randomly keep 80-100% of a clip before fixed resampling.
            keep = np.random.uniform(0.8, 1.0)
            window = max(self.seq_len, int(round(len(features) * keep)))
            if window < len(features):
                start = np.random.randint(0, len(features) - window + 1)
                features = features[start : start + window]

        features = resample_sequence(features, self.seq_len)
        label = self.labels[index]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


def filter_pose_files(files: list[Path], subjects: set[int]) -> list[Path]:
    filtered = []
    for path in files:
        meta = read_meta(path)
        if int(meta["subject"]) in subjects:
            filtered.append(path)
    return filtered
