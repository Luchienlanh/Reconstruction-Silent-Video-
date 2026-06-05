from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _load_r2inr_cache(path: str | Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("format") != "r2inr_v1":
        raise ValueError(f"{path} is not an r2inr_v1 cache file.")
    return item


def _load_av_feature_cache(path: str | Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("format") != "avhubert_feature_v1":
        raise ValueError(f"{path} is not an avhubert_feature_v1 file.")
    return item


def safe_stem(path: str | Path) -> str:
    name = Path(str(path)).stem
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in name]
    out = "".join(keep).strip("._")
    return out[:140] or "sample"


def _source_key(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lower()


def _feature_keys(item: dict[str, Any], path: Path) -> set[str]:
    keys = {path.stem.lower(), safe_stem(path).lower()}
    for field in ("source_video", "source_cache"):
        value = item.get(field, "")
        if value:
            keys.add(_source_key(value))
            keys.add(safe_stem(value).lower())
    return {key for key in keys if key}


def _r2_keys(item: dict[str, Any], path: Path) -> set[str]:
    keys = {path.stem.lower(), safe_stem(path).lower()}
    value = item.get("source_video", "")
    if value:
        keys.add(_source_key(value))
        keys.add(safe_stem(value).lower())
    return {key for key in keys if key}


def build_feature_index(feature_dir: str | Path | None) -> tuple[dict[str, Path], int]:
    if not feature_dir:
        return {}, 0
    root = Path(feature_dir)
    if not root.exists():
        return {}, 0
    index: dict[str, Path] = {}
    feature_dim = 0
    for path in sorted(root.glob("*.pt")):
        try:
            item = _load_av_feature_cache(path)
        except Exception:
            continue
        if feature_dim <= 0:
            feature_dim = int(item.get("feature_dim", 0)) or int(item["features"].shape[-1])
        for key in _feature_keys(item, path):
            index.setdefault(key, path)
    return index, feature_dim


def infer_av_feature_dim(feature_dir: str | Path | None) -> int:
    _, feature_dim = build_feature_index(feature_dir)
    return int(feature_dim)


def split_cache_files(
    data_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    limit_files: int | None = None,
) -> tuple[list[Path], list[Path]]:
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit_files is not None and int(limit_files) > 0:
        files = files[: max(1, min(int(limit_files), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt cache files found under {data_dir}")
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


class V10R2INRDataset(Dataset):
    """R2INR cache dataset with optional cached AV-HuBERT feature conditioning."""

    def __init__(
        self,
        data_dir: str | Path,
        files: list[str | Path] | None = None,
        max_frames: int = 0,
        random_crop: bool = False,
        seed: int = 42,
        limit: int | None = None,
        av_feature_dir: str | Path | None = None,
        require_av_features: bool = False,
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        else:
            resolved = []
            for file in files:
                path = Path(file)
                if path.is_absolute() or path.exists():
                    resolved.append(path)
                else:
                    resolved.append(self.data_dir / path)
            files = resolved
        if limit is not None:
            files = list(files)[: max(1, min(int(limit), len(files)))]
        if not files:
            raise RuntimeError(f"No .pt cache files found under {self.data_dir}")

        self.files = list(files)
        self.max_frames = int(max_frames)
        self.random_crop = bool(random_crop)
        self.rng = random.Random(seed)
        self.feature_index, self.av_feature_dim = build_feature_index(av_feature_dir)
        self.require_av_features = bool(require_av_features)
        if self.require_av_features and not self.feature_index:
            raise RuntimeError(f"No AV-HuBERT feature files found under {av_feature_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _feature_path_for(self, item: dict[str, Any], path: Path) -> Path | None:
        for key in _r2_keys(item, path):
            found = self.feature_index.get(key)
            if found is not None:
                return found
        return None

    def _crop(self, item: dict[str, Any]) -> tuple[dict[str, Any], int, int, int]:
        v_len = int(item["video_len"])
        start = 0
        end = v_len
        if self.max_frames > 0 and v_len > self.max_frames:
            if self.random_crop:
                start = self.rng.randint(0, v_len - self.max_frames)
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
            if "speech_units" in item:
                item["speech_units"] = item["speech_units"][mel_mask]
            item["video_len"] = int(item["video"].shape[1])
            item["mel_len"] = int(item["mel"].shape[0])
        return item, start, end, v_len

    @staticmethod
    def _crop_features(features: torch.Tensor, start: int, end: int, original_video_len: int) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] == 0:
            return features.float().view(0, int(features.shape[-1]) if features.ndim else 0)
        if start <= 0 and end >= original_video_len:
            return features.float()
        feat_len = int(features.shape[0])
        denom = max(1, int(original_video_len))
        f_start = int(round(float(start) * feat_len / denom))
        f_end = int(round(float(end) * feat_len / denom))
        f_start = max(0, min(feat_len - 1, f_start))
        f_end = max(f_start + 1, min(feat_len, f_end))
        return features[f_start:f_end].float()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        raw = _load_r2inr_cache(path)
        feature_path = self._feature_path_for(raw, path) if self.feature_index else None
        if self.require_av_features and feature_path is None:
            raise RuntimeError(f"No AV-HuBERT feature match for {path}")

        item, start, end, original_video_len = self._crop(raw)
        out: dict[str, Any] = {
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
            "transcript_text": item.get("transcript_text", ""),
        }
        if "speech_units" in item:
            out["speech_units"] = item["speech_units"].long()
            out["num_speech_units"] = int(item.get("num_speech_units", int(out["speech_units"].max().item()) + 1))

        if feature_path is not None:
            feat_item = _load_av_feature_cache(feature_path)
            features = self._crop_features(feat_item["features"], start, end, original_video_len)
            out["av_features"] = features
            out["av_feature_len"] = int(features.shape[0])
            out["av_feature_path"] = str(feature_path)
        return out


def _pad_video(video: torch.Tensor, length: int) -> torch.Tensor:
    return video if video.shape[1] == length else F.pad(video, (0, 0, 0, 0, 0, length - video.shape[1]))


def _pad_2d(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, 0, 0, length - x.shape[0]))


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_bool(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=False)


def _pad_long(x: torch.Tensor, length: int, value: int = -100) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=value)


def collate_v10(batch: list[dict[str, Any]]) -> dict[str, Any]:
    v_lens = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
    m_lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    t_video = int(v_lens.max().item())
    t_mel = int(m_lens.max().item())
    paths = [b["path"] for b in batch]
    sources = [b["source_video"] for b in batch]

    video = torch.stack([_pad_video(b["video"], t_video) for b in batch], dim=0)
    landmarks = torch.stack(
        [
            _pad_2d(b["landmarks"].flatten(1), t_video).view(
                t_video, b["landmarks"].shape[1], b["landmarks"].shape[2]
            )
            for b in batch
        ],
        dim=0,
    )
    mel = torch.stack([_pad_2d(b["mel"], t_mel) for b in batch], dim=0)
    video_times = torch.stack([_pad_1d(b["video_times"], t_video) for b in batch], dim=0)
    mel_times = torch.stack([_pad_1d(b["mel_times"], t_mel) for b in batch], dim=0)
    mouth_valid = torch.stack([_pad_bool(b["mouth_valid_mask"], t_video) for b in batch], dim=0)
    video_mask = torch.arange(t_video).unsqueeze(0) < v_lens.unsqueeze(1)
    mel_mask = torch.arange(t_mel).unsqueeze(0) < m_lens.unsqueeze(1)

    out: dict[str, Any] = {
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
        "transcript_texts": [b.get("transcript_text", "") for b in batch],
    }
    if all("speech_units" in b for b in batch):
        out["speech_units"] = torch.stack([_pad_long(b["speech_units"], t_mel, value=-100) for b in batch], dim=0)
        out["speech_unit_mask"] = mel_mask.clone()
        out["num_speech_units"] = max(int(b.get("num_speech_units", 0)) for b in batch)

    feature_items = [b for b in batch if "av_features" in b and int(b.get("av_feature_len", 0)) > 0]
    if feature_items:
        feature_dim = int(feature_items[0]["av_features"].shape[-1])
        feature_lengths = torch.tensor([int(b.get("av_feature_len", 0)) for b in batch], dtype=torch.long)
        max_len = max(1, int(feature_lengths.max().item()))
        padded = []
        for b in batch:
            feats = b.get("av_features")
            if not torch.is_tensor(feats) or feats.numel() == 0:
                feats = torch.zeros(0, feature_dim, dtype=torch.float32)
            padded.append(_pad_2d(feats.float(), max_len))
        out["av_features"] = torch.stack(padded, dim=0)
        out["av_feature_lengths"] = feature_lengths
        out["av_feature_mask"] = torch.arange(max_len).unsqueeze(0) < feature_lengths.unsqueeze(1)
        out["av_feature_present"] = feature_lengths.gt(0)
        out["av_feature_paths"] = [b.get("av_feature_path", "") for b in batch]
    return out
