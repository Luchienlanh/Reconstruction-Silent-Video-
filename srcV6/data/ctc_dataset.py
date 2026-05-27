from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .text import build_vocab, normalize_text, normalize_text_nodiac, tokenize_text_unit


def load_cache(path: str | Path) -> dict[str, Any]:
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("format") not in {"r2inr_text_v1", "r2inr_v1"}:
        raise ValueError(f"{path} is not an r2inr text cache file.")
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


def build_char_vocab_from_files(files: list[Path], min_freq: int = 1, text_unit: str = "char") -> dict[str, int]:
    texts = []
    for path in files:
        item = load_cache(path)
        texts.append(str(item.get("transcript_text", "")))
    return build_vocab(texts, min_freq=min_freq, text_unit=text_unit)


def text_to_ids(text: str, vocab: dict[str, int], text_unit: str = "char") -> torch.Tensor:
    unk = int(vocab.get("<unk>", 2))
    ids = [int(vocab.get(tok, unk)) for tok in tokenize_text_unit(text, text_unit=text_unit)]
    return torch.tensor(ids, dtype=torch.long)


class FullClipCTCDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        vocab: dict[str, int],
        files: list[str | Path] | None = None,
        frame_stride: int = 1,
        min_input_target_ratio: float = 1.05,
        text_unit: str = "char",
    ):
        self.data_dir = Path(data_dir)
        if files is None:
            files = sorted(self.data_dir.glob("*.pt"))
        resolved = [Path(f) if Path(f).is_absolute() or Path(f).exists() else self.data_dir / Path(f) for f in files]
        self.vocab = vocab
        self.frame_stride = max(1, int(frame_stride))
        self.min_input_target_ratio = float(min_input_target_ratio)
        self.text_unit = str(text_unit)
        self.files: list[Path] = []
        self.skipped: list[tuple[str, int, int]] = []
        for path in resolved:
            item = load_cache(path)
            token_len = int(text_to_ids(str(item.get("transcript_text", "")), vocab, self.text_unit).numel())
            input_len = max(1, (int(item["video_len"]) + self.frame_stride - 1) // self.frame_stride)
            if input_len >= max(1, int(round(token_len * self.min_input_target_ratio))):
                self.files.append(path)
            else:
                self.skipped.append((str(path), input_len, token_len))
        if not self.files:
            raise RuntimeError("No CTC-usable files after input/target length filtering.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = load_cache(path)
        stride = self.frame_stride
        video = item["video"][:, ::stride].float()
        landmarks = item["landmarks"][::stride].float()
        video_times = item["video_times"][::stride].float()
        mouth_valid = item["mouth_valid_mask"][::stride].bool()
        raw_text = str(item.get("transcript_text", ""))
        text = normalize_text_nodiac(raw_text) if self.text_unit.endswith("_nodiac") else normalize_text(raw_text)
        token_ids = text_to_ids(raw_text, self.vocab, self.text_unit)
        return {
            "video": video,
            "landmarks": landmarks,
            "video_times": video_times,
            "mouth_valid_mask": mouth_valid,
            "video_len": int(video.shape[1]),
            "target_ids": token_ids,
            "target_len": int(token_ids.numel()),
            "transcript_text": text,
            "path": str(path),
            "source_video": item.get("source_video", ""),
        }


def _pad_video(video: torch.Tensor, length: int) -> torch.Tensor:
    return video if video.shape[1] == length else F.pad(video, (0, 0, 0, 0, 0, length - video.shape[1]))


def _pad_landmarks(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, 0, 0, 0, 0, length - x.shape[0]))


def _pad_1d(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=value)


def _pad_bool(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=False)


def _pad_tokens(x: torch.Tensor, length: int) -> torch.Tensor:
    return x if x.shape[0] == length else F.pad(x, (0, length - x.shape[0]), value=0)


def collate_ctc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    video_lengths = torch.tensor([b["video_len"] for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)
    max_video = int(video_lengths.max().item())
    max_target = int(target_lengths.max().item())
    video = torch.stack([_pad_video(b["video"], max_video) for b in batch], dim=0)
    landmarks = torch.stack([_pad_landmarks(b["landmarks"], max_video) for b in batch], dim=0)
    video_times = torch.stack([_pad_1d(b["video_times"], max_video) for b in batch], dim=0)
    mouth_valid = torch.stack([_pad_bool(b["mouth_valid_mask"], max_video) for b in batch], dim=0)
    targets = torch.stack([_pad_tokens(b["target_ids"], max_target) for b in batch], dim=0)
    video_mask = torch.arange(max_video).unsqueeze(0) < video_lengths.unsqueeze(1)
    target_mask = torch.arange(max_target).unsqueeze(0) < target_lengths.unsqueeze(1)
    return {
        "video": video,
        "landmarks": landmarks,
        "video_times": video_times,
        "mouth_valid_mask": mouth_valid,
        "video_mask": video_mask,
        "video_lengths": video_lengths,
        "target_ids": targets,
        "target_lengths": target_lengths,
        "target_mask": target_mask,
        "transcript_texts": [b["transcript_text"] for b in batch],
        "paths": [b["path"] for b in batch],
        "source_videos": [b.get("source_video", "") for b in batch],
    }
