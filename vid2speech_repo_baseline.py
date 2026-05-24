"""
PyTorch baseline matching arielephrat/vid2speech more closely.

Original repo idea:
  - crop full grayscale face to 128x128
  - stack 9 neighboring frames as channels
  - predict the audio feature for the center video frame
  - normalize X and Y with train statistics
  - train a plain 2D CNN + FC model with MSE

This adaptation uses this project's processed .pt files. The target is a small
mel window around the center video frame, flattened to mel_per_frame * 80.
It is intentionally separate from src/ and avoids the current encoder/decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VNLipDatasetV2  # noqa: E402


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(str(path).replace("\\", "/"))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_video_mel(path: str) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    video = data["video"].float()
    if video.dim() == 3:
        video = video.unsqueeze(0)
    if video.dim() != 4:
        raise ValueError(f"Expected video (C,T,H,W), got {tuple(video.shape)} in {path}")

    mel = data["mel"].float()
    n_mels = int(data.get("n_mels", 80))
    if mel.dim() != 2:
        raise ValueError(f"Expected mel 2D, got {tuple(mel.shape)} in {path}")
    if mel.shape[0] == n_mels and mel.shape[1] != n_mels:
        mel = mel.transpose(0, 1).contiguous()

    video_len = min(int(data.get("video_len", video.shape[1])), video.shape[1])
    mel_len = min(int(data.get("mel_len", mel.shape[0])), mel.shape[0])
    return video[:, :video_len], mel[:mel_len], video_len, mel_len


def normalize_video_uint_range(video: torch.Tensor) -> torch.Tensor:
    video = torch.nan_to_num(video.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if video.shape[0] > 1:
        video = video.mean(dim=0, keepdim=True)
    vmax = float(video.max())
    vmin = float(video.min())
    if vmax > 1.5 or vmin < -0.5:
        video = video / 255.0
    return video.clamp(0.0, 1.0)


def resize_video(video: torch.Tensor, size: int) -> torch.Tensor:
    # video: (1,T,H,W)
    if video.shape[-2:] == (size, size):
        return video
    x = video.transpose(0, 1)  # (T,1,H,W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.transpose(0, 1).contiguous()


def target_window(mel: torch.Tensor, center_frame: int, video_len: int, mel_per_frame: int) -> torch.Tensor:
    mel_len = mel.shape[0]
    center = int(round((center_frame + 0.5) * mel_len / max(video_len, 1) - 0.5))
    half = mel_per_frame // 2
    indices = []
    for offset in range(mel_per_frame):
        idx = center + offset - half
        idx = max(0, min(mel_len - 1, idx))
        indices.append(idx)
    return mel[indices].reshape(-1)


class Vid2SpeechFrameDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        files: list[str],
        nframes: int = 9,
        frame_size: int = 128,
        mel_per_frame: int = 3,
        max_centers_per_file: int = 0,
        random_centers: bool = True,
        seed: int = 42,
    ):
        if nframes < 3 or nframes % 2 == 0:
            raise ValueError("--nframes must be odd and >= 3")
        self.data_dir = data_dir
        self.files = list(files)
        self.nframes = int(nframes)
        self.margin = self.nframes // 2
        self.frame_size = int(frame_size)
        self.mel_per_frame = int(mel_per_frame)
        self.samples: list[tuple[str, int]] = []

        rng = random.Random(seed)
        for file_name in self.files:
            path = file_name if os.path.isabs(str(file_name)) else os.path.join(self.data_dir, file_name)
            video, _mel, video_len, _mel_len = load_video_mel(path)
            valid = list(range(video_len))
            if max_centers_per_file > 0 and len(valid) > max_centers_per_file:
                if random_centers:
                    valid = sorted(rng.sample(valid, max_centers_per_file))
                else:
                    step = max(1, int(math.floor(len(valid) / max_centers_per_file)))
                    valid = valid[::step][:max_centers_per_file]
            self.samples.extend((file_name, int(center)) for center in valid)

    def __len__(self) -> int:
        return len(self.samples)

    def _frame_stack(self, video: torch.Tensor, center: int) -> torch.Tensor:
        video = resize_video(normalize_video_uint_range(video), self.frame_size)
        frames = []
        t = video.shape[1]
        for offset in range(-self.margin, self.margin + 1):
            idx = max(0, min(t - 1, center + offset))
            frames.append(video[0, idx])
        return torch.stack(frames, dim=0).float()

    def __getitem__(self, idx: int):
        file_name, center = self.samples[idx]
        path = file_name if os.path.isabs(str(file_name)) else os.path.join(self.data_dir, file_name)
        video, mel, video_len, _mel_len = load_video_mel(path)
        x = self._frame_stack(video, center)
        y = target_window(mel, center, video_len, self.mel_per_frame)
        return x, y, path, center


class Vid2SpeechCNN(nn.Module):
    def __init__(self, nframes: int = 9, out_dim: int = 240, dropout: float = 0.25, dropout_fc: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nframes, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_fc),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(dropout_fc),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, out_dim),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def split_files(args) -> tuple[list[str], list[str]]:
    data_dir = resolve_path(args.data_dir)
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))
    if args.limit is not None:
        files = files[: max(1, min(int(args.limit), len(files)))]
    rng = random.Random(args.seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * args.val_ratio))) if len(files) > 1 and args.val_ratio > 0 else 0
    train_files = sorted(files[val_count:])
    val_files = sorted(files[:val_count])
    if not train_files:
        raise RuntimeError("No train files after split.")
    return train_files, val_files


def collate_batch(batch):
    xs, ys, paths, centers = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(paths), torch.tensor(centers, dtype=torch.long)


def compute_stats(loader: DataLoader, device: torch.device) -> tuple[float, torch.Tensor, torch.Tensor]:
    x_sum = 0.0
    x_count = 0
    y_sum = None
    y_sq = None
    y_count = 0
    for x, y, _paths, _centers in tqdm(loader, desc="stats", leave=False):
        x_sum += float(x.sum())
        x_count += int(x.numel())
        y = y.float()
        y_sum = y.sum(dim=0) if y_sum is None else y_sum + y.sum(dim=0)
        y_sq = y.pow(2).sum(dim=0) if y_sq is None else y_sq + y.pow(2).sum(dim=0)
        y_count += int(y.shape[0])
    x_mean = x_sum / max(1, x_count)
    y_mean = y_sum / max(1, y_count)
    y_var = y_sq / max(1, y_count) - y_mean.pow(2)
    y_std = y_var.clamp_min(1e-6).sqrt().clamp_min(0.05)
    return float(x_mean), y_mean.to(device), y_std.to(device)


def mean_baseline(loader: Optional[DataLoader], y_mean: torch.Tensor, y_std: torch.Tensor, device: torch.device) -> Optional[float]:
    if loader is None:
        return None
    total = 0.0
    count = 0
    zero_pred = torch.zeros(1, y_mean.numel(), device=device)
    for _x, y, _paths, _centers in tqdm(loader, desc="mean-baseline", leave=False):
        y = y.to(device)
        y_norm = (y - y_mean) / y_std
        pred = zero_pred.expand_as(y_norm)
        total += float(F.mse_loss(pred, y_norm).detach().cpu())
        count += 1
    return total / max(1, count)


def run_epoch(model, loader, optimizer, scaler, device, args, x_mean: float, y_mean: torch.Tensor, y_std: torch.Tensor):
    model.train()
    total = 0.0
    count = 0
    amp_enabled = device.type == "cuda" and args.amp
    for x, y, _paths, _centers in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True).float() - x_mean
        y = y.to(device, non_blocking=True).float()
        y_norm = (y - y_mean) / y_std
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(x)
            loss = F.mse_loss(pred.float(), y_norm.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


@torch.no_grad()
def evaluate(model, loader, device, args, x_mean: float, y_mean: torch.Tensor, y_std: torch.Tensor) -> Optional[float]:
    if loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    for x, y, _paths, _centers in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True).float() - x_mean
        y = y.to(device, non_blocking=True).float()
        y_norm = (y - y_mean) / y_std
        pred = model(x)
        total += float(F.mse_loss(pred.float(), y_norm.float()).detach().cpu())
        count += 1
    return total / max(1, count)


def state_dict(model: nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def run(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "checkpoints_vid2speech_repo"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files = split_files(args)
    train_ds = Vid2SpeechFrameDataset(
        str(data_dir),
        train_files,
        nframes=args.nframes,
        frame_size=args.frame_size,
        mel_per_frame=args.mel_per_frame,
        max_centers_per_file=args.max_centers_per_file,
        random_centers=True,
        seed=args.seed,
    )
    val_ds = Vid2SpeechFrameDataset(
        str(data_dir),
        val_files,
        nframes=args.nframes,
        frame_size=args.frame_size,
        mel_per_frame=args.mel_per_frame,
        max_centers_per_file=args.max_val_centers_per_file or args.max_centers_per_file,
        random_centers=False,
        seed=args.seed,
    ) if val_files else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    ) if val_ds is not None else None

    stats_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    x_mean, y_mean, y_std = compute_stats(stats_loader, device)

    model = Vid2SpeechCNN(
        nframes=args.nframes,
        out_dim=args.mel_per_frame * 80,
        dropout=args.dropout,
        dropout_fc=args.dropout_fc,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    mean_train = mean_baseline(train_loader, y_mean, y_std, device)
    mean_val = mean_baseline(val_loader, y_mean, y_std, device)

    print(f"[device] {device}")
    print(f"[data] {safe_text(data_dir)}")
    print(f"[split] files train={len(train_files)} val={len(val_files)} frame_samples train={len(train_ds)} val={0 if val_ds is None else len(val_ds)}")
    print(f"[model] vid2speech_repo nframes={args.nframes} frame_size={args.frame_size} out={args.mel_per_frame * 80}")
    print(f"[stats] x_mean={x_mean:.6f} y_mean_avg={float(y_mean.mean()):.4f} y_std_avg={float(y_std.mean()):.4f}")
    mean_val_text = "n/a" if mean_val is None else f"{mean_val:.6f}"
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={mean_val_text}")

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, args, x_mean, y_mean, y_std)
        val_loss = evaluate(model, val_loader, device, args, x_mean, y_mean, y_std)
        score = train_loss if val_loss is None else val_loss
        is_best = score < best
        if is_best:
            best = score
            torch.save(
                {
                    "model_state_dict": state_dict(model),
                    "config": vars(args),
                    "x_mean": x_mean,
                    "y_mean": y_mean.detach().cpu(),
                    "y_std": y_std.detach().cpu(),
                    "epoch": epoch,
                    "best": best,
                },
                output_dir / "best_model.pth",
            )
        torch.save(
            {
                "model_state_dict": state_dict(model),
                "config": vars(args),
                "x_mean": x_mean,
                "y_mean": y_mean.detach().cpu(),
                "y_std": y_std.detach().cpu(),
                "epoch": epoch,
                "best": best,
            },
            output_dir / "last_model.pth",
        )
        row = {
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "best": best,
            "mean_train": mean_train,
            "mean_val": mean_val,
            "gap_vs_mean": (val_loss - mean_val) if val_loss is not None and mean_val is not None else None,
        }
        history.append(row)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"history": history, "config": vars(args)}, f, indent=2)
        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        gap_text = "" if row["gap_vs_mean"] is None else f" gap_vs_mean={row['gap_vs_mean']:+.6f}"
        mark = " best" if is_best else ""
        print(f"[epoch {epoch:04d}] train={train_loss:.6f} val={val_text} best={best:.6f}{gap_text}{mark}")

    print(f"[done] {safe_text(output_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(description="arielephrat/vid2speech-style CNN baseline adapted to mel data.")
    parser.add_argument("--data-dir", default="FullFrame_test")
    parser.add_argument("--output-dir", default="checkpoints_vid2speech_repo")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nframes", type=int, default=9)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--mel-per-frame", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--dropout-fc", type=float, default=0.5)
    parser.add_argument("--max-centers-per-file", type=int, default=0)
    parser.add_argument("--max-val-centers-per-file", type=int, default=0)
    parser.add_argument("--drop-last", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
