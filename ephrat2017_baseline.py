"""
Ephrat 2017-style baseline for silent video -> mel reconstruction.

This script is intentionally separate from src/. It implements the paper-style
pipeline as a diagnostic baseline:
  - grayscale clip tower
  - dense optical-flow clip tower
  - residual CNN towers
  - fully connected mel decoder
  - CBHG-like post-processing network

The original paper predicts mel and then a linear spectrogram. This repository's
processed data provides HiFi-GAN mel targets, so the postnet refines mel again.
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VNLipDatasetV2, collate_pad_v2  # noqa: E402


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    if str(path).strip().lower() in {"none", "null", "off", "false"}:
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


class Bottleneck2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        mid = max(out_channels // 4, 32)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        return self.act(self.body(x) + residual)


class EphratTower(nn.Module):
    def __init__(self, in_channels: int, width_scale: float = 1.0):
        super().__init__()
        raw_channels = [128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        channels = [max(32, int(c * width_scale)) for c in raw_channels]
        strides = [2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
        blocks = []
        prev = in_channels
        for out_ch, stride in zip(channels, strides):
            blocks.append(Bottleneck2D(prev, out_ch, stride=stride))
            prev = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(prev, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.out(x)


class Highway(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.h = nn.Linear(dim, dim)
        self.t = nn.Linear(dim, dim)
        nn.init.constant_(self.t.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.h(x))
        t = torch.sigmoid(self.t(x))
        return h * t + x * (1.0 - t)


class CBHGLitePostnet(nn.Module):
    def __init__(self, mel_dim: int = 80, channels: int = 128, bank_size: int = 8, highway_layers: int = 4):
        super().__init__()
        self.bank = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(mel_dim, channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                )
                for k in range(1, bank_size + 1)
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv1d(channels * bank_size, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, mel_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(mel_dim),
        )
        self.highways = nn.ModuleList([Highway(mel_dim) for _ in range(highway_layers)])
        self.gru = nn.GRU(mel_dim, channels // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(channels, mel_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, T, 80)
        x = mel.transpose(1, 2)
        bank = []
        for conv in self.bank:
            y = conv(x)
            if y.shape[-1] != x.shape[-1]:
                y = y[..., : x.shape[-1]]
            bank.append(y)
        y = torch.cat(bank, dim=1)
        y = F.max_pool1d(y, kernel_size=2, stride=1, padding=1)[..., : x.shape[-1]]
        y = self.proj(y).transpose(1, 2)
        y = y + mel
        for layer in self.highways:
            y = layer(y)
        y, _ = self.gru(y)
        return self.out(y)


class Ephrat2017Model(nn.Module):
    def __init__(
        self,
        clip_frames: int = 5,
        mel_per_frame: int = 3,
        width_scale: float = 1.0,
        hidden_dim: int = 1024,
        post_channels: int = 128,
    ):
        super().__init__()
        if clip_frames < 3 or clip_frames % 2 == 0:
            raise ValueError("--clip-frames must be an odd integer >= 3")
        self.clip_frames = clip_frames
        self.mel_per_frame = mel_per_frame
        self.image_tower = EphratTower(in_channels=clip_frames, width_scale=width_scale)
        self.flow_tower = EphratTower(in_channels=2 * (clip_frames - 1), width_scale=width_scale)
        self.decoder = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mel_per_frame * 80),
        )
        self.postnet = CBHGLitePostnet(mel_dim=80, channels=post_channels)
        self._init_output_bias(-4.0)

    def _init_output_bias(self, value: float) -> None:
        last = self.decoder[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, value)

    def _clip_channels(self, x: torch.Tensor, drop_first: bool = False) -> torch.Tensor:
        # x: (B, C, T, H, W) -> (B*T, C*K, H, W)
        b, c, t, h, w = x.shape
        pad = self.clip_frames // 2
        padded = F.pad(x, (0, 0, 0, 0, pad, pad), mode="replicate")
        chunks = []
        start = 1 if drop_first else 0
        for offset in range(start, self.clip_frames):
            chunks.append(padded[:, :, offset : offset + t])
        x = torch.cat(chunks, dim=1)
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, -1, h, w)

    def forward(self, video: torch.Tensor, flow: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if video.dim() != 5 or flow.dim() != 5:
            raise ValueError(f"Expected video/flow 5D, got video={tuple(video.shape)} flow={tuple(flow.shape)}")
        b, _, t, _, _ = video.shape
        image_clip = self._clip_channels(video, drop_first=False)
        flow_clip = self._clip_channels(flow, drop_first=True)
        z_image = self.image_tower(image_clip)
        z_flow = self.flow_tower(flow_clip)
        z = torch.cat([z_image, z_flow], dim=-1)
        mel = self.decoder(z).view(b, t * self.mel_per_frame, 80)
        if mel.shape[1] != target_len:
            mel = F.interpolate(mel.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        refined = self.postnet(mel)
        return mel, refined


def masked_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor, kind: str = "l1") -> torch.Tensor:
    max_t = target.shape[1]
    mask = torch.arange(max_t, device=target.device)[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).to(pred.dtype)
    if kind == "mse":
        loss = (pred - target).pow(2) * mask
    else:
        loss = (pred - target).abs() * mask
    return loss.sum() / mask.sum().clamp_min(1.0) / target.shape[-1]


def unpack_batch(batch, device: torch.device):
    if len(batch) != 6:
        raise ValueError("Ephrat baseline requires optical flow. Use --flow-cache-dir none or a cache dir.")
    video, flow, _landmarks, target, lengths, paths = batch
    return (
        video.to(device, non_blocking=True),
        flow.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        lengths.to(device, non_blocking=True),
        paths,
    )


def forward_model(model: nn.Module, video: torch.Tensor, flow: torch.Tensor, target_len: int):
    if isinstance(model, nn.DataParallel) and video.shape[0] < len(model.device_ids):
        return model.module(video, flow, target_len=target_len)
    return model(video, flow, target_len=target_len)


def make_dataset(args, files: list[str], random_crop: bool) -> VNLipDatasetV2:
    data_dir = resolve_path(args.data_dir)
    dataset_output_dir = resolve_path(args.dataset_output_dir)
    flow_cache_dir = resolve_path(args.flow_cache_dir)
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")
    ds = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=random_crop,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(dataset_output_dir or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
        force_full_frame=args.force_full_frame,
        use_optical_flow=True,
        flow_cache_dir=str(flow_cache_dir) if flow_cache_dir is not None else None,
        flow_method=args.flow_method,
        flow_scale=args.flow_scale,
    )
    ds.files = list(files)
    return ds


def split_files(args) -> tuple[list[str], list[str], int]:
    data_dir = resolve_path(args.data_dir)
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")
    probe = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=False,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(resolve_path(args.dataset_output_dir) or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
        force_full_frame=args.force_full_frame,
        use_optical_flow=False,
    )
    files = list(probe.files)
    if args.limit is not None:
        files = files[: max(1, min(int(args.limit), len(files)))]
    rng = random.Random(args.seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * args.val_ratio))) if len(files) > 1 and args.val_ratio > 0 else 0
    train_files = sorted(files[val_count:])
    val_files = sorted(files[:val_count])
    if not train_files:
        raise RuntimeError("No train files after split.")
    return train_files, val_files, int(probe.landmark_num_points or 0)


def make_loaders(args):
    train_files, val_files, _ = split_files(args)
    train_ds = make_dataset(args, train_files, random_crop=True)
    val_ds = make_dataset(args, val_files, random_crop=False) if val_files else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad_v2,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_pad_v2,
        )
    return train_ds, val_ds, train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, args) -> Optional[float]:
    if loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        video, flow, target, lengths, _ = unpack_batch(batch, device)
        mel, refined = forward_model(model, video, flow, target_len=target.shape[1])
        loss = masked_loss(mel.float(), target.float(), lengths, args.loss)
        loss = loss + args.postnet_weight * masked_loss(refined.float(), target.float(), lengths, args.loss)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def train_one_epoch(model, loader, optimizer, scaler, device, args) -> float:
    model.train()
    total = 0.0
    count = 0
    amp_enabled = device.type == "cuda" and args.amp
    for batch in tqdm(loader, desc="train", leave=False):
        video, flow, target, lengths, _ = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            mel, refined = forward_model(model, video, flow, target_len=target.shape[1])
            loss = masked_loss(mel.float(), target.float(), lengths, args.loss)
            loss = loss + args.postnet_weight * masked_loss(refined.float(), target.float(), lengths, args.loss)
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


def state_dict(model: nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def run(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "checkpoints_ephrat2017"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, train_loader, val_loader = make_loaders(args)

    model = Ephrat2017Model(
        clip_frames=args.clip_frames,
        mel_per_frame=args.mel_per_frame,
        width_scale=args.width_scale,
        hidden_dim=args.hidden_dim,
        post_channels=args.post_channels,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    best = float("inf")
    history = []

    print(f"[device] {device}")
    print(f"[data] {safe_text(resolve_path(args.data_dir))}")
    print(f"[model] ephrat2017 clip={args.clip_frames} mel_per_frame={args.mel_per_frame} width_scale={args.width_scale}")
    print(f"[flow] method={args.flow_method} cache={safe_text(resolve_path(args.flow_cache_dir)) if resolve_path(args.flow_cache_dir) else 'none'}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss = evaluate(model, val_loader, device, args)
        score = train_loss if val_loss is None else val_loss
        is_best = score < best
        if is_best:
            best = score
            torch.save({"model_state_dict": state_dict(model), "config": vars(args), "epoch": epoch, "best": best}, output_dir / "best_model.pth")
        torch.save({"model_state_dict": state_dict(model), "config": vars(args), "epoch": epoch, "best": best}, output_dir / "last_model.pth")

        row = {"epoch": epoch, "train": train_loss, "val": val_loss, "best": best}
        history.append(row)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"history": history, "config": vars(args)}, f, indent=2)
        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        mark = " best" if is_best else ""
        print(f"[epoch {epoch:04d}] train={train_loss:.6f} val={val_text} best={best:.6f}{mark}")

    print(f"[done] {safe_text(output_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ephrat 2017-style dual tower baseline.")
    parser.add_argument("--data-dir", default="FullFrame_test")
    parser.add_argument("--dataset-output-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default="checkpoints_ephrat2017")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-full-frame", action="store_true")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--flow-cache-dir", default="none")
    parser.add_argument("--flow-method", default="farneback", choices=["farneback", "pseudo"])
    parser.add_argument("--flow-scale", type=float, default=10.0)
    parser.add_argument("--clip-frames", type=int, default=5)
    parser.add_argument("--mel-per-frame", type=int, default=3)
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--post-channels", type=int, default=128)
    parser.add_argument("--postnet-weight", type=float, default=1.0)
    parser.add_argument("--loss", default="l1", choices=["l1", "mse"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
