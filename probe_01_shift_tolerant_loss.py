"""
Probe 01: shift-tolerant mel loss.

This script is also the shared probe harness for the other probe_*.py files.
It intentionally lives outside src/ and imports project modules from src/.
Default dataset is FullFrame_test and the run uses the whole split unless
--limit is provided.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Optional

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
from models.decoders.upsample import MelTemporalUpsampleDecoder  # noqa: E402
from models.encoders.factory import (  # noqa: E402
    LandmarkMotionEncoder,
    VisualLandmarkEncoder,
    VisualLandmarkEncoderLandmarkFirst,
    VisualLandmarkEncoderV2,
    build_encoder,
)
from models.loss import MelReconstructionLoss  # noqa: E402
import main as train_main  # noqa: E402


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def resolve_path(path: str | Path | None) -> Optional[Path]:
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


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-dir", default="FullFrame_test")
    parser.add_argument("--dataset-output-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "nonsnn", "cnn_transformer", "snn"])
    parser.add_argument("--fusion-type", default="landmark_first", choices=["landmark_first", "concat", "cross_attn"])
    parser.add_argument("--decoder-type", default="direct_tcn", choices=["direct_tcn", "siren"])
    parser.add_argument("--decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--decoder-num-layers", type=int, default=6)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--crop-mouth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mouth-roi", type=int, nargs=4, default=[45, 80, 32, 80])
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--force-full-frame", action="store_true")
    parser.add_argument("--mel-stats-max-samples", type=int, default=0)
    parser.add_argument("--lambda-delta", type=float, default=0.0)
    parser.add_argument("--lambda-delta2", type=float, default=0.0)
    parser.add_argument("--lambda-energy", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def device_from_args(args: argparse.Namespace) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_wrap_model(model: nn.Module, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if (
        device.type == "cuda"
        and getattr(args, "multi_gpu", True)
        and torch.cuda.device_count() > 1
        and not isinstance(model, nn.DataParallel)
    ):
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using nn.DataParallel.")
        return nn.DataParallel(model)
    return model


def make_dataset(args: argparse.Namespace, files: list[str], random_crop: bool) -> VNLipDatasetV2:
    data_dir = resolve_path(args.data_dir)
    dataset_output_dir = resolve_path(args.dataset_output_dir)
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
    )
    ds.files = list(files)
    return ds


def split_files(args: argparse.Namespace) -> tuple[list[str], list[str], int]:
    data_dir = resolve_path(args.data_dir)
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")
    probe_ds = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=False,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(resolve_path(args.dataset_output_dir) or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
        force_full_frame=args.force_full_frame,
    )
    files = list(probe_ds.files)
    if args.limit is not None:
        files = files[: max(1, min(int(args.limit), len(files)))]
    rng = random.Random(args.seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * args.val_ratio))) if len(shuffled) > 1 else 0
    train_files = sorted(shuffled[val_count:])
    val_files = sorted(shuffled[:val_count])
    if not train_files:
        raise RuntimeError("No training files after split.")
    return train_files, val_files, int(probe_ds.landmark_num_points)


def make_loaders(args: argparse.Namespace) -> tuple[VNLipDatasetV2, Optional[VNLipDatasetV2], DataLoader, Optional[DataLoader], int]:
    train_files, val_files, num_landmark_points = split_files(args)
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
    return train_ds, val_ds, train_loader, val_loader, num_landmark_points


def unpack_batch(batch, device: torch.device, args: argparse.Namespace):
    video, landmarks, target, lengths, paths = batch
    if args.crop_mouth:
        y1, y2, x1, x2 = args.mouth_roi
        video = video[:, :, :, y1:y2, x1:x2]
    return (
        video.to(device, non_blocking=True),
        landmarks.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        lengths.to(device, non_blocking=True),
        paths,
    )


def target_from_item(item) -> torch.Tensor:
    return item[2].float()


def compute_mel_stats(dataset: VNLipDatasetV2, max_samples: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    limit = len(dataset) if max_samples <= 0 else min(len(dataset), int(max_samples))
    total = None
    total_sq = None
    count = 0
    for idx in tqdm(range(limit), desc="mel-stats", leave=False):
        target = target_from_item(dataset[idx])
        total = target.sum(dim=0) if total is None else total + target.sum(dim=0)
        total_sq = target.pow(2).sum(dim=0) if total_sq is None else total_sq + target.pow(2).sum(dim=0)
        count += int(target.shape[0])
    count = max(count, 1)
    mean = total / count
    var = (total_sq / count) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt().clamp_min(0.05)
    return mean.cpu(), std.cpu()


def make_criterion(args: argparse.Namespace, device: torch.device, mel_mean: torch.Tensor, mel_std: torch.Tensor):
    return MelReconstructionLoss(
        lambda_mel=1.0,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
        mel_mean=mel_mean,
        mel_std=mel_std,
    ).to(device)


def init_decoder_bias(decoder: nn.Module, mel_mean: torch.Tensor) -> bool:
    return train_main.init_decoder_output_bias_from_mel_mean(decoder, mel_mean)


class ReconstructionModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, video: torch.Tensor, landmarks: torch.Tensor, target_len: int):
        z = self.encoder(video, landmarks)
        pred = self.decoder(z, target_len=target_len)
        return {"pred": pred, "z": z}


def build_reconstruction_model(args: argparse.Namespace, device: torch.device, num_landmark_points: int) -> ReconstructionModel:
    encoder, decoder = train_main.build_models(
        device,
        args.encoder_type,
        args.decoder_type,
        num_landmark_points,
        fusion_type=args.fusion_type,
        target_type="mel_hifigan",
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        decoder_dropout=args.decoder_dropout,
    )
    return ReconstructionModel(encoder, decoder).to(device)


def mean_baseline_loss(loader: Optional[DataLoader], criterion, device: torch.device, args: argparse.Namespace, mel_mean: torch.Tensor):
    if loader is None:
        return None
    mean = mel_mean.to(device=device, dtype=torch.float32).view(1, 1, -1)
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="mean-baseline", leave=False):
            _, _, target, lengths, _ = unpack_batch(batch, device, args)
            pred = mean.expand(target.shape[0], target.shape[1], -1).contiguous()
            loss = criterion(pred, target, lengths)
            total += float(loss.detach().cpu())
            count += 1
    return total / max(1, count)


def default_loss_fn(model, batch, criterion, device, args):
    video, landmarks, target, lengths, _ = unpack_batch(batch, device, args)
    out = model(video, landmarks, target.shape[1])
    return criterion(out["pred"], target, lengths), out


def train_probe(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion,
    device: torch.device,
    loss_fn: Callable = default_loss_fn,
    after_epoch: Optional[Callable] = None,
    mel_mean: Optional[torch.Tensor] = None,
) -> list[dict]:
    model = maybe_wrap_model(model, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    amp_enabled = device.type == "cuda" and args.amp
    mean_val = mean_baseline_loss(val_loader, criterion, device, args, mel_mean) if mel_mean is not None else None
    if mean_val is not None:
        print(f"[baseline] mean_val={mean_val:.6f}")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        last_out = None
        for batch in tqdm(train_loader, desc=f"train {epoch:04d}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, out = loss_fn(model, batch, criterion, device, args)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite train loss: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            last_out = out
            train_total += float(loss.detach().cpu())
            train_count += 1

        base_model = unwrap_model(model)
        if isinstance(last_out, dict) and "gate" in last_out and hasattr(base_model, "last_gate"):
            base_model.last_gate = last_out["gate"].detach()

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"val {epoch:04d}", leave=False):
                    loss, _ = loss_fn(model, batch, criterion, device, args)
                    val_total += float(loss.detach().cpu())
                    val_count += 1
            val_loss = val_total / max(1, val_count)

        extra = after_epoch(unwrap_model(model), epoch) if after_epoch is not None else {}
        row = {
            "epoch": epoch,
            "train": train_total / max(1, train_count),
            "val": val_loss,
            "mean_val": mean_val,
            "gap_vs_mean": (val_loss - mean_val) if (val_loss is not None and mean_val is not None) else None,
            **(extra or {}),
        }
        history.append(row)
        gap = "" if row["gap_vs_mean"] is None else f" gap_vs_mean={row['gap_vs_mean']:+.6f}"
        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        print(f"[epoch {epoch:04d}] train={row['train']:.6f} val={val_text}{gap}")
    return history


def save_history(args: argparse.Namespace, name: str, history: list[dict], extra: Optional[dict] = None) -> None:
    out = resolve_path(args.output_dir) if args.output_dir else PROJECT_ROOT / f"probe_results_{name}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"probe": name, "config": vars(args), "history": history}
    if extra:
        payload.update(extra)
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[output] {safe_text(out / 'history.json')}")


class ShiftTolerantMelLoss(nn.Module):
    def __init__(self, base_loss: MelReconstructionLoss, max_shift: int = 6):
        super().__init__()
        self.base_loss = base_loss
        self.max_shift = int(max_shift)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        losses = []
        for shift in range(-self.max_shift, self.max_shift + 1):
            if shift < 0:
                s = -shift
                if pred.shape[1] <= s:
                    continue
                p = pred[:, s:]
                t = target[:, : pred.shape[1] - s]
            elif shift > 0:
                if target.shape[1] <= shift:
                    continue
                p = pred[:, : pred.shape[1] - shift]
                t = target[:, shift:]
            else:
                p = pred
                t = target
            L = (lengths - abs(shift)).clamp_min(1).clamp_max(p.shape[1])
            losses.append(self.base_loss(p, t, L))
        return torch.stack(losses).min()


def run_shift_tolerant_probe(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_ds, _, train_loader, val_loader, num_landmark_points = make_loaders(args)
    mel_mean, mel_std = compute_mel_stats(train_ds, args.mel_stats_max_samples)
    base_criterion = make_criterion(args, device, mel_mean, mel_std)
    criterion = ShiftTolerantMelLoss(base_criterion, max_shift=args.max_shift).to(device)
    model = build_reconstruction_model(args, device, num_landmark_points)
    init_decoder_bias(model.decoder, mel_mean)
    print(f"[probe] shift_tolerant max_shift={args.max_shift} files={len(train_ds)}")
    history = train_probe(args, model, train_loader, val_loader, criterion, device, mel_mean=mel_mean)
    save_history(args, "01_shift_tolerant_loss", history, {"mel_mean_avg": float(mel_mean.mean()), "mel_std_avg": float(mel_std.mean())})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 01: train with shift-tolerant mel loss on FullFrame_test.")
    add_common_args(parser)
    parser.add_argument("--max-shift", type=int, default=6, help="Mel-frame shift search window.")
    return parser.parse_args()


if __name__ == "__main__":
    run_shift_tolerant_probe(parse_args())
