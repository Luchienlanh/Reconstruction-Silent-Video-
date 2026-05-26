from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV7.data import R2CacheDataset, collate_r2cache
from srcV7.models import MaskedMelLoss, R2CNNFiLMModel
from srcV7.utils.common import batch_to_device


def split_cache_files(data_dir: str | Path, val_ratio: float = 0.1, seed: int = 42, limit: int | None = None):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.pt"))
    if limit is not None:
        files = files[: max(1, min(int(limit), len(files)))]
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


def make_loader(
    data_dir,
    files,
    batch_size,
    max_frames=0,
    random_crop=False,
    seed=42,
    num_workers=0,
    shuffle=False,
    drop_last=False,
):
    ds = R2CacheDataset(data_dir, files=files, max_frames=max_frames, random_crop=random_crop, seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r2cache,
        drop_last=drop_last,
    )


def model_inputs(batch: dict) -> dict:
    allowed = (
        "video",
        "landmarks",
        "video_times",
        "mel_times",
        "video_mask",
        "mel_mask",
        "mouth_valid_mask",
        "video_lengths",
        "mel_lengths",
    )
    return {key: batch[key] for key in allowed if key in batch}


@torch.no_grad()
def compute_mel_stats(loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    total = None
    sq = None
    count = 0
    for batch in tqdm(loader, desc="mel-stats", leave=False):
        mel = batch["mel"].to(device).float()
        mask = batch["mel_mask"].to(device).bool()
        vals = mel[mask]
        total = vals.sum(dim=0) if total is None else total + vals.sum(dim=0)
        sq = vals.pow(2).sum(dim=0) if sq is None else sq + vals.pow(2).sum(dim=0)
        count += int(vals.shape[0])
    mean = total / max(1, count)
    var = (sq / max(1, count)) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt().clamp_min(0.05)
    return mean.detach(), std.detach()


@torch.no_grad()
def mean_baseline(loader: DataLoader | None, criterion: MaskedMelLoss, mel_mean: torch.Tensor, device: torch.device) -> float | None:
    if loader is None:
        return None
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="mean-baseline", leave=False):
        batch = batch_to_device(batch, device)
        pred = mel_mean.to(device).view(1, 1, -1).expand_as(batch["mel"])
        loss = criterion(pred, batch["mel"], batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def build_model(device: torch.device, args) -> R2CNNFiLMModel:
    model = R2CNNFiLMModel(
        dim=args.dim,
        spatial_tokens=args.spatial_tokens,
        num_points=args.num_landmark_points,
        dropout=args.dropout,
        upsample_mode=getattr(args, "upsample_mode", "conv_transpose"),
        decoder_channels=getattr(args, "decoder_channels", None),
        decoder_layers=getattr(args, "decoder_layers", 8),
        decoder_kernel_size=getattr(args, "decoder_kernel_size", 5),
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", True):
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def state_dict(model):
    return unwrap_model(model).state_dict()


def init_decoder_output_bias(model, mel_mean: torch.Tensor) -> None:
    decoder = getattr(unwrap_model(model), "decoder", None)
    if hasattr(decoder, "init_output_bias"):
        decoder.init_output_bias(mel_mean)


def save_checkpoint(path, model, optimizer, epoch, best, args, mel_mean, mel_std):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "best": float(best),
            "config": vars(args),
            "mel_mean": mel_mean.detach().cpu(),
            "mel_std": mel_std.detach().cpu(),
        },
        path,
    )


def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target = unwrap_model(model)
    missing, unexpected = target.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[checkpoint] loaded={path} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[checkpoint] missing:", missing[:12])
    if unexpected:
        print("[checkpoint] unexpected:", unexpected[:12])
    return ckpt


def masked_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    mask = mask.bool()
    p = pred.detach().float()[mask]
    t = target.detach().float()[mask]
    if p.numel() == 0:
        return {}

    def delta_abs(x):
        if x.shape[0] < 2:
            return torch.tensor(0.0)
        return (x[1:] - x[:-1]).abs().mean()

    p_energy = torch.logsumexp(p, dim=-1)
    t_energy = torch.logsumexp(t, dim=-1)
    return {
        "pred_std": float(p.std(unbiased=False).cpu()),
        "target_std": float(t.std(unbiased=False).cpu()),
        "std_ratio": float((p.std(unbiased=False) / t.std(unbiased=False).clamp_min(1e-6)).cpu()),
        "pred_delta": float(delta_abs(p).cpu()),
        "target_delta": float(delta_abs(t).cpu()),
        "delta_ratio": float((delta_abs(p) / delta_abs(t).clamp_min(1e-6)).cpu()),
        "energy_ratio": float((p_energy.std(unbiased=False) / t_energy.std(unbiased=False).clamp_min(1e-6)).cpu()),
    }


def write_history(path, history, args):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "config": vars(args)}, f, ensure_ascii=False, indent=2)


def target_mel_stats(mel: torch.Tensor, mel_mask: torch.Tensor, criterion: MaskedMelLoss) -> torch.Tensor:
    mel_norm = criterion._normalize(mel.float())
    mask = mel_mask.to(mel.device, dtype=mel_norm.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    mean = (mel_norm * mask).sum(dim=1) / denom
    centered = (mel_norm - mean.unsqueeze(1)) * mask
    std = (centered.pow(2).sum(dim=1) / denom).sqrt().clamp_min(1e-4)
    return torch.cat([mean, std], dim=-1)


def sanitize_batch(batch: dict) -> dict:
    for key in ("video", "landmarks", "mel", "video_times", "mel_times"):
        if key in batch and torch.is_tensor(batch[key]) and not torch.isfinite(batch[key]).all():
            paths = batch.get("paths", [])
            print(f"[warn] non-finite {key}; paths={paths[:4]}")
            batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=0.0, neginf=0.0)
    return batch

