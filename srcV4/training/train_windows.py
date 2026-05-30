from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV4.data import WindowedMelDataset, collate_windows, split_cache_files
from srcV4.models import V4MelLoss, V4SpeechModel, masked_stats
from srcV4.utils import batch_to_device, get_device, seed_everything, unwrap_model, write_json


def parse_layers(value: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("resnet layers must look like 1,1,1,1 or 2,2,2,2")
    return tuple(parts)  # type: ignore[return-value]


def make_loader(
    data_dir: str | Path,
    files: list[Path],
    batch_size: int,
    window_frames: int,
    hop_frames: int,
    max_windows_per_file: int,
    random_windows_per_file: int,
    seed: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    dataset = WindowedMelDataset(
        data_dir,
        files=files,
        window_frames=window_frames,
        hop_frames=hop_frames,
        max_windows_per_file=max_windows_per_file,
        random_windows_per_file=random_windows_per_file,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_windows,
        drop_last=drop_last,
    )


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
    if total is None or sq is None:
        raise RuntimeError("Could not compute mel stats from an empty loader.")
    mean = total / max(1, count)
    var = (sq / max(1, count)) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt().clamp_min(0.05)
    return mean.detach().cpu(), std.detach().cpu()


@torch.no_grad()
def mean_baseline(loader: DataLoader, criterion: V4MelLoss, mel_mean: torch.Tensor, device: torch.device) -> float:
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="mean-baseline", leave=False):
        batch = batch_to_device(batch, device)
        pred = mel_mean.to(device).view(1, 1, -1).expand_as(batch["mel"])
        loss = criterion(pred, batch["mel"], batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def build_model(args: argparse.Namespace, device: torch.device, mel_mean: torch.Tensor | None = None) -> torch.nn.Module:
    output_bias = float(mel_mean.mean().item()) if mel_mean is not None else -4.0
    model = V4SpeechModel(
        dim=args.dim,
        num_landmark_points=args.num_landmark_points,
        fusion_type=args.fusion_type,
        encoder_width=args.encoder_width,
        resnet_layers=args.resnet_layers,
        visual_temporal_layers=args.visual_temporal_layers,
        landmark_temporal_layers=args.landmark_temporal_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        output_bias_init=output_bias,
        use_snn=args.use_snn,
        snn_layers=args.snn_layers,
        snn_tau=args.snn_tau,
        siren_layers=args.siren_layers,
        siren_omega=args.siren_omega,
        visual_encoder_type=args.visual_encoder_type,
    ).to(device)
    
    if mel_mean is not None:
        unwrap_model(model).set_output_bias(mel_mean.to(device))
        
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best: float,
    args: argparse.Namespace,
    mel_mean: torch.Tensor,
    mel_std: torch.Tensor,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best": float(best),
            "config": vars(args),
            "mel_mean": mel_mean.detach().cpu(),
            "mel_std": mel_std.detach().cpu(),
        },
        path,
    )


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    raw = unwrap_model(model)
    visual_params = list(raw.visual.parameters())
    landmark_params = list(raw.landmarks.parameters())
    fusion_params = list(raw.fusion.parameters())
    decoder_params = list(raw.decoder.parameters())
    siren_params = list(raw.siren_residual.parameters())
    
    param_groups = [
        {"params": visual_params, "lr": args.visual_lr or args.lr * 0.3},
        {"params": landmark_params, "lr": args.landmark_lr or args.lr},
        {"params": fusion_params, "lr": args.fusion_lr or args.lr},
        {"params": decoder_params, "lr": args.decoder_lr or args.lr * 1.5},
        {"params": siren_params, "lr": args.siren_lr or args.lr * 0.5},
    ]
    
    if args.use_snn:
        snn_params = list(raw.snn.parameters())
        param_groups.append({"params": snn_params, "lr": args.snn_lr or args.lr * 0.3})
        
    return torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )


def get_lr_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int = 5) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def sanitize_batch(batch: dict) -> dict:
    for key in ("video", "landmarks", "mel", "video_times", "mel_times"):
        val = batch.get(key)
        if torch.is_tensor(val) and not torch.isfinite(val).all():
            batch[key] = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return batch


def augment_batch(batch: dict, jitter_scale: float = 0.003, time_mask_prob: float = 0.3, freq_mask_prob: float = 0.3) -> dict:
    # Landmark jittering
    if jitter_scale > 0 and random.random() < 0.5:
        landmarks = batch["landmarks"]
        noise = torch.randn_like(landmarks) * jitter_scale
        batch["landmarks"] = landmarks + noise
        
    # SpecAugment-style time and frequency masking on mel
    mel = batch["mel"]
    B, T, C = mel.shape
    
    if random.random() < time_mask_prob:
        for b in range(B):
            t_len = int(batch["mel_lengths"][b].item())
            if t_len > 10:
                width = random.randint(2, min(8, t_len // 5))
                start = random.randint(0, t_len - width)
                mel[b, start:start+width, :] = mel[b].mean()
                
    if random.random() < freq_mask_prob:
        for b in range(B):
            width = random.randint(4, 12)
            start = random.randint(0, C - width - 1)
            mel[b, :, start:start+width] = mel[b].mean()
            
    batch["mel"] = mel
    return batch


def set_visual_trainable(model: torch.nn.Module, trainable: bool) -> None:
    raw = unwrap_model(model)
    for param in raw.visual.parameters():
        param.requires_grad = trainable
    if trainable:
        raw.visual.train()
    else:
        raw.visual.eval()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, args, epoch: int) -> float:
    model.train()
    if hasattr(loader.dataset, "resample_windows"):
        loader.dataset.resample_windows(epoch)
        
    freeze_visual = epoch <= args.freeze_visual_epochs
    set_visual_trainable(model, not freeze_visual)
    
    amp_enabled = device.type == "cuda" and args.amp
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = sanitize_batch(batch_to_device(batch, device))
        
        # Apply data augmentations in training
        if args.augment:
            batch = augment_batch(
                batch,
                jitter_scale=args.jitter_scale,
                time_mask_prob=args.time_mask_prob,
                freq_mask_prob=args.freq_mask_prob
            )
            
        optimizer.zero_grad(set_to_none=True)
        if freeze_visual:
            unwrap_model(model).visual.eval()
            
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(batch, target_len=batch["mel"].shape[1])
            
        with torch.amp.autocast("cuda", enabled=False):
            loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
            
        if not torch.isfinite(loss):
            paths = batch.get("paths", [])[:4]
            raise FloatingPointError(f"Non-finite train loss at paths={paths}")
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
        scaler.step(optimizer)
        scaler.update()
        
        total += float(loss.detach().cpu())
        count += 1
        
    scheduler.step()
    return total / max(1, count)


@torch.no_grad()
def evaluate(model, loader, criterion, device, args) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    count = 0
    stats = {}
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = sanitize_batch(batch_to_device(batch, device))
        pred = model(batch, target_len=batch["mel"].shape[1])
        pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        loss = criterion(pred, batch["mel"].float(), batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
        if not stats:
            stats = masked_stats(pred, batch["mel"], batch["mel_mask"])
    return total / max(1, count), stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train srcV4 Lip-to-Speech Reconstruction Model.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2")
    parser.add_argument("--output-dir", default="checkpoints_srcV4_lrs2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--hop-frames", type=int, default=10)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    parser.add_argument("--random-windows-per-file", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--encoder-width", type=int, default=32)
    parser.add_argument("--resnet-layers", type=parse_layers, default=(1, 1, 1, 1))
    parser.add_argument("--visual-temporal-layers", type=int, default=1)
    parser.add_argument("--landmark-temporal-layers", type=int, default=1)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--fusion-type", choices=["landmark_first", "concat", "gated", "visual_only", "landmark_only"], default="landmark_first")
    parser.add_argument("--dropout", type=float, default=0.15)
    
    # Learning rates
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--visual-lr", type=float, default=0.0)
    parser.add_argument("--landmark-lr", type=float, default=0.0)
    parser.add_argument("--fusion-lr", type=float, default=0.0)
    parser.add_argument("--decoder-lr", type=float, default=0.0)
    parser.add_argument("--siren-lr", type=float, default=0.0)
    parser.add_argument("--snn-lr", type=float, default=0.0)
    
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-visual-epochs", type=int, default=0)
    
    # Loss weight multipliers
    parser.add_argument("--lambda-mel", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.15)
    parser.add_argument("--lambda-delta2", type=float, default=0.03)
    parser.add_argument("--lambda-energy", type=float, default=0.02)
    parser.add_argument("--lambda-mr-spectral", type=float, default=0.25)
    parser.add_argument("--shift-window", type=int, default=0)
    
    # SNN arguments
    parser.add_argument("--use-snn", action="store_true", help="Enable LIF Spiking Neural Network temporal processor.")
    parser.add_argument("--snn-layers", type=int, default=2)
    parser.add_argument("--snn-tau", type=float, default=2.0)
    
    # SIREN arguments
    parser.add_argument("--siren-layers", type=int, default=2)
    parser.add_argument("--siren-omega", type=float, default=20.0)
    
    # Visual encoder arguments
    parser.add_argument("--visual-encoder-type", choices=["r2plus1d", "av_hubert"], default="r2plus1d")
    
    # Data augmentation settings
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jitter-scale", type=float, default=0.003)
    parser.add_argument("--time-mask-prob", type=float, default=0.3)
    parser.add_argument("--freq-mask-prob", type=float, default=0.3)
    
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-last", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, limit_files=limit_files)
    
    train_loader = make_loader(
        args.data_dir,
        train_files,
        args.batch_size,
        args.window_frames,
        args.hop_frames,
        args.max_windows_per_file,
        args.random_windows_per_file,
        args.seed,
        args.num_workers,
        shuffle=True,
        drop_last=args.drop_last,
    )
    val_loader = None
    if val_files:
        val_loader = make_loader(
            args.data_dir,
            val_files,
            args.val_batch_size or args.batch_size,
            args.window_frames,
            args.hop_frames,
            args.max_windows_per_file,
            0,
            args.seed + 1,
            args.num_workers,
            shuffle=False,
        )

    stats_loader = make_loader(
        args.data_dir,
        train_files,
        args.batch_size,
        args.window_frames,
        args.hop_frames,
        args.max_windows_per_file,
        0,
        args.seed,
        args.num_workers,
        shuffle=False,
    )
    
    mel_mean, mel_std = compute_mel_stats(stats_loader, device)
    
    criterion = V4MelLoss(
        mel_mean,
        mel_std,
        lambda_mel=args.lambda_mel,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
        lambda_mr_spectral=args.lambda_mr_spectral,
        shift_window=args.shift_window,
    ).to(device)
    
    model = build_model(args, device, mel_mean=mel_mean)
    optimizer = make_optimizer(model, args)
    scheduler = get_lr_scheduler(optimizer, args.epochs, warmup_epochs=5)
    
    amp_enabled = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    mean_train = mean_baseline(stats_loader, criterion, mel_mean, device)
    mean_val = mean_baseline(val_loader, criterion, mel_mean, device) if val_loader is not None else None
    
    print(f"[device] {device}")
    print(f"[data] files train={len(train_files)} val={len(val_files)} windows train={len(train_loader.dataset)}")
    print(f"[window] frames={args.window_frames} hop={args.hop_frames}")
    print(f"[model] visual={args.visual_encoder_type} decoder=TFiLM-Conformer SIREN=Residual SNN={'Enabled' if args.use_snn else 'Disabled'}")
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={mean_val if mean_val is not None else 'n/a'}")

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[1]["lr"]  # Get landmark LR as reference
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, args, epoch)
        train_eval, train_stats = evaluate(model, stats_loader, criterion, device, args)
        
        val_loss = None
        val_stats = {}
        score = train_eval
        if val_loader is not None:
            val_loss, val_stats = evaluate(model, val_loader, criterion, device, args)
            score = val_loss
            
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
            
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        
        row = {
            "epoch": epoch,
            "train": train_loss,
            "train_eval": train_eval,
            "val": val_loss,
            "best": best,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        history.append(row)
        write_json(output_dir / "history.json", {"history": history, "config": vars(args)})
        
        val_txt = f"{val_loss:.6f}" if val_loss is not None else "n/a"
        std_r = train_stats.get("std_ratio", 0.0)
        del_r = train_stats.get("delta_ratio", 0.0)
        tag = " best" if is_best else ""
        print(
            f"[epoch {epoch:04d}] lr={current_lr:.6f} train={train_loss:.6f} train_eval={train_eval:.6f} "
            f"val={val_txt} best={best:.6f} std_r={std_r:.3f} del_r={del_r:.3f}{tag}"
        )


if __name__ == "__main__":
    run(parse_args())
