from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV9.data import LandmarkEnvelopeDataset, collate_envelope, split_cache_files
from srcV9.models import LandmarkSourceFilterVocoderModel
from srcV9.training.train_vocoder_envelope import delta, masked_l1, masked_stats
from srcV9.utils import batch_to_device, get_device, seed_everything, unwrap_model, write_json


def make_loader(args, files, batch_size: int, shuffle: bool, random_windows: int = 0) -> DataLoader:
    ds = LandmarkEnvelopeDataset(
        args.data_dir,
        files=files,
        window_frames=args.window_frames,
        hop_frames=args.hop_frames,
        max_windows_per_file=args.max_windows_per_file,
        random_windows_per_file=random_windows,
        smooth_target_frames=args.smooth_target_frames,
        seed=args.seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_envelope,
        drop_last=args.drop_last,
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


def source_filter_loss(out: dict[str, torch.Tensor], batch: dict, mel_mean: torch.Tensor, mel_std: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, float]]:
    pred_env = out["envelope"].float()
    pred_source = out["source"].float()
    pred_final = out["mel"].float()
    target_env = batch["target_mel"].float()
    target_final = batch["mel"].float()
    target_source = target_final - target_env
    mask = batch["mel_mask"].bool()
    mean = mel_mean.to(pred_final.device).view(1, 1, -1)
    std = mel_std.to(pred_final.device).view(1, 1, -1)
    pred_env_n = (pred_env - mean) / std
    target_env_n = (target_env - mean) / std
    pred_final_n = (pred_final - mean) / std
    target_final_n = (target_final - mean) / std
    pred_source_n = pred_source / std
    target_source_n = target_source / std

    env = masked_l1(pred_env_n, target_env_n, mask)
    source = masked_l1(pred_source_n, target_source_n, mask)
    final = masked_l1(pred_final_n, target_final_n, mask)
    loss = args.lambda_env * env + args.lambda_source * source + args.lambda_final * final
    final_delta = pred_final.new_tensor(0.0)
    source_delta = pred_final.new_tensor(0.0)
    energy = pred_final.new_tensor(0.0)
    if args.lambda_final_delta > 0 and pred_final.shape[1] > 1:
        d_mask = mask[:, 1:] & mask[:, :-1]
        final_delta = masked_l1(delta(pred_final_n), delta(target_final_n), d_mask)
        loss = loss + args.lambda_final_delta * final_delta
    if args.lambda_source_delta > 0 and pred_source.shape[1] > 1:
        d_mask = mask[:, 1:] & mask[:, :-1]
        source_delta = masked_l1(delta(pred_source_n), delta(target_source_n), d_mask)
        loss = loss + args.lambda_source_delta * source_delta
    if args.lambda_energy > 0:
        pred_e = pred_final.mean(dim=-1, keepdim=True)
        target_e = target_final.mean(dim=-1, keepdim=True)
        energy = masked_l1(pred_e, target_e, mask)
        loss = loss + args.lambda_energy * energy
    return loss, {
        "env": float(env.detach().cpu()),
        "source": float(source.detach().cpu()),
        "final": float(final.detach().cpu()),
        "final_delta": float(final_delta.detach().cpu()),
        "source_delta": float(source_delta.detach().cpu()),
        "energy": float(energy.detach().cpu()),
    }


def build_model(args, device: torch.device, mel_mean: torch.Tensor) -> torch.nn.Module:
    model = LandmarkSourceFilterVocoderModel(
        num_points=args.num_landmark_points,
        dim=args.dim,
        n_mels=args.n_mels,
        source_bands=args.source_bands,
        tcn_layers=args.tcn_layers,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        output_bias_init=float(mel_mean.mean().item()),
        source_scale_init=args.source_scale_init,
    ).to(device)
    unwrap_model(model).set_output_bias(mel_mean.to(device))
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu and args.batch_size >= torch.cuda.device_count():
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def save_checkpoint(path, model, optimizer, epoch, best, args, mel_mean, mel_std):
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
            "model_type": "source_filter_vocoder",
        },
        path,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, mel_mean, mel_std, args, epoch: int) -> dict[str, float]:
    model.train()
    if hasattr(loader.dataset, "resample_windows"):
        loader.dataset.resample_windows(epoch)
    amp_enabled = device.type == "cuda" and args.amp
    total = 0.0
    parts = {key: 0.0 for key in ("env", "source", "final", "final_delta", "source_delta", "energy")}
    count = 0
    for batch in tqdm(loader, desc="train-sf", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = model(batch, target_len=batch["mel"].shape[1])
        with torch.amp.autocast("cuda", enabled=False):
            loss, loss_parts = source_filter_loss(out, batch, mel_mean, mel_std, args)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite source-filter loss at paths={batch.get('paths', [])[:4]}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        for key in parts:
            parts[key] += float(loss_parts[key])
        count += 1
    row = {"loss": total / max(1, count)}
    row.update({key: value / max(1, count) for key, value in parts.items()})
    return row


@torch.no_grad()
def evaluate(model, loader, device, mel_mean, mel_std, args) -> dict:
    model.eval()
    total = 0.0
    count = 0
    stats = {}
    for batch in tqdm(loader, desc="eval-sf", leave=False):
        batch = batch_to_device(batch, device)
        out = model(batch, target_len=batch["mel"].shape[1])
        loss, _ = source_filter_loss(out, batch, mel_mean, mel_std, args)
        total += float(loss.detach().cpu())
        count += 1
        if not stats:
            stats = masked_stats(out["mel"].float(), batch["mel"].float(), batch["mel_mask"])
            stats["env_std_ratio"] = masked_stats(out["envelope"].float(), batch["target_mel"].float(), batch["mel_mask"]).get("std_ratio", 0.0)
            stats["source_std"] = float(out["source"][batch["mel_mask"].bool()].std(unbiased=False).detach().cpu())
            stats["source_scale"] = float(out["source_scale"].detach().float().mean().cpu())
            stats["source_gate"] = float(out["source_gate"][batch["mel_mask"].bool()].mean().detach().cpu())
    return {"loss": total / max(1, count), "stats": stats}


def mean_baseline(loader, device, mel_mean, mel_std, args) -> float:
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="mean-baseline", leave=False):
        batch = batch_to_device(batch, device)
        pred = mel_mean.to(device).view(1, 1, -1).expand_as(batch["mel"])
        out = {
            "mel": pred,
            "envelope": pred,
            "source": torch.zeros_like(pred),
            "source_scale": torch.tensor(0.0, device=device),
            "source_gate": torch.zeros(pred.shape[:2] + (1,), device=device),
        }
        loss, _ = source_filter_loss(out, batch, mel_mean, mel_std, args)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def run(args) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, limit_files=limit_files)
    train_loader = make_loader(args, train_files, args.batch_size, shuffle=True, random_windows=args.random_windows_per_file)
    stats_loader = make_loader(args, train_files, args.batch_size, shuffle=False, random_windows=0)
    val_loader = make_loader(args, val_files, args.val_batch_size or args.batch_size, shuffle=False, random_windows=0) if val_files else None
    mel_mean, mel_std = compute_mel_stats(stats_loader, device)
    model = build_model(args, device, mel_mean)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    mean_train = mean_baseline(stats_loader, device, mel_mean, mel_std, args)
    mean_val = mean_baseline(val_loader, device, mel_mean, mel_std, args) if val_loader is not None else None
    print(f"[device] {device}")
    print(f"[data] files train={len(train_files)} val={len(val_files)} windows train={len(train_loader.dataset)}")
    print(f"[model] srcV9 source_filter dim={args.dim} source_bands={args.source_bands} scale_init={args.source_scale_init:.3f}")
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={mean_val if mean_val is not None else 'n/a'}")
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = train_one_epoch(model, train_loader, optimizer, scaler, device, mel_mean, mel_std, args, epoch)
        train_eval = evaluate(model, stats_loader, device, mel_mean, mel_std, args)
        val_eval = evaluate(model, val_loader, device, mel_mean, mel_std, args) if val_loader is not None else None
        score = float((val_eval or train_eval)["loss"])
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        history.append({"epoch": epoch, "train": train, "train_eval": train_eval, "val_eval": val_eval, "best": best})
        write_json(output_dir / "history.json", {"history": history, "config": vars(args)})
        stats = (val_eval or train_eval)["stats"]
        val_txt = f"{float(val_eval['loss']):.6f}" if val_eval is not None else "n/a"
        print(
            f"[epoch {epoch:04d}] train={train['loss']:.6f} env={train['env']:.4f} src={train['source']:.4f} "
            f"final={train['final']:.4f} train_eval={train_eval['loss']:.6f} val={val_txt} best={best:.6f} "
            f"std_r={stats.get('std_ratio', 0.0):.3f} del_r={stats.get('delta_ratio', 0.0):.3f} "
            f"src_std={stats.get('source_std', 0.0):.3f} scale={stats.get('source_scale', 0.0):.3f} "
            f"gate={stats.get('source_gate', 0.0):.3f}{' best' if is_best else ''}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-filter log-mel vocoder target, then synthesize with Griffin-Lim.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--output-dir", default="checkpoints_srcV9_source_filter")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=45)
    parser.add_argument("--hop-frames", type=int, default=15)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    parser.add_argument("--random-windows-per-file", type=int, default=0)
    parser.add_argument("--smooth-target-frames", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--source-bands", type=int, default=16)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--tcn-layers", type=int, default=6)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-env", type=float, default=0.7)
    parser.add_argument("--lambda-source", type=float, default=0.8)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--lambda-final-delta", type=float, default=0.35)
    parser.add_argument("--lambda-source-delta", type=float, default=0.15)
    parser.add_argument("--lambda-energy", type=float, default=0.05)
    parser.add_argument("--source-scale-init", type=float, default=0.6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-last", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
