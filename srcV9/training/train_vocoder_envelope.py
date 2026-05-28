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
from srcV9.models import LandmarkEnvelopeVocoderModel
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
        mel = batch["target_mel"].to(device).float()
        mask = batch["mel_mask"].to(device).bool()
        vals = mel[mask]
        total = vals.sum(dim=0) if total is None else total + vals.sum(dim=0)
        sq = vals.pow(2).sum(dim=0) if sq is None else sq + vals.pow(2).sum(dim=0)
        count += int(vals.shape[0])
    if total is None or sq is None:
        raise RuntimeError("Could not compute mel stats from empty loader.")
    mean = total / max(1, count)
    var = (sq / max(1, count)) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt().clamp_min(0.05)
    return mean.detach().cpu(), std.detach().cpu()


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target).abs() * mask_f).sum() / mask_f.sum().clamp_min(1.0) / pred.shape[-1]


def delta(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1] if x.shape[1] > 1 else torch.zeros_like(x[:, :0])


def envelope_loss(out: dict[str, torch.Tensor], batch: dict, mel_mean: torch.Tensor, mel_std: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, float]]:
    pred_env = out.get("envelope", out["mel"]).float()
    pred_final = out["mel"].float()
    target_env = batch["target_mel"].float()
    target_final = batch["mel"].float()
    mask = batch["mel_mask"].bool()
    mean = mel_mean.to(pred_env.device).view(1, 1, -1)
    std = mel_std.to(pred_env.device).view(1, 1, -1)
    pred_env_n = (pred_env - mean) / std
    target_env_n = (target_env - mean) / std
    pred_final_n = (pred_final - mean) / std
    target_final_n = (target_final - mean) / std

    envelope = masked_l1(pred_env_n, target_env_n, mask)
    loss = envelope
    env_delta = pred_env.new_tensor(0.0)
    final_mel = pred_env.new_tensor(0.0)
    final_delta = pred_env.new_tensor(0.0)
    residual = pred_env.new_tensor(0.0)
    e_loss = pred_env.new_tensor(0.0)

    if args.lambda_delta > 0 and pred_env.shape[1] > 1:
        d_mask = mask[:, 1:] & mask[:, :-1]
        env_delta = masked_l1(delta(pred_env_n), delta(target_env_n), d_mask)
        loss = loss + float(args.lambda_delta) * env_delta
    if args.lambda_final_mel > 0:
        final_mel = masked_l1(pred_final_n, target_final_n, mask)
        loss = loss + float(args.lambda_final_mel) * final_mel
    if args.lambda_final_delta > 0 and pred_final.shape[1] > 1:
        d_mask = mask[:, 1:] & mask[:, :-1]
        final_delta = masked_l1(delta(pred_final_n), delta(target_final_n), d_mask)
        loss = loss + float(args.lambda_final_delta) * final_delta
    if args.lambda_residual > 0:
        residual = masked_l1(pred_final_n - pred_env_n, target_final_n - target_env_n, mask)
        loss = loss + float(args.lambda_residual) * residual
    if args.lambda_energy > 0:
        pred_e = pred_final.mean(dim=-1, keepdim=True)
        target_e = target_final.mean(dim=-1, keepdim=True)
        e_loss = masked_l1(pred_e, target_e, mask)
        loss = loss + float(args.lambda_energy) * e_loss
    return loss, {
        "envelope": float(envelope.detach().cpu()),
        "env_delta": float(env_delta.detach().cpu()),
        "final_mel": float(final_mel.detach().cpu()),
        "final_delta": float(final_delta.detach().cpu()),
        "residual": float(residual.detach().cpu()),
        "energy": float(e_loss.detach().cpu()),
    }


@torch.no_grad()
def masked_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    m = mask.bool()
    pv = pred[m]
    tv = target[m]
    if pv.numel() == 0 or tv.numel() == 0:
        return {"std_ratio": 0.0, "delta_ratio": 0.0}
    pred_std = float(pv.std(unbiased=False).detach().cpu())
    target_std = float(tv.std(unbiased=False).detach().cpu())
    if pred.shape[1] > 1:
        d_mask = m[:, 1:] & m[:, :-1]
        pd = delta(pred)[d_mask]
        td = delta(target)[d_mask]
        pred_delta = float(pd.abs().mean().detach().cpu()) if pd.numel() else 0.0
        target_delta = float(td.abs().mean().detach().cpu()) if td.numel() else 0.0
    else:
        pred_delta = target_delta = 0.0
    return {
        "std_ratio": pred_std / max(target_std, 1e-8),
        "delta_ratio": pred_delta / max(target_delta, 1e-8),
        "pred_std": pred_std,
        "target_std": target_std,
        "pred_delta": pred_delta,
        "target_delta": target_delta,
    }


def build_model(args, device: torch.device, mel_mean: torch.Tensor) -> torch.nn.Module:
    model = LandmarkEnvelopeVocoderModel(
        num_points=args.num_landmark_points,
        dim=args.dim,
        n_mels=args.n_mels,
        tcn_layers=args.tcn_layers,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        output_bias_init=float(mel_mean.mean().item()),
        residual_alpha_init=args.residual_alpha_init,
        enable_residual=not args.disable_residual,
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
        },
        path,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, mel_mean, mel_std, args, epoch: int) -> dict[str, float]:
    model.train()
    if hasattr(loader.dataset, "resample_windows"):
        loader.dataset.resample_windows(epoch)
    amp_enabled = device.type == "cuda" and args.amp
    total = 0.0
    parts = {
        "envelope": 0.0,
        "env_delta": 0.0,
        "final_mel": 0.0,
        "final_delta": 0.0,
        "residual": 0.0,
        "energy": 0.0,
    }
    count = 0
    for batch in tqdm(loader, desc="train-env", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = model(batch, target_len=batch["target_mel"].shape[1])
        with torch.amp.autocast("cuda", enabled=False):
            loss, loss_parts = envelope_loss(out, batch, mel_mean, mel_std, args)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite envelope loss at paths={batch.get('paths', [])[:4]}")
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
    out = {"loss": total / max(1, count)}
    out.update({key: val / max(1, count) for key, val in parts.items()})
    return out


@torch.no_grad()
def evaluate(model, loader, device, mel_mean, mel_std, args) -> dict:
    model.eval()
    total = 0.0
    count = 0
    stats = {}
    for batch in tqdm(loader, desc="eval-env", leave=False):
        batch = batch_to_device(batch, device)
        out = model(batch, target_len=batch["target_mel"].shape[1])
        loss, _ = envelope_loss(out, batch, mel_mean, mel_std, args)
        total += float(loss.detach().cpu())
        count += 1
        if not stats:
            stats = masked_stats(out["mel"].float(), batch["mel"].float(), batch["mel_mask"])
            stats.update(
                {
                    "env_std_ratio": masked_stats(
                        out.get("envelope", out["mel"]).float(),
                        batch["target_mel"].float(),
                        batch["mel_mask"],
                    ).get("std_ratio", 0.0),
                    "residual_alpha": float(out.get("residual_alpha", torch.tensor(0.0)).detach().float().mean().cpu()),
                }
            )
    return {"loss": total / max(1, count), "stats": stats}


def mean_baseline(loader, device, mel_mean, mel_std, args) -> float:
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="mean-baseline", leave=False):
        batch = batch_to_device(batch, device)
        pred = mel_mean.to(device).view(1, 1, -1).expand_as(batch["target_mel"])
        out = {"mel": pred, "envelope": pred, "energy": pred.mean(dim=-1, keepdim=True)}
        loss, _ = envelope_loss(out, batch, mel_mean, mel_std, args)
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
    print(f"[window] frames={args.window_frames} hop={args.hop_frames} smooth_target={args.smooth_target_frames}")
    print(
        f"[model] srcV9 landmark_envelope_residual_vocoder dim={args.dim} n_mels={args.n_mels} "
        f"residual={'off' if args.disable_residual else 'on'} alpha_init={args.residual_alpha_init:.3f}"
    )
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
        row = {"epoch": epoch, "train": train, "train_eval": train_eval, "val_eval": val_eval, "best": best}
        history.append(row)
        write_json(output_dir / "history.json", {"history": history, "config": vars(args)})
        val_txt = f"{float(val_eval['loss']):.6f}" if val_eval is not None else "n/a"
        stats = (val_eval or train_eval)["stats"]
        print(
            f"[epoch {epoch:04d}] train={train['loss']:.6f} env={train['envelope']:.6f} "
            f"final={train['final_mel']:.6f} res={train['residual']:.6f} "
            f"train_eval={train_eval['loss']:.6f} val={val_txt} best={best:.6f} "
            f"std_r={stats.get('std_ratio', 0.0):.3f} del_r={stats.get('delta_ratio', 0.0):.3f} "
            f"alpha={stats.get('residual_alpha', 0.0):.3f}"
            f"{' best' if is_best else ''}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train srcV9 landmark-to-envelope classical vocoder model.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--output-dir", default="checkpoints_srcV9_envelope_vocoder")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=45)
    parser.add_argument("--hop-frames", type=int, default=15)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    parser.add_argument("--random-windows-per-file", type=int, default=0)
    parser.add_argument("--smooth-target-frames", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--tcn-layers", type=int, default=6)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.2)
    parser.add_argument("--lambda-final-mel", type=float, default=0.35)
    parser.add_argument("--lambda-final-delta", type=float, default=0.15)
    parser.add_argument("--lambda-residual", type=float, default=0.15)
    parser.add_argument("--lambda-energy", type=float, default=0.05)
    parser.add_argument("--residual-alpha-init", type=float, default=0.25)
    parser.add_argument("--disable-residual", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-last", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
