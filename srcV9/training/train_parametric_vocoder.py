from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from srcV9.data import split_cache_files
from srcV9.models import VideoParametricVocoderModel
from srcV9.training.train_source_filter_vocoder import (
    compute_mel_stats,
    make_loader,
    masked_l1,
)
from srcV9.training.train_vocoder_envelope import delta, masked_stats
from srcV9.utils import batch_to_device, get_device, seed_everything, unwrap_model, write_json


def parse_layers(value: str) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(x) for x in value)  # type: ignore[return-value]
    parts = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("visual layers must look like 1,1,1,1")
    return tuple(parts)  # type: ignore[return-value]


def _bands_from_mel(x: torch.Tensor, bands: int) -> torch.Tensor:
    batch, frames, mel_bins = x.shape
    return F.interpolate(
        x.reshape(batch * frames, 1, mel_bins),
        size=int(bands),
        mode="linear",
        align_corners=False,
    ).reshape(batch, frames, int(bands))


def _bands_to_mel(x: torch.Tensor, n_mels: int) -> torch.Tensor:
    batch, frames, bands = x.shape
    y = F.interpolate(
        x.reshape(batch * frames, 1, bands),
        size=int(n_mels),
        mode="linear",
        align_corners=False,
    ).reshape(batch, frames, int(n_mels))
    return y - y.mean(dim=-1, keepdim=True)


def make_voicing_target(mel: torch.Tensor) -> torch.Tensor:
    low = mel[..., 5:34].mean(dim=-1, keepdim=True)
    high = mel[..., 48:].mean(dim=-1, keepdim=True)
    contrast = low - high
    return torch.sigmoid(1.4 * contrast).clamp(0.02, 0.98)


def compute_targets(batch: dict, n_mels: int, source_bands: int, model: VideoParametricVocoderModel) -> dict[str, torch.Tensor]:
    mel = batch["mel"].float()
    envelope = batch["target_mel"].float()
    energy = mel.mean(dim=-1, keepdim=True)
    filter_shape = envelope - envelope.mean(dim=-1, keepdim=True)
    voicing = make_voicing_target(mel)
    template = voicing * model.voiced_template.to(mel.device, mel.dtype) + (1.0 - voicing) * model.unvoiced_template.to(mel.device, mel.dtype)
    residual = mel - energy - filter_shape - template
    broad_bands = _bands_from_mel(residual, source_bands)
    broad_mel = _bands_to_mel(broad_bands, n_mels)
    return {
        "filter": filter_shape,
        "energy": energy,
        "voicing": voicing,
        "broad_bands": broad_bands,
        "broad_mel": broad_mel,
        "final": mel,
    }


def parametric_loss(out: dict[str, torch.Tensor], batch: dict, mel_mean: torch.Tensor, mel_std: torch.Tensor, args, model) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mel_mask"].bool()
    targets = compute_targets(batch, args.n_mels, args.source_bands, unwrap_model(model))
    mean = mel_mean.to(out["mel"].device).view(1, 1, -1)
    std = mel_std.to(out["mel"].device).view(1, 1, -1)
    pred_final_n = (out["mel"].float() - mean) / std
    target_final_n = (targets["final"] - mean) / std

    filter_loss = masked_l1(out["filter"].float(), targets["filter"], mask)
    energy_loss = masked_l1(out["energy"].float(), targets["energy"], mask)
    broad_target = _bands_to_mel(targets["broad_bands"], args.n_mels)
    broad_loss = masked_l1(out["broad_source"].float(), broad_target, mask)
    final_loss = masked_l1(pred_final_n, target_final_n, mask)
    voicing_loss = F.binary_cross_entropy(out["voicing"].float().clamp(1e-4, 1.0 - 1e-4), targets["voicing"], reduction="none")
    voicing_loss = (voicing_loss * mask.unsqueeze(-1).to(voicing_loss.dtype)).sum() / mask.sum().clamp_min(1).to(voicing_loss.dtype)
    final_delta = out["mel"].new_tensor(0.0)
    if args.lambda_final_delta > 0 and out["mel"].shape[1] > 1:
        d_mask = mask[:, 1:] & mask[:, :-1]
        final_delta = masked_l1(delta(pred_final_n), delta(target_final_n), d_mask)
    loss = (
        args.lambda_filter * filter_loss
        + args.lambda_energy * energy_loss
        + args.lambda_voicing * voicing_loss
        + args.lambda_broad * broad_loss
        + args.lambda_final * final_loss
        + args.lambda_final_delta * final_delta
    )
    return loss, {
        "filter": float(filter_loss.detach().cpu()),
        "energy": float(energy_loss.detach().cpu()),
        "voicing": float(voicing_loss.detach().cpu()),
        "broad": float(broad_loss.detach().cpu()),
        "final": float(final_loss.detach().cpu()),
        "final_delta": float(final_delta.detach().cpu()),
    }


def build_model(args, device: torch.device, mel_mean: torch.Tensor) -> torch.nn.Module:
    model = VideoParametricVocoderModel(
        num_points=args.num_landmark_points,
        dim=args.dim,
        n_mels=args.n_mels,
        source_bands=args.source_bands,
        visual_width=args.visual_width,
        visual_layers=args.visual_layers,
        visual_temporal_layers=args.visual_temporal_layers,
        landmark_tcn_layers=args.landmark_tcn_layers,
        landmark_transformer_layers=args.landmark_transformer_layers,
        nhead=args.nhead,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        source_scale_init=args.source_scale_init,
    ).to(device)
    unwrap_model(model).set_energy_bias(mel_mean.to(device))
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu and args.batch_size >= torch.cuda.device_count():
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def make_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    raw = unwrap_model(model)
    return torch.optim.AdamW(
        [
            {"params": raw.visual.parameters(), "lr": args.visual_lr or args.lr},
            {"params": raw.landmarks.parameters(), "lr": args.landmark_lr or args.lr},
            {
                "params": list(raw.filter_in.parameters())
                + list(raw.source_in.parameters())
                + list(raw.filter_blocks.parameters())
                + list(raw.source_blocks.parameters())
                + list(raw.filter_norm.parameters())
                + list(raw.source_norm.parameters())
                + list(raw.filter_head.parameters())
                + list(raw.energy_head.parameters())
                + list(raw.voicing_head.parameters())
                + list(raw.bands_head.parameters())
                + [raw.source_scale_logit],
                "lr": args.lr,
            },
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )


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
            "model_type": "video_parametric_vocoder",
        },
        path,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, mel_mean, mel_std, args, epoch: int) -> dict[str, float]:
    model.train()
    if hasattr(loader.dataset, "resample_windows"):
        loader.dataset.resample_windows(epoch)
    amp_enabled = device.type == "cuda" and args.amp
    total = 0.0
    parts = {key: 0.0 for key in ("filter", "energy", "voicing", "broad", "final", "final_delta")}
    count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = model(batch, target_len=batch["mel"].shape[1])
        with torch.amp.autocast("cuda", enabled=False):
            loss, loss_parts = parametric_loss(out, batch, mel_mean, mel_std, args, model)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite parametric loss at paths={batch.get('paths', [])[:4]}")
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
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(batch, target_len=batch["mel"].shape[1])
        loss, _ = parametric_loss(out, batch, mel_mean, mel_std, args, model)
        total += float(loss.detach().cpu())
        count += 1
        if not stats:
            stats = masked_stats(out["mel"].float(), batch["mel"].float(), batch["mel_mask"])
            stats["source_std"] = float(out["source"][batch["mel_mask"].bool()].std(unbiased=False).detach().cpu())
            stats["voicing_mean"] = float(out["voicing"][batch["mel_mask"].bool()].mean().detach().cpu())
            stats["scale"] = float(out["source_scale"].detach().cpu())
    return {"loss": total / max(1, count), "stats": stats}


def mean_baseline_parametric(loader, device, mel_mean, mel_std, args, model) -> float:
    total = 0.0
    count = 0
    raw = unwrap_model(model)
    energy_value = float(mel_mean.float().mean().item())
    for batch in loader:
        batch = batch_to_device(batch, device)
        batch_size, frames, mel_bins = batch["mel"].shape
        energy = torch.full((batch_size, frames, 1), energy_value, device=device, dtype=batch["mel"].dtype)
        voicing = torch.full((batch_size, frames, 1), 0.5, device=device, dtype=batch["mel"].dtype)
        template = 0.5 * raw.voiced_template.to(device, batch["mel"].dtype) + 0.5 * raw.unvoiced_template.to(device, batch["mel"].dtype)
        source = template.expand(batch_size, frames, mel_bins)
        out = {
            "mel": energy + source,
            "filter": torch.zeros_like(batch["mel"]),
            "energy": energy,
            "voicing": voicing,
            "broad_source": torch.zeros_like(batch["mel"]),
            "source": source,
            "source_scale": torch.tensor(0.0, device=device),
        }
        loss, _ = parametric_loss(out, batch, mel_mean, mel_std, args, raw)
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
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    mean_train = mean_baseline_parametric(stats_loader, device, mel_mean, mel_std, args, model)
    mean_val = mean_baseline_parametric(val_loader, device, mel_mean, mel_std, args, model) if val_loader is not None else None
    print(f"[device] {device}")
    print(f"[data] files train={len(train_files)} val={len(val_files)} windows train={len(train_loader.dataset)}")
    print(f"[model] srcV9 parametric_vocoder dim={args.dim} source_bands={args.source_bands} scale_init={args.source_scale_init:.3f}")
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
            f"[epoch {epoch:04d}] train={train['loss']:.6f} filt={train['filter']:.4f} "
            f"eng={train['energy']:.4f} voi={train['voicing']:.4f} broad={train['broad']:.4f} "
            f"final={train['final']:.4f} train_eval={train_eval['loss']:.6f} val={val_txt} best={best:.6f} "
            f"std_r={stats.get('std_ratio', 0.0):.3f} del_r={stats.get('delta_ratio', 0.0):.3f} "
            f"src_std={stats.get('source_std', 0.0):.3f} voi_m={stats.get('voicing_mean', 0.0):.3f} "
            f"scale={stats.get('scale', 0.0):.3f}{' best' if is_best else ''}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train video parametric source-filter vocoder controls.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--output-dir", default="checkpoints_srcV9_parametric_vocoder")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=45)
    parser.add_argument("--hop-frames", type=int, default=15)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    parser.add_argument("--random-windows-per-file", type=int, default=0)
    parser.add_argument("--smooth-target-frames", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--source-bands", type=int, default=8)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--visual-width", type=int, default=24)
    parser.add_argument("--visual-layers", type=parse_layers, default=(1, 1, 1, 1))
    parser.add_argument("--visual-temporal-layers", type=int, default=1)
    parser.add_argument("--landmark-tcn-layers", type=int, default=6)
    parser.add_argument("--landmark-transformer-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--visual-lr", type=float, default=1e-4)
    parser.add_argument("--landmark-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-filter", type=float, default=0.8)
    parser.add_argument("--lambda-energy", type=float, default=0.5)
    parser.add_argument("--lambda-voicing", type=float, default=0.2)
    parser.add_argument("--lambda-broad", type=float, default=0.4)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--lambda-final-delta", type=float, default=0.3)
    parser.add_argument("--source-scale-init", type=float, default=0.35)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-last", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
