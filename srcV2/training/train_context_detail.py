from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from srcV2.models import ContextDetailLipToSpeechModel, ContextMotionDetailLipToSpeechModel
from srcV2.training.common import (
    compute_mel_stats,
    load_checkpoint,
    make_loader,
    mean_baseline,
    save_checkpoint,
    split_cache_files,
    write_history,
)
from srcV2.training.train_context_sync import infer_num_units, make_criterion, train_one_epoch
from srcV2.training.train_simple import evaluate
from srcV2.utils.common import get_device, seed_everything


def init_output_bias(model, mel_mean: torch.Tensor) -> None:
    target = model.module if hasattr(model, "module") else model
    head = getattr(target, "mel_base_head", None)
    if head is None or head.bias is None or head.bias.numel() != mel_mean.numel():
        return
    with torch.no_grad():
        head.bias.copy_(mel_mean.to(device=head.bias.device, dtype=head.bias.dtype))


def build_context_detail_model(args, device: torch.device) -> torch.nn.Module:
    model_cls = ContextMotionDetailLipToSpeechModel if args.detail_source == "visual_motion" else ContextDetailLipToSpeechModel
    model = model_cls(
        dim=args.dim,
        spatial_tokens=args.spatial_tokens,
        num_points=args.num_landmark_points,
        dropout=args.dropout,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        num_units=args.num_units if args.unit_loss_weight > 0 else 0,
        detail_scale=args.detail_scale,
        detail_layers=args.detail_layers,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", True):
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    raw_model = model.module if hasattr(model, "module") else model
    detail_params = (
        list(raw_model.motion_proj.parameters())
        + list(raw_model.detail_refine.parameters())
        + list(raw_model.mel_base_head.parameters())
        + list(raw_model.mel_detail_head.parameters())
        + list(raw_model.energy_head.parameters())
    )
    if hasattr(raw_model, "detail_upsample"):
        detail_params += list(raw_model.detail_upsample.parameters())
    if hasattr(raw_model, "detail_gate"):
        detail_params += list(raw_model.detail_gate.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.visual.parameters(), "lr": args.lr * args.visual_lr_scale},
            {"params": raw_model.geometry.parameters(), "lr": args.lr * args.geometry_lr_scale},
            {
                "params": list(raw_model.fusion.parameters())
                + list(raw_model.encoder.parameters())
                + list(raw_model.global_film.parameters()),
                "lr": args.lr * args.fusion_lr_scale,
            },
            {"params": raw_model.upsample.parameters(), "lr": args.lr * args.decoder_lr_scale},
            {"params": raw_model.decoder.parameters(), "lr": args.lr * args.decoder_lr_scale},
            {"params": detail_params, "lr": args.lr * args.detail_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    if raw_model.unit_head is not None:
        optimizer.add_param_group({"params": raw_model.unit_head.parameters(), "lr": args.lr * args.decoder_lr_scale})
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, train_loader, args, has_unit_head: bool):
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.lr_scheduler == "onecycle":
        steps_per_epoch = len(train_loader) * max(1, args.steps_per_batch)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                args.lr * args.visual_lr_scale,
                args.lr * args.geometry_lr_scale,
                args.lr * args.fusion_lr_scale,
                args.lr * args.decoder_lr_scale,
                args.lr * args.decoder_lr_scale,
                args.lr * args.detail_lr_scale,
                *([args.lr * args.decoder_lr_scale] if has_unit_head else []),
            ],
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    return None


def run(args) -> None:
    args.model_type = "context_motion_detail" if args.detail_source == "visual_motion" else "context_detail"
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, args.limit)
    if args.num_units <= 0:
        args.num_units = infer_num_units(train_files)
    if args.unit_loss_weight > 0 and args.num_units <= 0:
        raise RuntimeError(
            "Unit loss is enabled but no speech_units/num_speech_units were found. "
            "Run srcV2.data.build_speech_units first or set --unit-loss-weight 0."
        )
    train_loader = make_loader(
        args.data_dir,
        train_files,
        args.batch_size,
        max_frames=args.max_frames,
        random_crop=args.random_crop,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=args.drop_last,
    )
    val_loader = (
        make_loader(
            args.data_dir,
            val_files,
            args.val_batch_size or args.batch_size,
            max_frames=args.max_frames,
            random_crop=False,
            seed=args.seed,
            num_workers=args.num_workers,
            shuffle=False,
        )
        if val_files
        else None
    )
    stats_loader = make_loader(
        args.data_dir,
        train_files,
        args.batch_size,
        max_frames=args.max_frames,
        random_crop=False,
        seed=args.seed,
    )
    mel_mean, mel_std = compute_mel_stats(stats_loader, device)
    criterion = make_criterion(args, mel_mean, mel_std, device)
    model = build_context_detail_model(args, device)

    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_checkpoint(args.resume, model, device)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
    else:
        init_output_bias(model, mel_mean)

    optimizer = build_optimizer(model, args)
    if resume_ckpt is not None and resume_ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    raw_model = model.module if hasattr(model, "module") else model
    scheduler = build_scheduler(optimizer, train_loader, args, raw_model.unit_head is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    mean_train = mean_baseline(train_loader, criterion, mel_mean, device)
    mean_val = mean_baseline(val_loader, criterion, mel_mean, device) if val_loader is not None else None
    print(f"[device] {device}")
    print(f"[data] train={len(train_files)} val={len(val_files)}")
    model_name = "ContextMotionDetail" if args.detail_source == "visual_motion" else "ContextDetail"
    print(
        f"[model] {model_name} dim={args.dim} spatial_tokens={args.spatial_tokens} "
        f"enc={args.encoder_layers} dec={args.decoder_layers} heads={args.heads} units={args.num_units} "
        f"detail_scale={args.detail_scale} detail_source={args.detail_source}"
    )
    print(
        f"[loss] delta={args.lambda_delta} delta2={args.lambda_delta2} energy={args.lambda_energy} "
        f"mfcc={args.lambda_mfcc} flux={args.lambda_flux} voice={args.lambda_voicing} "
        f"mismatch_w={args.mismatch_loss_weight} unit_w={args.unit_loss_weight}"
    )
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={'n/a' if mean_val is None else f'{mean_val:.6f}'}")

    best = float(resume_ckpt.get("best", float("inf"))) if resume_ckpt is not None else float("inf")
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, args, epoch)
        train_loss = train_metrics["loss"]
        if scheduler is not None and args.lr_scheduler == "cosine":
            scheduler.step()

        val_loss, stats = (
            evaluate(
                model,
                val_loader,
                criterion,
                device,
                args,
                plot_path=output_dir / f"mel_epoch_{epoch:04d}.png"
                if val_loader is not None and epoch % args.plot_every == 0
                else None,
                epoch=epoch,
            )
            if val_loader is not None
            else (None, {})
        )
        score = train_loss if val_loss is None else val_loss
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)

        row = {
            "epoch": epoch,
            "train": train_loss,
            "train_mel": train_metrics["mel"],
            "train_unit": train_metrics["unit"],
            "unit_weight": train_metrics["unit_weight"],
            "train_sync": train_metrics["sync"],
            "train_mismatch": train_metrics["mismatch"],
            "val": val_loss,
            "best": best,
            "mean_train": mean_train,
            "mean_val": mean_val,
            "gap_vs_mean": (val_loss - mean_val) if val_loss is not None and mean_val is not None else None,
            **stats,
        }
        history.append(row)
        write_history(output_dir / "history.json", history, args)

        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        gap = "" if row["gap_vs_mean"] is None else f" gap_vs_mean={row['gap_vs_mean']:+.6f}"
        phase_text = " visual=frozen" if train_metrics["visual_frozen"] else " visual=train"
        stat_text = "" if not stats else f" std_r={stats.get('std_ratio', 0):.3f} del_r={stats.get('delta_ratio', 0):.3f}"
        print(
            f"[epoch {epoch:04d}] train={train_loss:.6f} mel={train_metrics['mel']:.6f} "
            f"unit={train_metrics['unit']:.4f} unit_w={train_metrics['unit_weight']:.3f} "
            f"sync={train_metrics['sync']:.4f} mismatch={train_metrics['mismatch']:.6f} "
            f"val={val_text} best={best:.6f}{gap}{stat_text}{phase_text}{' best' if is_best else ''}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ContextDetail lip-to-speech model.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR")
    parser.add_argument("--output-dir", default="checkpoints_context_detail")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--random-crop", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--visual-lr-scale", type=float, default=0.5)
    parser.add_argument("--geometry-lr-scale", type=float, default=2.0)
    parser.add_argument("--fusion-lr-scale", type=float, default=1.5)
    parser.add_argument("--decoder-lr-scale", type=float, default=2.0)
    parser.add_argument("--detail-lr-scale", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=3.0)
    parser.add_argument("--steps-per-batch", type=int, default=1)
    parser.add_argument("--freeze-visual-epochs", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop-last", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--spatial-tokens", type=int, default=2)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--detail-scale", type=float, default=0.35)
    parser.add_argument("--detail-layers", type=int, default=3)
    parser.add_argument("--detail-source", default="decoder_hidden", choices=["decoder_hidden", "visual_motion"])
    parser.add_argument("--num-units", type=int, default=0)
    parser.add_argument("--unit-loss-weight", type=float, default=0.02)
    parser.add_argument("--unit-warmup-epochs", type=int, default=5)
    parser.add_argument("--unit-ramp-epochs", type=int, default=10)
    parser.add_argument("--lambda-mel", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.50)
    parser.add_argument("--lambda-delta2", type=float, default=0.12)
    parser.add_argument("--lambda-energy", type=float, default=0.15)
    parser.add_argument("--lambda-mfcc", type=float, default=0.03)
    parser.add_argument("--lambda-flux", type=float, default=0.02)
    parser.add_argument("--lambda-voicing", type=float, default=0.005)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--shift-window", type=int, default=0)
    parser.add_argument("--mismatch-loss-weight", type=float, default=0.75)
    parser.add_argument("--mismatch-margin", type=float, default=0.12)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["constant", "onecycle", "cosine"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
