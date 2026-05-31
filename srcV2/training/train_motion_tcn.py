from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from srcV2.models import MaskedMelLoss, MotionTCNLipToSpeechModel
from srcV2.training.common import (
    compute_mel_stats,
    load_checkpoint,
    make_loader,
    mean_baseline,
    save_checkpoint,
    split_cache_files,
    write_history,
)
from srcV2.training.train_simple import evaluate, init_output_bias, train_one_epoch
from srcV2.utils.common import get_device, seed_everything


def run(args) -> None:
    args.model_type = "motion_tcn"
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, args.limit)
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
    val_loader = make_loader(
        args.data_dir,
        val_files,
        args.val_batch_size or args.batch_size,
        max_frames=args.max_frames,
        random_crop=False,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
    ) if val_files else None
    stats_loader = make_loader(
        args.data_dir,
        train_files,
        args.batch_size,
        max_frames=args.max_frames,
        random_crop=False,
        seed=args.seed,
    )
    mel_mean, mel_std = compute_mel_stats(stats_loader, device)
    criterion = MaskedMelLoss(mel_mean, mel_std).to(device)

    model = MotionTCNLipToSpeechModel(
        dim=args.dim,
        spatial_tokens=args.spatial_tokens,
        num_points=args.num_landmark_points,
        dropout=args.dropout,
        decoder_layers=args.decoder_layers,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", True):
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)

    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    else:
        init_output_bias(model, mel_mean)

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.visual.parameters(), "lr": args.lr * args.visual_lr_scale},
            {"params": raw_model.landmarks.parameters(), "lr": args.lr * args.landmark_lr_scale},
            {"params": list(raw_model.fusion.parameters()) + list(raw_model.frame_refine.parameters()), "lr": args.lr * args.fusion_lr_scale},
            {"params": raw_model.upsample.parameters(), "lr": args.lr * args.fusion_lr_scale},
            {"params": raw_model.decoder.parameters(), "lr": args.lr * args.decoder_lr_scale},
            {"params": raw_model.mel_head.parameters(), "lr": args.lr * args.decoder_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "onecycle":
        steps_per_epoch = len(train_loader) * max(1, args.steps_per_batch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                args.lr * args.visual_lr_scale,
                args.lr * args.landmark_lr_scale,
                args.lr * args.fusion_lr_scale,
                args.lr * args.fusion_lr_scale,
                args.lr * args.decoder_lr_scale,
                args.lr * args.decoder_lr_scale,
            ],
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    mean_train = mean_baseline(train_loader, criterion, mel_mean, device)
    mean_val = mean_baseline(val_loader, criterion, mel_mean, device) if val_loader is not None else None
    print(f"[device] {device}")
    print(f"[data] train={len(train_files)} val={len(val_files)}")
    print(f"[model] Motion-TCN Lip-to-Speech dim={args.dim} spatial_tokens={args.spatial_tokens} decoder_layers={args.decoder_layers}")
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={'n/a' if mean_val is None else f'{mean_val:.6f}'}")

    best = float("inf")
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, args, epoch)
        train_loss = train_metrics["loss"]
        if scheduler is not None and args.lr_scheduler == "cosine":
            scheduler.step()

        val_loss, stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args,
            plot_path=output_dir / f"mel_epoch_{epoch:04d}.png" if val_loader is not None and epoch % args.plot_every == 0 else None,
            epoch=epoch,
        ) if val_loader is not None else (None, {})
        score = train_loss if val_loss is None else val_loss
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)

        row = {
            "epoch": epoch,
            "train": train_loss,
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
        print(f"[epoch {epoch:04d}] train={train_loss:.6f} val={val_text} best={best:.6f}{gap}{stat_text}{phase_text}{' best' if is_best else ''}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Motion-TCN lip-to-speech model.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR")
    parser.add_argument("--output-dir", default="checkpoints_motion_tcn")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--random-crop", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--visual-lr-scale", type=float, default=0.5)
    parser.add_argument("--landmark-lr-scale", type=float, default=1.5)
    parser.add_argument("--fusion-lr-scale", type=float, default=1.5)
    parser.add_argument("--decoder-lr-scale", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--steps-per-batch", type=int, default=1)
    parser.add_argument("--freeze-visual-epochs", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop-last", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--spatial-tokens", type=int, default=2)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--lr-scheduler", default="onecycle", choices=["constant", "onecycle", "cosine"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
