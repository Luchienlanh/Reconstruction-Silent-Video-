from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm.auto import tqdm

from srcV7.models import MaskedMelLoss
from srcV7.training.common import (
    build_model,
    compute_mel_stats,
    init_decoder_output_bias,
    load_checkpoint,
    make_loader,
    masked_stats,
    mean_baseline,
    model_inputs,
    sanitize_batch,
    save_checkpoint,
    split_cache_files,
    target_mel_stats,
    unwrap_model,
    write_history,
)
from srcV7.utils.common import batch_to_device, get_device, seed_everything
from srcV7.utils.plotting import save_mel_comparison


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args, epoch):
    model.train()
    raw_model = unwrap_model(model)
    freeze_visual = epoch <= args.freeze_visual_epochs
    for param in raw_model.encoder.visual.parameters():
        param.requires_grad = not freeze_visual
    if freeze_visual:
        raw_model.encoder.visual.eval()

    criterion.set_shift_window(args.shift_warmup if epoch <= args.shift_warmup_epochs else args.shift_final)
    total = 0.0
    total_mel = 0.0
    total_stats = 0.0
    count = 0
    amp_enabled = device.type == "cuda" and args.amp
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        batch = sanitize_batch(batch)
        for _ in range(max(1, args.steps_per_batch)):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            if freeze_visual:
                raw_model.encoder.visual.eval()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(**model_inputs(batch), return_aux=args.lambda_stats > 0)
                pred = out["mel"] if isinstance(out, dict) else out
            with torch.amp.autocast("cuda", enabled=False):
                if not torch.isfinite(pred).all():
                    print(f"[warn] non-finite pred; paths={batch.get('paths', [])[:4]}")
                    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                mel_loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
                loss = mel_loss
                stats_loss = pred.new_tensor(0.0)
                if args.lambda_stats > 0:
                    stats_target = target_mel_stats(batch["mel"], batch["mel_mask"], criterion)
                    stats_loss = torch.nn.functional.smooth_l1_loss(out["mel_stats"].float(), stats_target.float())
                    loss = loss + args.lambda_stats * stats_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite train loss: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu())
            total_mel += float(mel_loss.detach().cpu())
            total_stats += float(stats_loss.detach().cpu())
            count += 1
    denom = max(1, count)
    return {
        "loss": total / denom,
        "mel": total_mel / denom,
        "stats": total_stats / denom,
        "visual_frozen": freeze_visual,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, plot_path=None, epoch=0):
    if loader is None:
        return None, {}
    model.eval()
    criterion.set_shift_window(0)
    total = 0.0
    count = 0
    first_stats = None
    plotted = False
    for batch in tqdm(loader, desc="val", leave=False):
        batch = batch_to_device(batch, device)
        batch = sanitize_batch(batch)
        pred = model(**model_inputs(batch))
        if not torch.isfinite(pred).all():
            print(f"[warn] non-finite eval pred; paths={batch.get('paths', [])[:4]}")
            pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
        if first_stats is None:
            first_stats = masked_stats(pred, batch["mel"], batch["mel_mask"])
        if plot_path is not None and not plotted:
            save_mel_comparison(pred, batch["mel"], plot_path, title=f"epoch {epoch}")
            plotted = True
    return total / max(1, count), (first_stats or {})


def run(args) -> None:
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
    train_eval_loader = (
        make_loader(
            args.data_dir,
            train_files,
            args.val_batch_size or args.batch_size,
            max_frames=args.max_frames,
            random_crop=False,
            seed=args.seed,
            num_workers=args.num_workers,
            shuffle=False,
        )
        if args.eval_train_every > 0
        else None
    )
    stats_loader = make_loader(args.data_dir, train_files, args.batch_size, max_frames=args.max_frames, random_crop=False, seed=args.seed)
    mel_mean, mel_std = compute_mel_stats(stats_loader, device)
    criterion = MaskedMelLoss(
        mel_mean,
        mel_std,
        lambda_mel=args.lambda_mel,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
    ).to(device)

    model = build_model(device, args)
    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    else:
        init_decoder_output_bias(model, mel_mean)

    raw_model = unwrap_model(model)
    fusion_params = (
        list(raw_model.encoder.time.parameters())
        + list(raw_model.encoder.fuse.parameters())
        + list(raw_model.encoder.norm.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.encoder.visual.parameters(), "lr": args.lr * args.visual_lr_scale},
            {"params": raw_model.encoder.landmarks.parameters(), "lr": args.lr * args.landmark_lr_scale},
            {"params": fusion_params, "lr": args.lr * args.fusion_lr_scale},
            {"params": raw_model.mel_stats_head.parameters(), "lr": args.lr * args.fusion_lr_scale},
            {"params": raw_model.decoder.parameters(), "lr": args.lr * args.decoder_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    mean_train = mean_baseline(train_loader, criterion, mel_mean, device)
    mean_val = mean_baseline(val_loader, criterion, mel_mean, device)
    print(f"[device] {device}")
    print(f"[data] train={len(train_files)} val={len(val_files)}")
    print(
        f"[model] r2plus1d_{args.decoder_type} dim={args.dim} spatial_tokens={args.spatial_tokens} "
        f"decoder_channels={args.decoder_channels or args.dim} layers={args.decoder_layers}"
    )
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={'n/a' if mean_val is None else f'{mean_val:.6f}'}")

    best = float("inf")
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args, epoch)
        train_loss = train_metrics["loss"]
        val_loss, stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args,
            plot_path=output_dir / f"mel_epoch_{epoch:04d}.png" if val_loader is not None and epoch % args.plot_every == 0 else None,
            epoch=epoch,
        )
        train_eval_loss = None
        if train_eval_loader is not None and (epoch == 1 or epoch % args.eval_train_every == 0 or epoch == args.epochs):
            train_eval_loss, _ = evaluate(model, train_eval_loader, criterion, device, args)
        score = (train_eval_loss if train_eval_loss is not None else train_loss) if val_loss is None else val_loss
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        row = {
            "epoch": epoch,
            "train": train_loss,
            "train_mel": train_metrics["mel"],
            "train_stats": train_metrics["stats"],
            "train_eval": train_eval_loss,
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
        train_eval_text = "" if train_eval_loss is None else f" train_eval={train_eval_loss:.6f}"
        gap = "" if row["gap_vs_mean"] is None else f" gap_vs_mean={row['gap_vs_mean']:+.6f}"
        phase_text = " visual=frozen" if train_metrics["visual_frozen"] else " visual=train"
        aux_text = f" mel={train_metrics['mel']:.6f}"
        if args.lambda_stats > 0:
            aux_text += f" stats={train_metrics['stats']:.6f}"
        stat_text = "" if not stats else f" std_r={stats.get('std_ratio', 0):.3f} del_r={stats.get('delta_ratio', 0):.3f}"
        print(
            f"[epoch {epoch:04d}] train={train_loss:.6f}{aux_text}{train_eval_text} "
            f"val={val_text} best={best:.6f}{gap}{stat_text}{phase_text}{' best' if is_best else ''}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train srcV7 ResNet2+1D encoder + CNN mel decoder.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR")
    parser.add_argument("--output-dir", default="checkpoints_srcV7_cnn_plain")
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
    parser.add_argument("--decoder-lr-scale", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--steps-per-batch", type=int, default=1)
    parser.add_argument("--freeze-visual-epochs", type=int, default=0)
    parser.add_argument("--lambda-stats", type=float, default=0.0)
    parser.add_argument("--lambda-mel", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.25)
    parser.add_argument("--lambda-delta2", type=float, default=0.05)
    parser.add_argument("--lambda-energy", type=float, default=0.05)
    parser.add_argument("--eval-train-every", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop-last", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-tokens", type=int, default=4)
    parser.add_argument("--upsample-mode", default="conv_transpose", choices=["linear", "conv_transpose"])
    parser.add_argument("--decoder-type", default="cnn_plain", choices=["cnn_plain", "cnn_film"])
    parser.add_argument("--decoder-channels", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--decoder-kernel-size", type=int, default=5)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--shift-warmup", type=int, default=2)
    parser.add_argument("--shift-final", type=int, default=0)
    parser.add_argument("--shift-warmup-epochs", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
