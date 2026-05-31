from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from srcV2.models import ContextSyncLipToSpeechModel, MaskedMelLoss
from srcV2.training.common import (
    compute_mel_stats,
    load_checkpoint,
    make_loader,
    mean_baseline,
    model_inputs,
    save_checkpoint,
    split_cache_files,
    write_history,
)
from srcV2.training.train_simple import evaluate, init_output_bias, sanitize_batch
from srcV2.utils.common import batch_to_device, get_device, seed_everything


def make_mismatch_inputs(batch: dict) -> dict | None:
    if batch["video"].shape[0] < 2:
        return None
    out = model_inputs(batch)
    source_keys = {
        "video",
        "landmarks",
        "mouth_valid_mask",
        "video_mask",
        "video_times",
        "video_lengths",
    }
    for key in source_keys:
        if key in out and torch.is_tensor(out[key]):
            out[key] = torch.roll(out[key], shifts=1, dims=0)
    return out


def infer_num_units(files: list[Path]) -> int:
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if "num_speech_units" in item:
            return int(item["num_speech_units"])
        if "speech_units" in item:
            return int(item["speech_units"].max().item()) + 1
    return 0


def model_outputs(model, batch: dict, return_aux: bool = False):
    inputs = model_inputs(batch)
    if return_aux:
        inputs["return_aux"] = True
    out = model(inputs)
    if isinstance(out, dict):
        return out
    return {"mel": out}


def speech_unit_loss(outputs: dict, batch: dict, args) -> torch.Tensor:
    logits = outputs.get("unit_logits")
    if logits is None or "speech_units" not in batch or args.unit_loss_weight <= 0:
        mel = outputs["mel"]
        return mel.new_tensor(0.0)
    if not torch.isfinite(logits).all():
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    targets = batch["speech_units"].long()
    if targets.shape[1] != logits.shape[1]:
        x = targets.float().unsqueeze(1)
        targets = F.interpolate(x, size=logits.shape[1], mode="nearest").squeeze(1).long()
    return F.cross_entropy(logits.transpose(1, 2).float(), targets, ignore_index=-100)


def unit_weight_for_epoch(args, epoch: int) -> float:
    if args.unit_loss_weight <= 0:
        return 0.0
    if epoch <= args.unit_warmup_epochs:
        return 0.0
    ramp = max(1, int(args.unit_ramp_epochs))
    progress = min(1.0, max(0.0, (epoch - args.unit_warmup_epochs) / float(ramp)))
    return float(args.unit_loss_weight) * progress


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, args, epoch):
    model.train()
    raw_model = model.module if hasattr(model, "module") else model

    freeze_visual = epoch <= args.freeze_visual_epochs
    for param in raw_model.visual.parameters():
        param.requires_grad = not freeze_visual
    if freeze_visual:
        raw_model.visual.eval()

    amp_enabled = device.type == "cuda" and args.amp
    total = 0.0
    mel_total = 0.0
    unit_total = 0.0
    sync_total = 0.0
    mismatch_total = 0.0
    count = 0
    effective_unit_weight = unit_weight_for_epoch(args, epoch)

    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        batch = sanitize_batch(batch)

        for _ in range(max(1, args.steps_per_batch)):
            optimizer.zero_grad(set_to_none=True)
            if freeze_visual:
                raw_model.visual.eval()

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model_outputs(model, batch, return_aux=effective_unit_weight > 0)
                pred = outputs["mel"]
                mismatch_inputs = make_mismatch_inputs(batch) if args.mismatch_loss_weight > 0 else None
                mismatch_pred = model(mismatch_inputs) if mismatch_inputs is not None else None

            with torch.amp.autocast("cuda", enabled=False):
                if not torch.isfinite(pred).all():
                    print(f"[warn] non-finite pred; paths={batch.get('paths', [])[:4]}")
                    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                mel_loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
                unit_loss = speech_unit_loss(outputs, batch, args)
                loss = mel_loss + effective_unit_weight * unit_loss
                sync_loss = pred.new_tensor(0.0)
                mismatch_loss = pred.new_tensor(0.0)
                if mismatch_pred is not None:
                    if not torch.isfinite(mismatch_pred).all():
                        mismatch_pred = torch.nan_to_num(mismatch_pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                    mismatch_loss = criterion(mismatch_pred.float(), batch["mel"].float(), batch["mel_mask"])
                    sync_loss = torch.relu(args.mismatch_margin + mel_loss.detach() - mismatch_loss)
                    loss = loss + args.mismatch_loss_weight * sync_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite train loss: {float(loss.detach().cpu())}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and args.lr_scheduler == "onecycle":
                scheduler.step()

            total += float(loss.detach().cpu())
            mel_total += float(mel_loss.detach().cpu())
            unit_total += float(unit_loss.detach().cpu())
            sync_total += float(sync_loss.detach().cpu())
            mismatch_total += float(mismatch_loss.detach().cpu())
            count += 1

    denom = max(1, count)
    return {
        "loss": total / denom,
        "mel": mel_total / denom,
        "unit": unit_total / denom,
        "unit_weight": effective_unit_weight,
        "sync": sync_total / denom,
        "mismatch": mismatch_total / denom,
        "visual_frozen": freeze_visual,
    }


def make_criterion(args, mel_mean, mel_std, device):
    return MaskedMelLoss(
        mel_mean,
        mel_std,
        lambda_mel=args.lambda_mel,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
        lambda_mfcc=args.lambda_mfcc,
        lambda_flux=args.lambda_flux,
        lambda_voicing=args.lambda_voicing,
        n_mfcc=args.n_mfcc,
        shift_window=args.shift_window,
    ).to(device)


def run(args) -> None:
    args.model_type = "context_sync"
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

    model = ContextSyncLipToSpeechModel(
        dim=args.dim,
        spatial_tokens=args.spatial_tokens,
        num_points=args.num_landmark_points,
        dropout=args.dropout,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        num_units=args.num_units if args.unit_loss_weight > 0 else 0,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", True):
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)

    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_checkpoint(args.resume, model, device)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
    else:
        init_output_bias(model, mel_mean)

    raw_model = model.module if hasattr(model, "module") else model
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
            {"params": list(raw_model.mel_head.parameters()) + list(raw_model.energy_head.parameters()), "lr": args.lr * args.decoder_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    if raw_model.unit_head is not None:
        optimizer.add_param_group({"params": raw_model.unit_head.parameters(), "lr": args.lr * args.decoder_lr_scale})
    if resume_ckpt is not None and resume_ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "onecycle":
        steps_per_epoch = len(train_loader) * max(1, args.steps_per_batch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                args.lr * args.visual_lr_scale,
                args.lr * args.geometry_lr_scale,
                args.lr * args.fusion_lr_scale,
                args.lr * args.decoder_lr_scale,
                args.lr * args.decoder_lr_scale,
                args.lr * args.decoder_lr_scale,
                *([args.lr * args.decoder_lr_scale] if raw_model.unit_head is not None else []),
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
    print(
        f"[model] ContextSync dim={args.dim} spatial_tokens={args.spatial_tokens} "
        f"enc={args.encoder_layers} dec={args.decoder_layers} heads={args.heads} units={args.num_units}"
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
    parser = argparse.ArgumentParser(description="Train ContextSync lip-to-speech model.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR")
    parser.add_argument("--output-dir", default="checkpoints_context_sync")
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
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--visual-lr-scale", type=float, default=0.5)
    parser.add_argument("--geometry-lr-scale", type=float, default=2.0)
    parser.add_argument("--fusion-lr-scale", type=float, default=1.5)
    parser.add_argument("--decoder-lr-scale", type=float, default=2.0)
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
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-units", type=int, default=0)
    parser.add_argument("--unit-loss-weight", type=float, default=0.0)
    parser.add_argument("--unit-warmup-epochs", type=int, default=0)
    parser.add_argument("--unit-ramp-epochs", type=int, default=1)
    parser.add_argument("--lambda-mel", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.50)
    parser.add_argument("--lambda-delta2", type=float, default=0.12)
    parser.add_argument("--lambda-energy", type=float, default=0.15)
    parser.add_argument("--lambda-mfcc", type=float, default=0.0)
    parser.add_argument("--lambda-flux", type=float, default=0.0)
    parser.add_argument("--lambda-voicing", type=float, default=0.0)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--shift-window", type=int, default=0)
    parser.add_argument("--mismatch-loss-weight", type=float, default=0.10)
    parser.add_argument("--mismatch-margin", type=float, default=0.05)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["constant", "onecycle", "cosine"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
