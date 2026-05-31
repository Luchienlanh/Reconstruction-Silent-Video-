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

from srcV2.models import ContentUnitLipToSpeechModel
from srcV2.training.common import (
    compute_mel_stats,
    load_checkpoint,
    make_loader,
    mean_baseline,
    masked_stats,
    save_checkpoint,
    split_cache_files,
    write_history,
)
from srcV2.training.train_context_sync import (
    infer_num_units,
    make_criterion,
    make_mismatch_inputs,
    model_outputs,
    unit_weight_for_epoch,
)
from srcV2.training.train_simple import sanitize_batch
from srcV2.utils.common import batch_to_device, get_device, seed_everything
from srcV2.utils.plotting import save_mel_comparison


def init_output_bias(model, mel_mean: torch.Tensor) -> None:
    target = model.module if hasattr(model, "module") else model
    head = getattr(target, "mel_head", None)
    if head is None or head.bias is None or head.bias.numel() != mel_mean.numel():
        return
    with torch.no_grad():
        head.bias.copy_(mel_mean.to(device=head.bias.device, dtype=head.bias.dtype))


def unit_frame_accuracy(outputs: dict, batch: dict) -> torch.Tensor:
    logits = outputs.get("unit_logits")
    if logits is None or "speech_units" not in batch:
        mel = outputs["mel"]
        return mel.new_tensor(0.0)
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    targets = batch["speech_units"].long()
    if targets.shape[1] != logits.shape[1]:
        x = targets.float().unsqueeze(1)
        targets = F.interpolate(x, size=logits.shape[1], mode="nearest").squeeze(1).long()
    mask = targets.ge(0)
    if "mel_mask" in batch and batch["mel_mask"].shape[1] == logits.shape[1]:
        mask = mask & batch["mel_mask"].to(mask.device, dtype=torch.bool)
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    pred = logits.argmax(dim=-1)
    return (pred.eq(targets) & mask).float().sum() / mask.float().sum().clamp_min(1.0)


def speech_unit_loss(outputs: dict, batch: dict, args) -> torch.Tensor:
    logits = outputs.get("unit_logits")
    if logits is None or "speech_units" not in batch or args.unit_loss_weight <= 0:
        mel = outputs["mel"]
        return mel.new_tensor(0.0)
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    clamp = float(getattr(args, "unit_logit_clamp", 0.0))
    if clamp > 0:
        logits = logits.clamp(-clamp, clamp)
    targets = batch["speech_units"].long()
    if targets.shape[1] != logits.shape[1]:
        x = targets.float().unsqueeze(1)
        targets = F.interpolate(x, size=logits.shape[1], mode="nearest").squeeze(1).long()
    return F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        ignore_index=-100,
        label_smoothing=float(getattr(args, "unit_label_smoothing", 0.0)),
    )


def teacher_prob_for_epoch(args, epoch: int) -> float:
    start = float(args.unit_teacher_prob)
    if start <= 0:
        return 0.0
    decay = int(args.unit_teacher_decay_epochs)
    if decay <= 0:
        return start
    progress = min(1.0, max(0.0, (epoch - 1) / float(decay)))
    return max(float(args.unit_teacher_min), start * (1.0 - progress))


def build_model(args, device: torch.device) -> torch.nn.Module:
    model = ContentUnitLipToSpeechModel(
        dim=args.dim,
        spatial_tokens=args.spatial_tokens,
        num_points=args.num_landmark_points,
        dropout=args.dropout,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        num_units=args.num_units,
        unit_temperature=args.unit_temperature,
        detach_unit_condition=args.detach_unit_condition,
        detach_content_hidden=args.detach_content_hidden,
        unit_teacher_prob=args.unit_teacher_prob,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", True):
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    raw = model.module if hasattr(model, "module") else model
    return torch.optim.AdamW(
        [
            {"params": raw.visual.parameters(), "lr": args.lr * args.visual_lr_scale},
            {"params": raw.geometry.parameters(), "lr": args.lr * args.geometry_lr_scale},
            {
                "params": list(raw.fusion.parameters()) + list(raw.encoder.parameters()),
                "lr": args.lr * args.encoder_lr_scale,
            },
            {
                "params": list(raw.unit_upsample.parameters())
                + list(raw.unit_refine.parameters())
                + list(raw.unit_norm.parameters())
                + list(raw.unit_head.parameters()),
                "lr": args.lr * args.unit_lr_scale,
            },
            {
                "params": list(raw.unit_embedding.parameters())
                + list(raw.content_fusion.parameters())
                + list(raw.decoder.parameters())
                + list(raw.out_norm.parameters())
                + list(raw.mel_head.parameters())
                + list(raw.energy_head.parameters()),
                "lr": args.lr * args.decoder_lr_scale,
            },
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, train_loader, args):
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.lr_scheduler == "onecycle":
        steps_per_epoch = len(train_loader) * max(1, args.steps_per_batch)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                args.lr * args.visual_lr_scale,
                args.lr * args.geometry_lr_scale,
                args.lr * args.encoder_lr_scale,
                args.lr * args.unit_lr_scale,
                args.lr * args.decoder_lr_scale,
            ],
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    return None


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, args, epoch: int):
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.unit_teacher_prob = teacher_prob_for_epoch(args, epoch)

    freeze_visual = epoch <= args.freeze_visual_epochs
    for param in raw_model.visual.parameters():
        param.requires_grad = not freeze_visual
    if freeze_visual:
        raw_model.visual.eval()

    amp_enabled = device.type == "cuda" and args.amp
    unit_w = unit_weight_for_epoch(args, epoch)
    unit_mismatch_w = float(args.unit_mismatch_loss_weight)
    if epoch < int(args.unit_mismatch_start_epoch):
        unit_mismatch_w = 0.0
    totals = {
        "loss": 0.0,
        "mel": 0.0,
        "unit": 0.0,
        "unit_acc": 0.0,
        "sync": 0.0,
        "unit_sync": 0.0,
        "mismatch": 0.0,
        "mismatch_unit": 0.0,
    }
    count = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        batch = sanitize_batch(batch)

        for _ in range(max(1, args.steps_per_batch)):
            optimizer.zero_grad(set_to_none=True)
            if freeze_visual:
                raw_model.visual.eval()

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model_outputs(model, batch, return_aux=True)
                pred = outputs["mel"]
                mismatch_inputs = make_mismatch_inputs(batch) if args.mismatch_loss_weight > 0 or unit_mismatch_w > 0 else None
                mismatch_outputs = None
                if mismatch_inputs is not None:
                    teacher_prob = raw_model.unit_teacher_prob
                    raw_model.unit_teacher_prob = 0.0
                    mismatch_outputs = model_outputs(model, mismatch_inputs, return_aux=True)
                    raw_model.unit_teacher_prob = teacher_prob

            with torch.amp.autocast("cuda", enabled=False):
                if not torch.isfinite(pred).all():
                    print(f"[warn] non-finite pred; paths={batch.get('paths', [])[:4]}")
                    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                mel_loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
                unit_loss = speech_unit_loss(outputs, batch, args)
                unit_acc = unit_frame_accuracy(outputs, batch)
                loss = mel_loss + unit_w * unit_loss

                sync_loss = pred.new_tensor(0.0)
                unit_sync_loss = pred.new_tensor(0.0)
                mismatch_loss = pred.new_tensor(0.0)
                mismatch_unit_loss = pred.new_tensor(0.0)
                if mismatch_outputs is not None:
                    mismatch_pred = mismatch_outputs["mel"]
                    if not torch.isfinite(mismatch_pred).all():
                        mismatch_pred = torch.nan_to_num(mismatch_pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                    mismatch_loss = criterion(mismatch_pred.float(), batch["mel"].float(), batch["mel_mask"])
                    mismatch_unit_loss = speech_unit_loss(mismatch_outputs, batch, args)
                    if args.mismatch_loss_weight > 0:
                        sync_loss = torch.relu(args.mismatch_margin + mel_loss.detach() - mismatch_loss)
                        loss = loss + args.mismatch_loss_weight * sync_loss
                    if unit_mismatch_w > 0 and unit_w > 0:
                        cap = float(args.unit_mismatch_ce_cap)
                        mismatch_unit_for_sync = mismatch_unit_loss.clamp_max(cap) if cap > 0 else mismatch_unit_loss
                        unit_sync_loss = torch.relu(args.unit_mismatch_margin + unit_loss.detach() - mismatch_unit_for_sync)
                        loss = loss + unit_mismatch_w * unit_sync_loss

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

            totals["loss"] += float(loss.detach().cpu())
            totals["mel"] += float(mel_loss.detach().cpu())
            totals["unit"] += float(unit_loss.detach().cpu())
            totals["unit_acc"] += float(unit_acc.detach().cpu())
            totals["sync"] += float(sync_loss.detach().cpu())
            totals["unit_sync"] += float(unit_sync_loss.detach().cpu())
            totals["mismatch"] += float(mismatch_loss.detach().cpu())
            totals["mismatch_unit"] += float(mismatch_unit_loss.detach().cpu())
            count += 1

    denom = max(1, count)
    return {
        **{key: value / denom for key, value in totals.items()},
        "unit_weight": unit_w,
        "teacher_prob": raw_model.unit_teacher_prob,
        "visual_frozen": freeze_visual,
    }


@torch.no_grad()
def evaluate_content(model, loader, criterion, device, args, plot_path=None, epoch: int = 0):
    model.eval()
    total_mel = 0.0
    total_unit = 0.0
    total_acc = 0.0
    count = 0
    first_stats = None
    plotted = False
    for batch in tqdm(loader, desc="val", leave=False):
        batch = batch_to_device(batch, device)
        batch = sanitize_batch(batch)
        outputs = model_outputs(model, batch, return_aux=True)
        pred = outputs["mel"]
        if not torch.isfinite(pred).all():
            print(f"[warn] non-finite eval pred; paths={batch.get('paths', [])[:4]}")
            pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        mel_loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
        unit_loss = speech_unit_loss(outputs, batch, args)
        unit_acc = unit_frame_accuracy(outputs, batch)
        total_mel += float(mel_loss.detach().cpu())
        total_unit += float(unit_loss.detach().cpu())
        total_acc += float(unit_acc.detach().cpu())
        count += 1
        if first_stats is None:
            first_stats = masked_stats(pred, batch["mel"], batch["mel_mask"])
        if plot_path is not None and not plotted:
            save_mel_comparison(pred, batch["mel"], plot_path, title=f"epoch {epoch}")
            plotted = True
    denom = max(1, count)
    return {
        "mel": total_mel / denom,
        "unit": total_unit / denom,
        "unit_acc": total_acc / denom,
        "score": (total_unit / denom) + args.val_mel_score_weight * (total_mel / denom),
        **(first_stats or {}),
    }


def run(args) -> None:
    args.model_type = "content_unit"
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, args.limit)
    if args.num_units <= 0:
        args.num_units = infer_num_units(train_files)
    if args.num_units <= 0:
        raise RuntimeError("Content-unit training requires speech_units. Run srcV2.data.build_speech_units first.")

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
    model = build_model(args, device)

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
    scheduler = build_scheduler(optimizer, train_loader, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    mean_train = mean_baseline(train_loader, criterion, mel_mean, device)
    mean_val = mean_baseline(val_loader, criterion, mel_mean, device) if val_loader is not None else None
    print(f"[device] {device}")
    print(f"[data] train={len(train_files)} val={len(val_files)}")
    print(
        f"[model] ContentUnit dim={args.dim} spatial_tokens={args.spatial_tokens} "
        f"enc={args.encoder_layers} dec={args.decoder_layers} heads={args.heads} units={args.num_units}"
    )
    print(
        f"[loss] mel={args.lambda_mel} delta={args.lambda_delta} delta2={args.lambda_delta2} "
        f"energy={args.lambda_energy} unit_w={args.unit_loss_weight} unit_mismatch_w={args.unit_mismatch_loss_weight}"
    )
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={'n/a' if mean_val is None else f'{mean_val:.6f}'}")

    best = float(resume_ckpt.get("best", float("inf"))) if resume_ckpt is not None else float("inf")
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, args, epoch)
        if scheduler is not None and args.lr_scheduler == "cosine":
            scheduler.step()

        val_metrics = (
            evaluate_content(
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
            else None
        )
        score = train_metrics["unit"] + args.val_mel_score_weight * train_metrics["mel"] if val_metrics is None else val_metrics["score"]
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)

        row = {
            "epoch": epoch,
            "train": train_metrics["loss"],
            "train_mel": train_metrics["mel"],
            "train_unit": train_metrics["unit"],
            "train_unit_acc": train_metrics["unit_acc"],
            "unit_weight": train_metrics["unit_weight"],
            "teacher_prob": train_metrics["teacher_prob"],
            "train_sync": train_metrics["sync"],
            "train_unit_sync": train_metrics["unit_sync"],
            "train_mismatch": train_metrics["mismatch"],
            "train_mismatch_unit": train_metrics["mismatch_unit"],
            "val": None if val_metrics is None else val_metrics["score"],
            "val_mel": None if val_metrics is None else val_metrics["mel"],
            "val_unit": None if val_metrics is None else val_metrics["unit"],
            "val_unit_acc": None if val_metrics is None else val_metrics["unit_acc"],
            "best": best,
            "mean_train": mean_train,
            "mean_val": mean_val,
            **({} if val_metrics is None else {k: v for k, v in val_metrics.items() if k not in {"score", "mel", "unit", "unit_acc"}}),
        }
        history.append(row)
        write_history(output_dir / "history.json", history, args)

        val_text = "n/a" if val_metrics is None else f"{val_metrics['score']:.6f}"
        val_unit_text = "n/a" if val_metrics is None else f"{val_metrics['unit']:.4f}"
        val_acc_text = "n/a" if val_metrics is None else f"{val_metrics['unit_acc']:.3f}"
        stat_text = (
            ""
            if val_metrics is None
            else f" std_r={val_metrics.get('std_ratio', 0):.3f} del_r={val_metrics.get('delta_ratio', 0):.3f}"
        )
        print(
            f"[epoch {epoch:04d}] train={train_metrics['loss']:.6f} mel={train_metrics['mel']:.6f} "
            f"unit={train_metrics['unit']:.4f} unit_acc={train_metrics['unit_acc']:.3f} "
            f"unit_w={train_metrics['unit_weight']:.3f} teacher={train_metrics['teacher_prob']:.2f} "
            f"sync={train_metrics['sync']:.4f} unit_sync={train_metrics['unit_sync']:.4f} "
            f"val={val_text} val_unit={val_unit_text} val_acc={val_acc_text} best={best:.6f}"
            f"{stat_text}{' visual=frozen' if train_metrics['visual_frozen'] else ' visual=train'}{' best' if is_best else ''}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train content-first speech-unit lip-to-speech model.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_NewVideo_mouth_units50")
    parser.add_argument("--output-dir", default="checkpoints_content_unit")
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
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--visual-lr-scale", type=float, default=0.5)
    parser.add_argument("--geometry-lr-scale", type=float, default=1.0)
    parser.add_argument("--encoder-lr-scale", type=float, default=1.5)
    parser.add_argument("--unit-lr-scale", type=float, default=2.5)
    parser.add_argument("--decoder-lr-scale", type=float, default=1.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
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
    parser.add_argument("--num-units", type=int, default=0)
    parser.add_argument("--unit-temperature", type=float, default=1.0)
    parser.add_argument("--detach-unit-condition", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--detach-content-hidden", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--unit-teacher-prob", type=float, default=0.6)
    parser.add_argument("--unit-teacher-decay-epochs", type=int, default=20)
    parser.add_argument("--unit-teacher-min", type=float, default=0.1)
    parser.add_argument("--unit-loss-weight", type=float, default=0.25)
    parser.add_argument("--unit-warmup-epochs", type=int, default=0)
    parser.add_argument("--unit-ramp-epochs", type=int, default=3)
    parser.add_argument("--unit-label-smoothing", type=float, default=0.05)
    parser.add_argument("--unit-logit-clamp", type=float, default=12.0)
    parser.add_argument("--unit-mismatch-loss-weight", type=float, default=0.0)
    parser.add_argument("--unit-mismatch-start-epoch", type=int, default=8)
    parser.add_argument("--unit-mismatch-ce-cap", type=float, default=6.0)
    parser.add_argument("--unit-mismatch-margin", type=float, default=0.15)
    parser.add_argument("--lambda-mel", type=float, default=0.30)
    parser.add_argument("--lambda-delta", type=float, default=0.15)
    parser.add_argument("--lambda-delta2", type=float, default=0.04)
    parser.add_argument("--lambda-energy", type=float, default=0.03)
    parser.add_argument("--lambda-mfcc", type=float, default=0.0)
    parser.add_argument("--lambda-flux", type=float, default=0.0)
    parser.add_argument("--lambda-voicing", type=float, default=0.0)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--shift-window", type=int, default=0)
    parser.add_argument("--mismatch-loss-weight", type=float, default=0.25)
    parser.add_argument("--mismatch-margin", type=float, default=0.05)
    parser.add_argument("--val-mel-score-weight", type=float, default=0.10)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["constant", "onecycle", "cosine"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
