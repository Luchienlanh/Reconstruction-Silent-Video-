from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from srcV2.models import MaskedMelLoss
from srcV2.training.common import (
    build_model,
    compute_mel_stats,
    init_decoder_output_bias,
    load_checkpoint,
    make_loader,
    mean_baseline,
    masked_stats,
    model_inputs,
    save_checkpoint,
    split_cache_files,
    write_history,
)
from srcV2.utils.common import batch_to_device, get_device, seed_everything
from srcV2.utils.plotting import save_mel_comparison


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
        if key in batch and torch.is_tensor(batch[key]):
            if not torch.isfinite(batch[key]).all():
                paths = batch.get("paths", [])
                print(f"[warn] non-finite {key}; paths={paths[:4]}")
                batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=0.0, neginf=0.0)
    return batch


def configure_train_decoder(model, args) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    decoder = raw_model.decoder
    if args.time_direct_scale is not None:
        with torch.no_grad():
            decoder.time_direct_scale.fill_(float(args.time_direct_scale))
    if args.time_conditioned_scale is not None:
        with torch.no_grad():
            decoder.time_conditioned_scale.fill_(float(args.time_conditioned_scale))
    if args.disable_time_direct:
        with torch.no_grad():
            decoder.time_direct_scale.zero_()
        decoder.time_direct_scale.requires_grad = False
        for param in decoder.time_direct.parameters():
            param.requires_grad = False
    if args.freeze_time_direct:
        decoder.time_direct_scale.requires_grad = False
        for param in decoder.time_direct.parameters():
            param.requires_grad = False


def clone_batch(batch: dict) -> dict:
    return {key: value.clone() if torch.is_tensor(value) else deepcopy(value) for key, value in batch.items()}


def slice_batch(batch: dict, size: int) -> dict:
    if size <= 0:
        return batch
    batch_size = None
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            batch_size = int(value.shape[0])
            break
    if batch_size is None or size >= batch_size:
        return batch
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            out[key] = value[:size]
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = value[:size]
        else:
            out[key] = value
    return out


def zero_video_inputs(batch: dict) -> dict:
    out = clone_batch(batch)
    if "video" in out:
        out["video"].zero_()
    return out


def zero_visual_inputs(batch: dict) -> dict:
    out = zero_video_inputs(batch)
    if "landmarks" in out:
        out["landmarks"].zero_()
    if "mouth_motion" in out:
        out["mouth_motion"].zero_()
    if "mouth_valid_mask" in out:
        out["mouth_valid_mask"].zero_()
    return out


def mismatch_visual_inputs(batch: dict) -> dict | None:
    if "video" not in batch or batch["video"].shape[0] < 2:
        return None
    out = clone_batch(batch)
    for key in ("video", "landmarks", "mouth_motion", "mouth_valid_mask"):
        if key in out and torch.is_tensor(out[key]):
            out[key] = torch.roll(out[key], shifts=1, dims=0)
    return out


def normalized_masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, criterion: MaskedMelLoss) -> torch.Tensor:
    a = criterion._normalize(a.float())
    b = criterion._normalize(b.float())
    mask_f = mask.to(a.device, dtype=a.dtype).unsqueeze(-1)
    denom = (mask_f.sum() * a.shape[-1]).clamp_min(1.0)
    return ((a - b).abs() * mask_f).sum() / denom


def set_visual_trainability(raw_model, trainable: bool) -> None:
    for param in raw_model.encoder.visual.parameters():
        param.requires_grad = trainable
    if trainable:
        raw_model.encoder.visual.train()
    else:
        raw_model.encoder.visual.eval()


def set_decoder_bias_trainability(raw_model, trainable: bool) -> None:
    coarse = getattr(raw_model.decoder, "coarse", None)
    if coarse is None:
        return
    last = coarse[-1]
    if hasattr(last, "bias") and last.bias is not None:
        last.bias.requires_grad = trainable


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args, epoch):
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    landmark_pretrain = epoch <= args.landmark_pretrain_epochs
    freeze_visual = epoch <= max(args.freeze_visual_epochs, args.landmark_pretrain_epochs)
    freeze_decoder_bias = epoch <= args.freeze_decoder_bias_epochs
    set_visual_trainability(raw_model, not freeze_visual)
    set_decoder_bias_trainability(raw_model, not freeze_decoder_bias)

    criterion.set_shift_window(args.shift_warmup if epoch <= args.shift_warmup_epochs else args.shift_final)
    total = 0.0
    total_mel = 0.0
    total_stats = 0.0
    total_mismatch = 0.0
    total_sensitivity = 0.0
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
            if freeze_decoder_bias:
                set_decoder_bias_trainability(raw_model, False)
            train_batch = zero_video_inputs(batch) if landmark_pretrain else batch
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(model_inputs(train_batch), return_aux=args.lambda_stats > 0)
                pred = out["mel"] if isinstance(out, dict) else out
            with torch.amp.autocast("cuda", enabled=False):
                if not torch.isfinite(pred).all():
                    print(f"[warn] non-finite pred; paths={batch.get('paths', [])[:4]}")
                    pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                mel_loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
                loss = mel_loss
                stats_loss = pred.new_tensor(0.0)
                mismatch_loss = pred.new_tensor(0.0)
                sensitivity_loss = pred.new_tensor(0.0)
                if args.lambda_stats > 0:
                    stats_target = target_mel_stats(batch["mel"], batch["mel_mask"], criterion)
                    stats_loss = torch.nn.functional.smooth_l1_loss(out["mel_stats"].float(), stats_target.float())
                    loss = loss + args.lambda_stats * stats_loss
                if args.lambda_visual_mismatch > 0 and epoch >= args.visual_mismatch_start_epoch:
                    aux_train_batch = slice_batch(train_batch, args.visual_aux_batch_size)
                    aux_target_batch = slice_batch(batch, args.visual_aux_batch_size)
                    bad_batch = mismatch_visual_inputs(aux_train_batch)
                    if bad_batch is not None:
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            pred_bad = model(model_inputs(bad_batch))
                        if torch.isfinite(pred_bad).all():
                            aux_n = int(pred_bad.shape[0])
                            matched_aux_loss = criterion(
                                pred[:aux_n].float(),
                                aux_target_batch["mel"].float(),
                                aux_target_batch["mel_mask"],
                            ).detach()
                            bad_target_loss = criterion(
                                pred_bad.float(),
                                aux_target_batch["mel"].float(),
                                aux_target_batch["mel_mask"],
                            )
                            mismatch_loss = F.relu(args.visual_mismatch_margin + matched_aux_loss - bad_target_loss)
                            loss = loss + args.lambda_visual_mismatch * mismatch_loss
                if args.lambda_visual_sensitivity > 0 and epoch >= args.visual_sensitivity_start_epoch:
                    aux_train_batch = slice_batch(train_batch, args.visual_aux_batch_size)
                    aux_target_batch = slice_batch(batch, args.visual_aux_batch_size)
                    zero_batch = zero_visual_inputs(aux_train_batch)
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
                        pred_zero = model(model_inputs(zero_batch))
                    if torch.isfinite(pred_zero).all():
                        aux_n = int(pred_zero.shape[0])
                        sensitivity = normalized_masked_l1(
                            pred[:aux_n].float(),
                            pred_zero.float(),
                            aux_target_batch["mel_mask"],
                            criterion,
                        )
                        sensitivity_loss = F.relu(args.visual_sensitivity_margin - sensitivity)
                        loss = loss + args.lambda_visual_sensitivity * sensitivity_loss
                if args.lambda_time_direct_scale > 0:
                    raw_model = model.module if hasattr(model, "module") else model
                    scale_loss = raw_model.decoder.time_direct_scale.float().pow(2)
                    loss = loss + args.lambda_time_direct_scale * scale_loss
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
            total_mismatch += float(mismatch_loss.detach().cpu())
            total_sensitivity += float(sensitivity_loss.detach().cpu())
            count += 1
    denom = max(1, count)
    return {
        "loss": total / denom,
        "mel": total_mel / denom,
        "stats": total_stats / denom,
        "mismatch": total_mismatch / denom,
        "sensitivity": total_sensitivity / denom,
        "visual_frozen": freeze_visual,
        "landmark_pretrain": landmark_pretrain,
        "decoder_bias_frozen": freeze_decoder_bias,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, plot_path=None, epoch=0, max_batches=0):
    model.eval()
    criterion.set_shift_window(0)
    total = 0.0
    count = 0
    first_stats = None
    plotted = False
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = batch_to_device(batch, device)
        batch = sanitize_batch(batch)
        pred = model(model_inputs(batch))
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
        windowed=args.windowed,
        window_frames=args.window_frames,
        hop_frames=args.hop_frames,
        max_windows_per_file=args.max_windows_per_file,
        cache_size=args.dataset_cache_size,
        audio_target_shift_frames=args.audio_target_shift_frames,
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
        windowed=args.windowed,
        window_frames=args.window_frames,
        hop_frames=args.hop_frames,
        max_windows_per_file=args.val_max_windows_per_file or args.max_windows_per_file,
        cache_size=args.dataset_cache_size,
        audio_target_shift_frames=args.audio_target_shift_frames,
    ) if val_files else None
    train_eval_loader = make_loader(
        args.data_dir,
        train_files,
        args.val_batch_size or args.batch_size,
        max_frames=args.max_frames,
        random_crop=False,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False,
        windowed=args.windowed,
        window_frames=args.window_frames,
        hop_frames=args.hop_frames,
        max_windows_per_file=args.val_max_windows_per_file or args.max_windows_per_file,
        cache_size=args.dataset_cache_size,
        audio_target_shift_frames=args.audio_target_shift_frames,
    ) if args.eval_train_every > 0 else None
    stats_loader = make_loader(
        args.data_dir,
        train_files,
        args.batch_size,
        max_frames=args.max_frames,
        random_crop=False,
        seed=args.seed,
        num_workers=args.num_workers,
        windowed=args.windowed,
        window_frames=args.window_frames,
        hop_frames=args.hop_frames,
        max_windows_per_file=args.stats_max_windows_per_file or args.max_windows_per_file,
        cache_size=args.dataset_cache_size,
        audio_target_shift_frames=args.audio_target_shift_frames,
    )
    mel_mean, mel_std = compute_mel_stats(stats_loader, device, max_batches=args.max_stats_batches)
    criterion = MaskedMelLoss(
        mel_mean,
        mel_std,
        lambda_mel=args.lambda_mel,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
        lambda_band=args.lambda_band,
        lambda_energy_delta=args.lambda_energy_delta,
        lambda_variance=args.lambda_variance,
        band_bins=args.band_bins,
    ).to(device)

    model = build_model(device, args)
    configure_train_decoder(model, args)
    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    else:
        init_decoder_output_bias(model, mel_mean)
    trainable_model = model.module if hasattr(model, "module") else model
    fusion_params = (
        list(trainable_model.encoder.time.parameters())
        + list(trainable_model.encoder.fuse.parameters())
        + list(trainable_model.encoder.norm.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_model.encoder.visual.parameters(), "lr": args.lr * args.visual_lr_scale},
            {"params": trainable_model.encoder.landmarks.parameters(), "lr": args.lr * args.landmark_lr_scale},
            {"params": trainable_model.encoder.motion.parameters(), "lr": args.lr * args.motion_lr_scale},
            {"params": fusion_params, "lr": args.lr * args.fusion_lr_scale},
            {"params": trainable_model.mel_stats_head.parameters(), "lr": args.lr * args.fusion_lr_scale},
            {"params": trainable_model.decoder.parameters(), "lr": args.lr * args.decoder_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    mean_train = mean_baseline(train_loader, criterion, mel_mean, device, max_batches=args.max_baseline_batches)
    mean_val = mean_baseline(
        val_loader,
        criterion,
        mel_mean,
        device,
        max_batches=args.max_baseline_batches,
    ) if val_loader is not None else None
    print(f"[device] {device}")
    print(f"[data] train_files={len(train_files)} val_files={len(val_files)} train_items={len(train_loader.dataset)}")
    if args.windowed:
        print(f"[window] frames={args.window_frames} hop={args.hop_frames} max_windows_per_file={args.max_windows_per_file}")
    print(f"[alignment] audio_target_shift_frames={args.audio_target_shift_frames} shift_warmup={args.shift_warmup} shift_final={args.shift_final}")
    print(
        "[visual-conditioning] "
        f"landmark_pretrain_epochs={args.landmark_pretrain_epochs} freeze_visual_epochs={args.freeze_visual_epochs} "
        f"freeze_decoder_bias_epochs={args.freeze_decoder_bias_epochs} mismatch={args.lambda_visual_mismatch} "
        f"sensitivity={args.lambda_visual_sensitivity} aux_batch={args.visual_aux_batch_size}"
    )
    print(
        "[lr-scale] "
        f"visual={args.visual_lr_scale} landmark={args.landmark_lr_scale} motion={args.motion_lr_scale} "
        f"fusion={args.fusion_lr_scale} decoder={args.decoder_lr_scale}"
    )
    print(f"[model] r2plus1d_inr dim={args.dim} spatial_tokens={args.spatial_tokens}")
    print(
        "[loss] "
        f"mel={args.lambda_mel} band={args.lambda_band} delta={args.lambda_delta} "
        f"delta2={args.lambda_delta2} energy={args.lambda_energy} "
        f"energy_delta={args.lambda_energy_delta} variance={args.lambda_variance}"
    )
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={'n/a' if mean_val is None else f'{mean_val:.6f}'}")

    best = float("inf")
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args, epoch)
        train_loss = train_metrics["loss"]
        should_eval_val = val_loader is not None and (
            epoch == 1
            or epoch == args.epochs
            or args.eval_every <= 1
            or epoch % args.eval_every == 0
        )
        val_loss, stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args,
            plot_path=output_dir / f"mel_epoch_{epoch:04d}.png" if should_eval_val and epoch % args.plot_every == 0 else None,
            epoch=epoch,
            max_batches=args.max_val_batches,
        ) if should_eval_val else (None, {})
        train_eval_loss = None
        if train_eval_loader is not None and (epoch == 1 or epoch % args.eval_train_every == 0 or epoch == args.epochs):
            train_eval_loss, _ = evaluate(
                model,
                train_eval_loader,
                criterion,
                device,
                args,
                max_batches=args.max_train_eval_batches or args.max_val_batches,
            )
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
            "train_mismatch": train_metrics["mismatch"],
            "train_sensitivity": train_metrics["sensitivity"],
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
        if train_metrics["landmark_pretrain"]:
            phase_text += " landmark_pretrain"
        if train_metrics["decoder_bias_frozen"]:
            phase_text += " bias=frozen"
        aux_text = f" mel={train_metrics['mel']:.6f}"
        if args.lambda_stats > 0:
            aux_text += f" stats={train_metrics['stats']:.6f}"
        if args.lambda_visual_mismatch > 0:
            aux_text += f" mismatch={train_metrics['mismatch']:.6f}"
        if args.lambda_visual_sensitivity > 0:
            aux_text += f" sens={train_metrics['sensitivity']:.6f}"
        stat_text = "" if not stats else f" std_r={stats.get('std_ratio', 0):.3f} del_r={stats.get('delta_ratio', 0):.3f}"
        print(f"[epoch {epoch:04d}] train={train_loss:.6f}{aux_text}{train_eval_text} val={val_text} best={best:.6f}{gap}{stat_text}{phase_text}{' best' if is_best else ''}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train improved srcV2 ResNet2+1D + lip-motion-aware mel decoder.")
    parser.add_argument("--data-dir", default="../Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--output-dir", default="checkpoints_r2inr_frontal_v2")
    parser.add_argument("--resume", default=None, help="Resume model weights from a srcV2 checkpoint. Optimizer is reset for staged training.")
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
    parser.add_argument("--windowed", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--hop-frames", type=int, default=10)
    parser.add_argument("--max-windows-per-file", type=int, default=12)
    parser.add_argument("--val-max-windows-per-file", type=int, default=2)
    parser.add_argument("--stats-max-windows-per-file", type=int, default=6)
    parser.add_argument("--dataset-cache-size", type=int, default=2)
    parser.add_argument("--audio-target-shift-frames", type=int, default=4, help="Use mel[t+shift] as the target for visual time t. Positive values compensate audio that lags the lips.")
    parser.add_argument("--eval-every", type=int, default=3, help="Run validation every N epochs; epoch 1 and final epoch are always evaluated.")
    parser.add_argument("--max-val-batches", type=int, default=160)
    parser.add_argument("--max-train-eval-batches", type=int, default=0)
    parser.add_argument("--max-stats-batches", type=int, default=256)
    parser.add_argument("--max-baseline-batches", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--visual-lr-scale", type=float, default=1.5)
    parser.add_argument("--landmark-lr-scale", type=float, default=3.0)
    parser.add_argument("--motion-lr-scale", type=float, default=4.0)
    parser.add_argument("--fusion-lr-scale", type=float, default=2.0)
    parser.add_argument("--decoder-lr-scale", type=float, default=1.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--steps-per-batch", type=int, default=1, help="Repeat optimizer updates on each batch; useful for tiny limit-overfit tests.")
    parser.add_argument("--landmark-pretrain-epochs", type=int, default=5, help="Train first on landmarks/mouth-motion by zeroing raw video and freezing the visual tower.")
    parser.add_argument("--freeze-visual-epochs", type=int, default=5, help="Freeze only the visual R2+1D tower for the first N epochs.")
    parser.add_argument("--freeze-decoder-bias-epochs", type=int, default=5, help="Keep the mel-mean output bias fixed early so the decoder cannot solve training by bias drift.")
    parser.add_argument("--lambda-stats", type=float, default=0.0, help="Auxiliary loss weight for predicting per-clip normalized mel mean/std from encoder global token.")
    parser.add_argument("--disable-time-direct", default=True, action=argparse.BooleanOptionalAction, help="Disable sample-agnostic time->mel branch; default on so predictions must use lip motion.")
    parser.add_argument("--freeze-time-direct", action="store_true", help="Freeze time-direct branch parameters while keeping its configured scale.")
    parser.add_argument("--time-direct-scale", type=float, default=None)
    parser.add_argument("--time-conditioned-scale", type=float, default=None)
    parser.add_argument("--lambda-time-direct-scale", type=float, default=0.0)
    parser.add_argument("--lambda-visual-mismatch", type=float, default=0.20, help="Ranking loss that makes mismatched visual inputs score worse than matched inputs.")
    parser.add_argument("--visual-mismatch-margin", type=float, default=0.08)
    parser.add_argument("--visual-mismatch-start-epoch", type=int, default=1)
    parser.add_argument("--lambda-visual-sensitivity", type=float, default=0.05, help="Encourage predictions to change when all visual/landmark evidence is removed.")
    parser.add_argument("--visual-sensitivity-margin", type=float, default=0.05)
    parser.add_argument("--visual-sensitivity-start-epoch", type=int, default=1)
    parser.add_argument("--visual-aux-batch-size", type=int, default=2, help="Run mismatch/sensitivity losses on this many samples to keep VRAM bounded.")
    parser.add_argument("--eval-train-every", type=int, default=0, help="Evaluate deterministic train-set loss every N epochs.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop-last", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-tokens", type=int, default=4)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--motion-dim", type=int, default=19)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-mel", type=float, default=0.50)
    parser.add_argument("--lambda-delta", type=float, default=0.25)
    parser.add_argument("--lambda-delta2", type=float, default=0.05)
    parser.add_argument("--lambda-energy", type=float, default=0.05)
    parser.add_argument("--lambda-band", type=float, default=0.35, help="Low-resolution spectral-envelope loss; less speaker/pitch-specific than full-bin L1.")
    parser.add_argument("--lambda-energy-delta", type=float, default=0.15)
    parser.add_argument("--lambda-variance", type=float, default=0.15, help="Penalize collapsed/flat mel dynamics by matching per-window normalized std.")
    parser.add_argument("--band-bins", type=int, default=8)
    parser.add_argument("--shift-warmup", type=int, default=6)
    parser.add_argument("--shift-final", type=int, default=0)
    parser.add_argument("--shift-warmup-epochs", type=int, default=15)
    parser.add_argument("--plot-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
