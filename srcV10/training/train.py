from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV4.models.loss import V4MelLoss, masked_stats, speech_unit_loss, unit_frame_accuracy
from srcV10.data import V10R2INRDataset, collate_v10, infer_av_feature_dim, split_cache_files
from srcV10.models import V10PretrainedFusionSpeechModel
from srcV10.utils import batch_to_device, get_device, seed_everything, unwrap_model, write_json


def make_loader(args, files: list[Path], batch_size: int, random_crop: bool, shuffle: bool) -> DataLoader:
    ds = V10R2INRDataset(
        args.data_dir,
        files=files,
        max_frames=args.max_frames,
        random_crop=random_crop,
        seed=args.seed,
        av_feature_dir=args.av_feature_dir if args.use_avhubert_features else None,
        require_av_features=args.require_av_features,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_v10,
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
        raise RuntimeError("Could not compute mel stats from empty loader.")
    mean = total / max(1, count)
    std = ((sq / max(1, count)) - mean.pow(2)).clamp_min(1e-6).sqrt().clamp_min(0.05)
    return mean.detach().cpu(), std.detach().cpu()


@torch.no_grad()
def mean_baseline(loader: DataLoader | None, criterion: V4MelLoss, mel_mean: torch.Tensor, device: torch.device) -> float | None:
    if loader is None:
        return None
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="baseline", leave=False):
        batch = batch_to_device(batch, device)
        pred = mel_mean.to(device).view(1, 1, -1).expand_as(batch["mel"])
        loss = criterion(pred, batch["mel"], batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def infer_num_units(files: list[Path]) -> int:
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if "num_speech_units" in item:
            return int(item["num_speech_units"])
        if "speech_units" in item:
            units = item["speech_units"].long()
            valid = units[units.ge(0)]
            if valid.numel():
                return int(valid.max().item()) + 1
    return 0


def teacher_prob_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    start = float(args.unit_teacher_prob)
    if not args.use_content_units or start <= 0:
        return 0.0
    decay = int(args.unit_teacher_decay_epochs)
    if decay <= 0:
        return start
    progress = min(1.0, max(0.0, (epoch - 1) / float(decay)))
    return max(float(args.unit_teacher_min), start * (1.0 - progress))


def sanitize_batch(batch: dict) -> dict:
    for key in ("video", "landmarks", "mel", "video_times", "mel_times", "av_features"):
        if key in batch and torch.is_tensor(batch[key]):
            batch[key] = torch.nan_to_num(batch[key].float(), nan=0.0, posinf=0.0, neginf=0.0)
    return batch


def configure_amp(args: argparse.Namespace, device: torch.device) -> None:
    args.amp_enabled = False
    args.amp_resolved_dtype = "off"
    args.amp_scaler_enabled = False
    if device.type != "cuda" or not args.amp:
        return

    requested = str(args.amp_dtype).lower()
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if requested in {"auto", "bf16"} and bf16_supported:
        args.amp_enabled = True
        args.amp_resolved_dtype = "bf16"
        args.amp_scaler_enabled = False
        return
    if requested == "bf16":
        print("[amp] BF16 is not supported on this CUDA device; disabling AMP for stability.")
        return
    if requested == "auto":
        print("[amp] BF16 is not supported; disabling AMP. Use --amp-dtype fp16 to force FP16.")
        return

    args.amp_enabled = True
    args.amp_resolved_dtype = "fp16"
    args.amp_scaler_enabled = True
    print("[amp] WARNING: FP16 AMP can produce non-finite losses in srcV10; BF16 or no AMP is safer.")


def amp_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if getattr(args, "amp_resolved_dtype", "off") == "bf16" else torch.float16


def build_model(args: argparse.Namespace, device: torch.device, mel_mean: torch.Tensor | None = None) -> torch.nn.Module:
    model = V10PretrainedFusionSpeechModel(
        dim=args.dim,
        spatial_tokens=args.spatial_tokens,
        num_points=args.num_landmark_points,
        dropout=args.dropout,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        num_units=args.num_units,
        use_content_units=args.use_content_units,
        unit_temperature=args.unit_temperature,
        unit_teacher_prob=args.unit_teacher_prob,
        detach_unit_condition=args.detach_unit_condition,
        detach_content_hidden=args.detach_content_hidden,
        siren_layers=args.siren_layers,
        siren_omega=args.siren_omega,
        use_avhubert_features=args.use_avhubert_features,
        av_feature_dim=args.av_feature_dim,
        av_feature_layers=args.av_feature_layers,
        av_feature_scale=args.av_feature_scale,
        use_prosody_head=args.use_prosody_head,
        use_flow_refiner=args.use_flow_refiner,
        flow_layers=args.flow_layers,
        flow_steps=args.flow_steps,
    ).to(device)
    if mel_mean is not None:
        unwrap_model(model).set_output_bias(mel_mean.to(device))
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu and args.batch_size >= torch.cuda.device_count():
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def _trainable(params) -> list[torch.nn.Parameter]:
    return [p for p in params if p.requires_grad]


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    raw = unwrap_model(model)
    groups = []

    def add(params, lr: float) -> None:
        ps = _trainable(params)
        if ps:
            groups.append({"params": ps, "lr": lr})

    add(raw.encoder.visual.parameters(), args.visual_lr or args.lr * 0.35)
    add(raw.encoder.landmarks.parameters(), args.landmark_lr or args.lr)
    add(
        list(raw.time.parameters())
        + list(raw.query.parameters())
        + list(raw.attn.parameters())
        + list(raw.cross_norm.parameters())
        + list(raw.aligned_proj.parameters())
        + list(raw.pre_refine.parameters()),
        args.lr,
    )
    if raw.use_avhubert_features:
        add(
            list(raw.av_adapter.parameters())
            + list(raw.av_frame_gate.parameters())
            + list(raw.av_frame_norm.parameters())
            + list(raw.av_global_norm.parameters()),
            args.av_feature_lr or args.lr,
        )
    if raw.use_content_units:
        add(
            list(raw.unit_head.parameters())
            + list(raw.unit_embedding.parameters())
            + list(raw.content_fusion.parameters()),
            args.unit_lr or args.lr,
        )
    if raw.use_prosody_head:
        add(raw.prosody_head.parameters(), args.prosody_lr or args.lr)
    add(raw.decoder.parameters(), args.decoder_lr or args.lr * 1.5)
    add(raw.siren_residual.parameters(), args.siren_lr or args.lr * 0.5)
    if raw.use_flow_refiner:
        add(raw.flow_refiner.parameters(), args.flow_lr or args.lr)
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.98))


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(pred.device, pred.dtype).unsqueeze(-1)
    denom = (mask_f.sum() * pred.shape[-1]).clamp_min(1.0)
    return ((pred - target).abs() * mask_f).sum() / denom


def prosody_targets(mel: torch.Tensor) -> torch.Tensor:
    mel_f = torch.nan_to_num(mel.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    energy = torch.logsumexp(mel_f, dim=-1, keepdim=True)
    bins = torch.linspace(0.0, 1.0, mel_f.shape[-1], device=mel_f.device, dtype=mel_f.dtype).view(1, 1, -1)
    centroid = (torch.softmax(mel_f, dim=-1) * bins).sum(dim=-1, keepdim=True)
    return torch.cat([energy, centroid], dim=-1)


def prosody_loss(outputs: dict[str, torch.Tensor], batch: dict, weight_centroid: float = 0.25) -> torch.Tensor:
    pred = outputs.get("prosody")
    if pred is None:
        return batch["mel"].new_tensor(0.0)
    target = prosody_targets(batch["mel"]).to(pred.device, pred.dtype)
    energy_loss = _masked_l1(pred[..., :1], target[..., :1], batch["mel_mask"])
    centroid_loss = _masked_l1(pred[..., 1:2], target[..., 1:2], batch["mel_mask"])
    return energy_loss + float(weight_centroid) * centroid_loss


def flow_loss(model: torch.nn.Module, outputs: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
    raw = unwrap_model(model)
    if not getattr(raw, "use_flow_refiner", False):
        return batch["mel"].new_tensor(0.0)
    return raw.flow_loss(outputs, batch)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args, epoch: int) -> dict[str, float]:
    model.train()
    raw = unwrap_model(model)
    raw.unit_teacher_prob = teacher_prob_for_epoch(args, epoch)
    freeze_visual = epoch <= args.freeze_visual_epochs
    for param in raw.encoder.visual.parameters():
        param.requires_grad = not freeze_visual
    if freeze_visual:
        raw.encoder.visual.eval()

    amp_enabled = bool(getattr(args, "amp_enabled", False))
    autocast_dtype = amp_torch_dtype(args)
    totals = {"loss": 0.0, "mel": 0.0, "unit": 0.0, "prosody": 0.0, "flow": 0.0, "unit_acc": 0.0}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = sanitize_batch(batch_to_device(batch, device))
        batch["return_aux"] = True
        optimizer.zero_grad(set_to_none=True)
        if freeze_visual:
            raw.encoder.visual.eval()
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=autocast_dtype):
            outputs = model(batch)
            pred = outputs["mel"] if isinstance(outputs, dict) else outputs
        with torch.amp.autocast("cuda", enabled=False):
            mel_loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
            unit_logits = outputs.get("unit_logits") if isinstance(outputs, dict) else None
            unit = speech_unit_loss(unit_logits, batch, label_smoothing=args.unit_label_smoothing)
            pros = prosody_loss(outputs, batch, weight_centroid=args.prosody_centroid_weight) if isinstance(outputs, dict) else pred.new_tensor(0.0)
            flow = flow_loss(model, outputs, batch) if isinstance(outputs, dict) else pred.new_tensor(0.0)
            loss = (
                mel_loss
                + float(args.unit_loss_weight) * unit
                + float(args.prosody_loss_weight) * pros
                + float(args.flow_loss_weight) * flow
            )
        if not torch.isfinite(loss):
            pred_finite = bool(torch.isfinite(pred).all())
            raise FloatingPointError(
                "Non-finite train loss "
                f"mel={float(mel_loss.detach().nan_to_num().cpu())} "
                f"unit={float(unit.detach().nan_to_num().cpu())} "
                f"prosody={float(pros.detach().nan_to_num().cpu())} "
                f"flow={float(flow.detach().nan_to_num().cpu())} "
                f"pred_finite={pred_finite} "
                f"amp={getattr(args, 'amp_resolved_dtype', 'off')} "
                f"paths={batch.get('paths', [])[:4]}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        totals["mel"] += float(mel_loss.detach().cpu())
        totals["unit"] += float(unit.detach().cpu())
        totals["prosody"] += float(pros.detach().cpu())
        totals["flow"] += float(flow.detach().cpu())
        totals["unit_acc"] += unit_frame_accuracy(unit_logits, batch)
        count += 1
    denom = max(1, count)
    out = {key: value / denom for key, value in totals.items()}
    out["teacher_prob"] = float(raw.unit_teacher_prob)
    out["visual_frozen"] = float(freeze_visual)
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device, args) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0.0
    count = 0
    stats = {}
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = sanitize_batch(batch_to_device(batch, device))
        batch["return_aux"] = True
        outputs = model(batch)
        pred = outputs["mel"] if isinstance(outputs, dict) else outputs
        pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        mel_loss = criterion(pred, batch["mel"].float(), batch["mel_mask"])
        unit_logits = outputs.get("unit_logits") if isinstance(outputs, dict) else None
        unit = speech_unit_loss(unit_logits, batch, label_smoothing=args.unit_label_smoothing)
        pros = prosody_loss(outputs, batch, weight_centroid=args.prosody_centroid_weight) if isinstance(outputs, dict) else pred.new_tensor(0.0)
        flow = flow_loss(model, outputs, batch) if isinstance(outputs, dict) else pred.new_tensor(0.0)
        loss = (
            mel_loss
            + float(args.unit_loss_weight) * unit
            + float(args.prosody_loss_weight) * pros
            + float(args.flow_loss_weight) * flow
        )
        total += float(loss.detach().cpu())
        count += 1
        if not stats:
            stats = masked_stats(pred, batch["mel"], batch["mel_mask"])
            stats["mel_loss"] = float(mel_loss.detach().cpu())
            stats["unit_loss"] = float(unit.detach().cpu())
            stats["prosody_loss"] = float(pros.detach().cpu())
            stats["flow_loss"] = float(flow.detach().cpu())
            stats["unit_acc"] = unit_frame_accuracy(unit_logits, batch)
    return total / max(1, count), stats


def save_checkpoint(path, model, optimizer, epoch, best, args, mel_mean, mel_std) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "best": float(best),
            "config": vars(args),
            "mel_mean": mel_mean.detach().cpu(),
            "mel_std": mel_std.detach().cpu(),
            "model_type": "srcV10_pretrained_fusion",
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train srcV10 pretrained-feature conditioned speech reconstruction.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2_10k")
    parser.add_argument("--output-dir", default="checkpoints_srcV10_pretrained_fusion")
    parser.add_argument("--av-feature-dir", default="", help="Optional directory produced by srcV8 cache_avhubert_features.py.")
    parser.add_argument("--use-avhubert-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-av-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--random-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-tokens", type=int, default=4)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--siren-layers", type=int, default=2)
    parser.add_argument("--siren-omega", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--av-feature-dim", type=int, default=0)
    parser.add_argument("--av-feature-layers", type=int, default=1)
    parser.add_argument("--av-feature-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--visual-lr", type=float, default=0.0)
    parser.add_argument("--landmark-lr", type=float, default=0.0)
    parser.add_argument("--decoder-lr", type=float, default=0.0)
    parser.add_argument("--siren-lr", type=float, default=0.0)
    parser.add_argument("--unit-lr", type=float, default=0.0)
    parser.add_argument("--av-feature-lr", type=float, default=0.0)
    parser.add_argument("--prosody-lr", type=float, default=0.0)
    parser.add_argument("--flow-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-visual-epochs", type=int, default=8)
    parser.add_argument("--lambda-mel", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.15)
    parser.add_argument("--lambda-delta2", type=float, default=0.03)
    parser.add_argument("--lambda-energy", type=float, default=0.02)
    parser.add_argument("--lambda-mr-spectral", type=float, default=0.25)
    parser.add_argument("--shift-window", type=int, default=0)
    parser.add_argument("--use-content-units", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-units", type=int, default=0)
    parser.add_argument("--unit-loss-weight", type=float, default=0.25)
    parser.add_argument("--unit-label-smoothing", type=float, default=0.05)
    parser.add_argument("--unit-temperature", type=float, default=1.0)
    parser.add_argument("--unit-teacher-prob", type=float, default=0.0)
    parser.add_argument("--unit-teacher-decay-epochs", type=int, default=20)
    parser.add_argument("--unit-teacher-min", type=float, default=0.05)
    parser.add_argument("--detach-unit-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detach-content-hidden", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-prosody-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prosody-loss-weight", type=float, default=0.05)
    parser.add_argument("--prosody-centroid-weight", type=float, default=0.25)
    parser.add_argument("--use-flow-refiner", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-loss-weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    configure_amp(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.av_feature_dir:
        args.use_avhubert_features = False
    elif args.use_avhubert_features and args.av_feature_dim <= 0:
        args.av_feature_dim = infer_av_feature_dim(args.av_feature_dir)
        if args.av_feature_dim <= 0:
            print(f"[avhubert] no usable feature cache found under {args.av_feature_dir}; disabling feature branch")
            args.use_avhubert_features = False
    elif not args.use_avhubert_features:
        args.av_feature_dir = ""
    if not args.use_avhubert_features:
        args.require_av_features = False
        args.av_feature_dim = max(1, args.av_feature_dim or 768)

    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, limit_files=limit_files)
    if args.use_content_units and args.num_units <= 0:
        args.num_units = infer_num_units(train_files)
        if args.num_units <= 0:
            print("[content-units] speech_units not found; disabling content units")
            args.use_content_units = False

    train_loader = make_loader(args, train_files, args.batch_size, random_crop=args.random_crop, shuffle=True)
    val_loader = (
        make_loader(args, val_files, args.val_batch_size or args.batch_size, random_crop=False, shuffle=False)
        if val_files
        else None
    )
    stats_loader = make_loader(args, train_files, args.batch_size, random_crop=False, shuffle=False)
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
    scaler = torch.amp.GradScaler("cuda", enabled=bool(getattr(args, "amp_scaler_enabled", False)))
    mean_train = mean_baseline(stats_loader, criterion, mel_mean, device)
    mean_val = mean_baseline(val_loader, criterion, mel_mean, device)

    print(f"[device] {device}")
    print(f"[amp] enabled={args.amp_enabled} dtype={args.amp_resolved_dtype} scaler={args.amp_scaler_enabled}")
    print(f"[data] train={len(train_files)} val={len(val_files)} max_frames={args.max_frames}")
    print(
        "[model] srcV10 V6Align+AVHubertFeatures+Prosody+Flow "
        f"dim={args.dim} units={args.num_units if args.use_content_units else 'off'} "
        f"av={'on' if args.use_avhubert_features else 'off'} flow={'on' if args.use_flow_refiner else 'off'}"
    )
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={mean_val if mean_val is not None else 'n/a'}")

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args, epoch)
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
            "train": train,
            "train_eval": train_eval,
            "val": val_loss,
            "best": best,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        history.append(row)
        write_json(output_dir / "history.json", {"history": history, "config": vars(args)})
        val_txt = f"{val_loss:.6f}" if val_loss is not None else "n/a"
        unit_txt = ""
        if args.use_content_units:
            unit_txt = f" unit={train['unit']:.4f} acc={train['unit_acc']:.3f} teacher={train['teacher_prob']:.2f}"
            if val_stats:
                unit_txt += f" val_unit={val_stats.get('unit_loss', 0.0):.4f} val_acc={val_stats.get('unit_acc', 0.0):.3f}"
        print(
            f"[epoch {epoch:04d}] train={train['loss']:.6f} mel={train['mel']:.6f}"
            f"{unit_txt} pros={train['prosody']:.4f} flow={train['flow']:.4f} "
            f"train_eval={train_eval:.6f} val={val_txt} best={best:.6f} "
            f"std_r={(val_stats or train_stats).get('std_ratio', 0.0):.3f} "
            f"del_r={(val_stats or train_stats).get('delta_ratio', 0.0):.3f}"
            f"{' best' if is_best else ''}"
        )


if __name__ == "__main__":
    run(parse_args())
