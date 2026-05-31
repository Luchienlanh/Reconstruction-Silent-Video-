from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV2.data import R2INRDataset, collate_r2inr
from srcV2.models import (
    ContextAlignLipToSpeechModel,
    ContextDetailLipToSpeechModel,
    ContextMotionDetailLipToSpeechModel,
    ContextSyncLipToSpeechModel,
    ContentUnitLipToSpeechModel,
    MaskedMelLoss,
    MotionTCNLipToSpeechModel,
    SimpleLipToSpeechModel,
)
from srcV2.training.common import build_model, load_checkpoint, masked_stats, model_inputs
from srcV2.utils.common import batch_to_device, get_device, seed_everything


def apply_variant(batch: dict, variant: str) -> dict:
    out = {k: v.clone() if torch.is_tensor(v) else deepcopy(v) for k, v in batch.items()}
    if variant == "normal":
        return out
    if variant in {"zero_video", "zero_both"}:
        out["video"].zero_()
    if variant in {"zero_landmarks", "zero_both"}:
        out["landmarks"].zero_()
        out["mouth_valid_mask"].zero_()
    if variant == "reverse_time":
        out["video"] = torch.flip(out["video"], dims=[2])
        out["landmarks"] = torch.flip(out["landmarks"], dims=[1])
        out["mouth_valid_mask"] = torch.flip(out["mouth_valid_mask"], dims=[1])
    if variant == "mismatch_sample":
        if out["video"].shape[0] < 2:
            return out
        out["video"] = torch.roll(out["video"], shifts=1, dims=0)
        out["landmarks"] = torch.roll(out["landmarks"], shifts=1, dims=0)
        out["mouth_valid_mask"] = torch.roll(out["mouth_valid_mask"], shifts=1, dims=0)
    return out


def unit_metrics(outputs: dict, batch: dict) -> tuple[float, float]:
    logits = outputs.get("unit_logits") if isinstance(outputs, dict) else None
    if logits is None or "speech_units" not in batch:
        return 0.0, 0.0
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    targets = batch["speech_units"].long()
    if targets.shape[1] != logits.shape[1]:
        x = targets.float().unsqueeze(1)
        targets = F.interpolate(x, size=logits.shape[1], mode="nearest").squeeze(1).long()
    mask = targets.ge(0)
    if "mel_mask" in batch and batch["mel_mask"].shape[1] == logits.shape[1]:
        mask = mask & batch["mel_mask"].to(mask.device, dtype=torch.bool)
    if not bool(mask.any()):
        return 0.0, 0.0
    unit_loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)
    pred = logits.argmax(dim=-1)
    unit_acc = (pred.eq(targets) & mask).float().sum() / mask.float().sum().clamp_min(1.0)
    return float(unit_loss.detach().cpu()), float(unit_acc.detach().cpu())


@torch.no_grad()
def evaluate_variant(model, loader, criterion, device, variant: str, max_batches: int):
    total = 0.0
    count = 0
    first_stats = None
    for batch_idx, batch in enumerate(tqdm(loader, desc=variant, leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = batch_to_device(batch, device)
        vbatch = apply_variant(batch, variant)
        inputs = model_inputs(vbatch)
        inputs["return_aux"] = True
        outputs = model(inputs)
        pred = outputs["mel"] if isinstance(outputs, dict) else outputs
        loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
        if first_stats is None:
            first_stats = masked_stats(pred, batch["mel"], batch["mel_mask"])
            first_stats["unit_loss"], first_stats["unit_acc"] = unit_metrics(outputs, batch)
    return total / max(1, count), (first_stats or {})


def run(args) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    ds = R2INRDataset(args.data_dir, max_frames=args.max_frames, random_crop=False, seed=args.seed, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_r2inr)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_type = args.model_type
    if model_type == "auto":
        model_type = cfg.get("model_type", "r2inr")
    for key in ("dim", "spatial_tokens", "num_landmark_points", "dropout", "multi_gpu"):
        if hasattr(args, key):
            setattr(args, key, cfg.get(key.replace("_", "-"), cfg.get(key, getattr(args, key))))
    if model_type == "simple":
        model = SimpleLipToSpeechModel(
            dim=args.dim,
            spatial_tokens=args.spatial_tokens,
            num_points=args.num_landmark_points,
            dropout=args.dropout,
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", False):
            model = torch.nn.DataParallel(model)
    elif model_type == "motion_tcn":
        model = MotionTCNLipToSpeechModel(
            dim=args.dim,
            spatial_tokens=args.spatial_tokens,
            num_points=args.num_landmark_points,
            dropout=args.dropout,
            decoder_layers=cfg.get("decoder_layers", getattr(args, "decoder_layers", 8)),
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", False):
            model = torch.nn.DataParallel(model)
    elif model_type == "context_sync":
        model = ContextSyncLipToSpeechModel(
            dim=args.dim,
            spatial_tokens=args.spatial_tokens,
            num_points=args.num_landmark_points,
            dropout=args.dropout,
            encoder_layers=cfg.get("encoder_layers", getattr(args, "encoder_layers", 3)),
            decoder_layers=cfg.get("decoder_layers", getattr(args, "decoder_layers", 4)),
            heads=cfg.get("heads", getattr(args, "heads", 4)),
            num_units=cfg.get("num_units", getattr(args, "num_units", 0))
            if cfg.get("unit_loss_weight", 0.0) > 0
            else 0,
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", False):
            model = torch.nn.DataParallel(model)
    elif model_type == "context_align":
        model = ContextAlignLipToSpeechModel(
            dim=args.dim,
            spatial_tokens=args.spatial_tokens,
            num_points=args.num_landmark_points,
            dropout=args.dropout,
            encoder_layers=cfg.get("encoder_layers", getattr(args, "encoder_layers", 2)),
            decoder_layers=cfg.get("decoder_layers", getattr(args, "decoder_layers", 2)),
            heads=cfg.get("heads", getattr(args, "heads", 4)),
            num_units=cfg.get("num_units", getattr(args, "num_units", 0))
            if cfg.get("unit_loss_weight", 0.0) > 0
            else 0,
            align_window=cfg.get("align_window", getattr(args, "align_window", 0.18)),
            modality_dropout=cfg.get("modality_dropout", getattr(args, "modality_dropout", 0.0)),
            detail_scale=cfg.get("detail_scale", getattr(args, "detail_scale", 0.35)),
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", False):
            model = torch.nn.DataParallel(model)
    elif model_type == "context_detail":
        model = ContextDetailLipToSpeechModel(
            dim=args.dim,
            spatial_tokens=args.spatial_tokens,
            num_points=args.num_landmark_points,
            dropout=args.dropout,
            encoder_layers=cfg.get("encoder_layers", getattr(args, "encoder_layers", 2)),
            decoder_layers=cfg.get("decoder_layers", getattr(args, "decoder_layers", 2)),
            heads=cfg.get("heads", getattr(args, "heads", 4)),
            num_units=cfg.get("num_units", getattr(args, "num_units", 0))
            if cfg.get("unit_loss_weight", 0.0) > 0
            else 0,
            detail_scale=cfg.get("detail_scale", getattr(args, "detail_scale", 0.35)),
            detail_layers=cfg.get("detail_layers", getattr(args, "detail_layers", 3)),
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", False):
            model = torch.nn.DataParallel(model)
    elif model_type == "context_motion_detail":
        model = ContextMotionDetailLipToSpeechModel(
            dim=args.dim,
            spatial_tokens=args.spatial_tokens,
            num_points=args.num_landmark_points,
            dropout=args.dropout,
            encoder_layers=cfg.get("encoder_layers", getattr(args, "encoder_layers", 2)),
            decoder_layers=cfg.get("decoder_layers", getattr(args, "decoder_layers", 2)),
            heads=cfg.get("heads", getattr(args, "heads", 4)),
            num_units=cfg.get("num_units", getattr(args, "num_units", 0))
            if cfg.get("unit_loss_weight", 0.0) > 0
            else 0,
            detail_scale=cfg.get("detail_scale", getattr(args, "detail_scale", 0.35)),
            detail_layers=cfg.get("detail_layers", getattr(args, "detail_layers", 3)),
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", False):
            model = torch.nn.DataParallel(model)
    elif model_type == "content_unit":
        model = ContentUnitLipToSpeechModel(
            dim=args.dim,
            spatial_tokens=args.spatial_tokens,
            num_points=args.num_landmark_points,
            dropout=args.dropout,
            encoder_layers=cfg.get("encoder_layers", getattr(args, "encoder_layers", 2)),
            decoder_layers=cfg.get("decoder_layers", getattr(args, "decoder_layers", 2)),
            heads=cfg.get("heads", getattr(args, "heads", 4)),
            num_units=cfg.get("num_units", getattr(args, "num_units", 50)),
            unit_temperature=cfg.get("unit_temperature", getattr(args, "unit_temperature", 1.0)),
            detach_unit_condition=cfg.get("detach_unit_condition", getattr(args, "detach_unit_condition", True)),
            detach_content_hidden=cfg.get("detach_content_hidden", getattr(args, "detach_content_hidden", True)),
            unit_teacher_prob=0.0,
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1 and getattr(args, "multi_gpu", False):
            model = torch.nn.DataParallel(model)
    else:
        model = build_model(device, args)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    criterion = MaskedMelLoss(ckpt["mel_mean"], ckpt["mel_std"]).to(device)

    print(f"[checkpoint] {args.checkpoint}")
    print(f"[model_type] {model_type}")
    print(f"[data] {args.data_dir} samples={len(ds)}")
    extra = f" {'unit':>10} {'u_acc':>10}" if args.include_units else ""
    print(f"{'variant':<18} {'loss':>10} {'delta':>10} {'std_r':>10} {'del_r':>10} {'eng_r':>10}{extra}")
    normal_loss = None
    for variant in ["normal", "zero_video", "zero_landmarks", "zero_both", "reverse_time", "mismatch_sample"]:
        loss, stats = evaluate_variant(model, loader, criterion, device, variant, args.max_batches)
        if variant == "normal":
            normal_loss = loss
        delta = 0.0 if normal_loss is None else loss - normal_loss
        text = (
            f"{variant:<18} {loss:10.6f} {delta:10.6f} "
            f"{stats.get('std_ratio', 0):10.3f} {stats.get('delta_ratio', 0):10.3f} {stats.get('energy_ratio', 0):10.3f}"
        )
        if args.include_units:
            text += f" {stats.get('unit_loss', 0):10.4f} {stats.get('unit_acc', 0):10.3f}"
        print(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Input ablation for srcV2 R2INR checkpoints.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR")
    parser.add_argument("--checkpoint", default="checkpoints_r2inr_v2/best_model.pth")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--include-units", action="store_true")
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=[
            "auto",
            "r2inr",
            "simple",
            "motion_tcn",
            "context_sync",
            "context_align",
            "context_detail",
            "context_motion_detail",
            "content_unit",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-tokens", type=int, default=4)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-units", type=int, default=0)
    parser.add_argument("--align-window", type=float, default=0.18)
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--detail-scale", type=float, default=0.35)
    parser.add_argument("--detail-layers", type=int, default=3)
    parser.add_argument("--unit-temperature", type=float, default=1.0)
    parser.add_argument("--detach-unit-condition", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--detach-content-hidden", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
