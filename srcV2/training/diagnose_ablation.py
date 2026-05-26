from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV2.data import R2INRDataset, collate_r2inr
from srcV2.models import MaskedMelLoss
from srcV2.training.common import build_model, load_checkpoint, masked_stats, model_inputs
from srcV2.utils.common import batch_to_device, get_device, seed_everything


def legacy_mouth_motion_features(landmarks: torch.Tensor) -> torch.Tensor:
    xy = torch.nan_to_num(landmarks[..., :2].float(), nan=0.0, posinf=0.0, neginf=0.0)
    min_xy = xy.amin(dim=2)
    max_xy = xy.amax(dim=2)
    width_height = max_xy - min_xy
    center = xy.mean(dim=2)
    d1 = torch.zeros_like(xy)
    d2 = torch.zeros_like(xy)
    if xy.shape[1] > 1:
        d1[:, 1:] = xy[:, 1:] - xy[:, :-1]
    if xy.shape[1] > 2:
        d2[:, 1:] = d1[:, 1:] - d1[:, :-1]
    speed = d1.norm(dim=-1).mean(dim=2, keepdim=True)
    accel = d2.norm(dim=-1).mean(dim=2, keepdim=True)
    mouth_open = xy[..., 1].std(dim=2, unbiased=False, keepdim=True)
    area = width_height[:, :, :1] * width_height[:, :, 1:2]
    return torch.cat([center, width_height, mouth_open, area, speed, accel], dim=-1)


def apply_variant(batch: dict, variant: str) -> dict:
    out = {k: v.clone() if torch.is_tensor(v) else deepcopy(v) for k, v in batch.items()}
    if variant == "normal":
        return out
    if variant in {"zero_video", "zero_both"}:
        out["video"].zero_()
    if variant in {"zero_landmarks", "zero_both"}:
        out["landmarks"].zero_()
        out["mouth_valid_mask"].zero_()
        if "mouth_motion" in out:
            out["mouth_motion"].zero_()
    if variant == "reverse_time":
        out["video"] = torch.flip(out["video"], dims=[2])
        out["landmarks"] = torch.flip(out["landmarks"], dims=[1])
        out["mouth_valid_mask"] = torch.flip(out["mouth_valid_mask"], dims=[1])
        if "mouth_motion" in out:
            out["mouth_motion"] = torch.flip(out["mouth_motion"], dims=[1])
    if variant == "mismatch_sample":
        if out["video"].shape[0] < 2:
            return out
        out["video"] = torch.roll(out["video"], shifts=1, dims=0)
        out["landmarks"] = torch.roll(out["landmarks"], shifts=1, dims=0)
        out["mouth_valid_mask"] = torch.roll(out["mouth_valid_mask"], shifts=1, dims=0)
        if "mouth_motion" in out:
            out["mouth_motion"] = torch.roll(out["mouth_motion"], shifts=1, dims=0)
    return out


@torch.no_grad()
def evaluate_variant(model, loader, criterion, device, variant: str, max_batches: int, motion_dim: int):
    total = 0.0
    count = 0
    first_stats = None
    for batch_idx, batch in enumerate(tqdm(loader, desc=variant, leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = batch_to_device(batch, device)
        if motion_dim <= 8:
            batch["mouth_motion"] = legacy_mouth_motion_features(batch["landmarks"])
        vbatch = apply_variant(batch, variant)
        pred = model(model_inputs(vbatch))
        loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
        if first_stats is None:
            first_stats = masked_stats(pred, batch["mel"], batch["mel_mask"])
    return total / max(1, count), (first_stats or {})


def run(args) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    ds = R2INRDataset(args.data_dir, max_frames=args.max_frames, random_crop=False, seed=args.seed, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_r2inr)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    motion_weight = ckpt.get("model_state_dict", {}).get("encoder.motion.0.weight")
    args.motion_dim = int(motion_weight.shape[1]) if torch.is_tensor(motion_weight) else int(cfg.get("motion_dim", 19))
    for key in ("dim", "spatial_tokens", "num_landmark_points", "dropout", "multi_gpu"):
        if hasattr(args, key):
            setattr(args, key, cfg.get(key.replace("_", "-"), cfg.get(key, getattr(args, key))))
    model = build_model(device, args)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    criterion = MaskedMelLoss(ckpt["mel_mean"], ckpt["mel_std"]).to(device)

    print(f"[checkpoint] {args.checkpoint}")
    print(f"[data] {args.data_dir} samples={len(ds)} motion_dim={args.motion_dim}")
    print(f"{'variant':<18} {'loss':>10} {'delta':>10} {'std_r':>10} {'del_r':>10} {'eng_r':>10}")
    normal_loss = None
    for variant in ["normal", "zero_video", "zero_landmarks", "zero_both", "reverse_time", "mismatch_sample"]:
        loss, stats = evaluate_variant(model, loader, criterion, device, variant, args.max_batches, args.motion_dim)
        if variant == "normal":
            normal_loss = loss
        delta = 0.0 if normal_loss is None else loss - normal_loss
        print(
            f"{variant:<18} {loss:10.6f} {delta:10.6f} "
            f"{stats.get('std_ratio', 0):10.3f} {stats.get('delta_ratio', 0):10.3f} {stats.get('energy_ratio', 0):10.3f}"
        )


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-tokens", type=int, default=4)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
