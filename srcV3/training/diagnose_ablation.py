from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV3.data import WindowedMelDataset, collate_windows, split_cache_files
from srcV3.inference.overlap_add import build_model_from_checkpoint
from srcV3.models import MaskedMelLoss
from srcV3.utils import batch_to_device, get_device


def clone_batch(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.clone() if torch.is_tensor(value) else value
    return out


def apply_variant(batch: dict, variant: str) -> dict:
    batch = clone_batch(batch)
    if variant in ("zero_video", "zero_both"):
        batch["video"].zero_()
    if variant in ("zero_landmarks", "zero_both"):
        batch["landmarks"].zero_()
    if variant == "reverse_time":
        batch["video"] = batch["video"].flip(dims=[2])
        batch["landmarks"] = batch["landmarks"].flip(dims=[1])
    if variant == "mismatch_batch" and batch["video"].shape[0] > 1:
        perm = torch.roll(torch.arange(batch["video"].shape[0], device=batch["video"].device), shifts=1)
        batch["video"] = batch["video"][perm]
        batch["landmarks"] = batch["landmarks"][perm]
    return batch


@torch.no_grad()
def eval_variant(model, loader, criterion, device, variant: str) -> float:
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc=variant, leave=False):
        batch = batch_to_device(batch, device)
        batch = apply_variant(batch, variant)
        pred = model(batch, target_len=batch["mel"].shape[1])
        loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate srcV3 checkpoint on window samples.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--hop-frames", type=int, default=10)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-windows-per-file", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(ckpt, device)
    model.eval()
    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, _ = split_cache_files(args.data_dir, val_ratio=0.0, seed=42, limit_files=limit_files)
    ds = WindowedMelDataset(
        args.data_dir,
        files=train_files,
        window_frames=args.window_frames,
        hop_frames=args.hop_frames,
        max_windows_per_file=args.max_windows_per_file,
        random_windows_per_file=0,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_windows,
    )
    criterion = MaskedMelLoss(ckpt["mel_mean"], ckpt["mel_std"]).to(device)
    variants = ["normal", "zero_video", "zero_landmarks", "zero_both", "reverse_time", "mismatch_batch"]
    results = {variant: eval_variant(model, loader, criterion, device, variant) for variant in variants}
    normal = results["normal"]
    print("variant              loss      delta")
    for variant in variants:
        print(f"{variant:<16} {results[variant]:9.6f} {results[variant] - normal:+9.6f}")


if __name__ == "__main__":
    run(parse_args())

