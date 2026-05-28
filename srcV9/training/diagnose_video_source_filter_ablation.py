from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV9.data import LandmarkEnvelopeDataset, collate_envelope, split_cache_files
from srcV9.inference.synthesize_source_filter import build_model_from_checkpoint
from srcV9.training.train_source_filter_vocoder import source_filter_loss, masked_stats
from srcV9.utils import batch_to_device, get_device


def clone_batch(batch: dict) -> dict:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def apply_variant(batch: dict, variant: str) -> dict:
    batch = clone_batch(batch)
    if variant in {"zero_video", "zero_both"}:
        batch["video"].zero_()
    if variant in {"zero_landmarks", "zero_both"}:
        batch["landmarks"].zero_()
    if variant == "reverse_time":
        batch["video"] = batch["video"].flip(dims=[2])
        batch["landmarks"] = batch["landmarks"].flip(dims=[1])
        batch["mouth_valid_mask"] = batch["mouth_valid_mask"].flip(dims=[1])
    if variant == "mismatch_batch" and batch["video"].shape[0] > 1:
        perm = torch.roll(torch.arange(batch["video"].shape[0], device=batch["video"].device), shifts=1)
        for key in ("video", "landmarks", "video_times", "mouth_valid_mask", "landmark_mask", "landmark_lengths"):
            batch[key] = batch[key][perm]
    return batch


@torch.no_grad()
def eval_variant(model, loader, device, checkpoint: dict, loss_args: SimpleNamespace, variant: str) -> dict[str, float]:
    total = 0.0
    count = 0
    sums: dict[str, float] = {}
    for batch in tqdm(loader, desc=variant, leave=False):
        batch = batch_to_device(batch, device)
        batch = apply_variant(batch, variant)
        out = model(batch, target_len=batch["mel"].shape[1])
        loss, _ = source_filter_loss(out, batch, checkpoint["mel_mean"], checkpoint["mel_std"], loss_args)
        total += float(loss.detach().cpu())
        count += 1
        stats = masked_stats(out["mel"].float(), batch["mel"].float(), batch["mel_mask"])
        stats["source_std"] = float(out["source"][batch["mel_mask"].bool()].std(unbiased=False).detach().cpu())
        stats["source_scale"] = float(out["source_scale"].detach().float().mean().cpu())
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    row = {"loss": total / max(1, count)}
    for key, value in sums.items():
        row[key] = value / max(1, count)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate video+landmark source-filter vocoder checkpoint.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--window-frames", type=int, default=0)
    parser.add_argument("--hop-frames", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-windows-per-file", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    loss_args = SimpleNamespace(
        lambda_env=float(cfg.get("lambda_env", 0.7)),
        lambda_source=float(cfg.get("lambda_source", 0.8)),
        lambda_final=float(cfg.get("lambda_final", 1.0)),
        lambda_final_delta=float(cfg.get("lambda_final_delta", 0.35)),
        lambda_source_delta=float(cfg.get("lambda_source_delta", 0.15)),
        lambda_energy=float(cfg.get("lambda_energy", 0.05)),
    )
    window_frames = int(args.window_frames or cfg.get("window_frames", 45))
    hop_frames = int(args.hop_frames or cfg.get("hop_frames", 15))
    smooth = int(cfg.get("smooth_target_frames", 5))
    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, int(cfg.get("seed", 42)), limit_files=limit_files)
    files = val_files if args.split == "val" and val_files else train_files
    dataset = LandmarkEnvelopeDataset(
        args.data_dir,
        files=files,
        window_frames=window_frames,
        hop_frames=hop_frames,
        max_windows_per_file=args.max_windows_per_file,
        smooth_target_frames=smooth,
        seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_envelope)
    model = build_model_from_checkpoint(checkpoint, device)
    variants = ["normal", "zero_video", "zero_landmarks", "zero_both", "reverse_time", "mismatch_batch"]
    results = {variant: eval_variant(model, loader, device, checkpoint, loss_args, variant) for variant in variants}
    normal = results["normal"]["loss"]
    print(f"[data] files={len(files)} windows={len(dataset)} split={args.split}")
    print("variant              loss      delta     std_r     del_r   src_std")
    for variant in variants:
        row = results[variant]
        print(
            f"{variant:<16} {row['loss']:9.6f} {row['loss'] - normal:+9.6f} "
            f"{row.get('std_ratio', 0.0):9.3f} {row.get('delta_ratio', 0.0):9.3f} {row.get('source_std', 0.0):9.3f}"
        )


if __name__ == "__main__":
    run(parse_args())
