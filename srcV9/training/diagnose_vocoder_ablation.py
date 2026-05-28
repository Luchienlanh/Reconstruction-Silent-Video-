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
from srcV9.models import LandmarkEnvelopeVocoderModel
from srcV9.training.train_vocoder_envelope import envelope_loss, masked_stats
from srcV9.utils import batch_to_device, get_device


def clone_batch(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.clone() if torch.is_tensor(value) else value
    return out


def apply_variant(batch: dict, variant: str) -> dict:
    batch = clone_batch(batch)
    if variant == "zero_landmarks":
        batch["landmarks"].zero_()
    elif variant == "reverse_time":
        batch["landmarks"] = batch["landmarks"].flip(dims=[1])
        batch["mouth_valid_mask"] = batch["mouth_valid_mask"].flip(dims=[1])
    elif variant == "mismatch_batch" and batch["landmarks"].shape[0] > 1:
        perm = torch.roll(torch.arange(batch["landmarks"].shape[0], device=batch["landmarks"].device), shifts=1)
        batch["landmarks"] = batch["landmarks"][perm]
        batch["video_times"] = batch["video_times"][perm]
        batch["mouth_valid_mask"] = batch["mouth_valid_mask"][perm]
        batch["landmark_mask"] = batch["landmark_mask"][perm]
        batch["landmark_lengths"] = batch["landmark_lengths"][perm]
    return batch


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> LandmarkEnvelopeVocoderModel:
    cfg = checkpoint.get("config", {})
    state = dict(checkpoint["model_state_dict"])
    if "mel_head.weight" in state and "envelope_head.weight" not in state:
        state["envelope_head.weight"] = state.pop("mel_head.weight")
        state["envelope_head.bias"] = state.pop("mel_head.bias")
    has_residual = "residual_head.weight" in state
    model = LandmarkEnvelopeVocoderModel(
        num_points=int(cfg.get("num_landmark_points", 40)),
        dim=int(cfg.get("dim", 384)),
        n_mels=int(cfg.get("n_mels", 80)),
        tcn_layers=int(cfg.get("tcn_layers", 6)),
        transformer_layers=int(cfg.get("transformer_layers", 2)),
        nhead=int(cfg.get("nhead", 6)),
        decoder_layers=int(cfg.get("decoder_layers", 6)),
        dropout=0.0,
        output_bias_init=float(checkpoint.get("mel_mean", torch.tensor([-4.0])).float().mean().item()),
        residual_alpha_init=float(cfg.get("residual_alpha_init", 0.25)),
        enable_residual=bool(has_residual and not cfg.get("disable_residual", False)),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def eval_variant(model, loader, device, checkpoint: dict, loss_args: SimpleNamespace, variant: str) -> dict[str, float]:
    total = 0.0
    count = 0
    stat_sums: dict[str, float] = {}
    for batch in tqdm(loader, desc=variant, leave=False):
        batch = batch_to_device(batch, device)
        batch = apply_variant(batch, variant)
        out = model(batch, target_len=batch["target_mel"].shape[1])
        loss, _ = envelope_loss(out, batch, checkpoint["mel_mean"], checkpoint["mel_std"], loss_args)
        total += float(loss.detach().cpu())
        count += 1
        stats = masked_stats(out["mel"].float(), batch["mel"].float(), batch["mel_mask"])
        stats["env_std_ratio"] = masked_stats(
            out.get("envelope", out["mel"]).float(),
            batch["target_mel"].float(),
            batch["mel_mask"],
        ).get("std_ratio", 0.0)
        stats["residual_alpha"] = float(out.get("residual_alpha", torch.tensor(0.0)).detach().float().mean().cpu())
        for key, value in stats.items():
            stat_sums[key] = stat_sums.get(key, 0.0) + float(value)
    out = {"loss": total / max(1, count)}
    for key, value in stat_sums.items():
        out[key] = value / max(1, count)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate srcV9 envelope-vocoder checkpoint on landmark windows.")
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
    parser.add_argument("--lambda-delta", type=float, default=None)
    parser.add_argument("--lambda-energy", type=float, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    window_frames = int(args.window_frames or cfg.get("window_frames", 45))
    hop_frames = int(args.hop_frames or cfg.get("hop_frames", 15))
    smooth_target_frames = int(cfg.get("smooth_target_frames", 3))
    loss_args = SimpleNamespace(
        lambda_delta=float(args.lambda_delta if args.lambda_delta is not None else cfg.get("lambda_delta", 0.2)),
        lambda_final_mel=float(cfg.get("lambda_final_mel", 0.35)),
        lambda_final_delta=float(cfg.get("lambda_final_delta", 0.15)),
        lambda_residual=float(cfg.get("lambda_residual", 0.15)),
        lambda_energy=float(args.lambda_energy if args.lambda_energy is not None else cfg.get("lambda_energy", 0.05)),
    )

    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, val_files = split_cache_files(args.data_dir, val_ratio=args.val_ratio, seed=int(cfg.get("seed", 42)), limit_files=limit_files)
    files = val_files if args.split == "val" and val_files else train_files
    dataset = LandmarkEnvelopeDataset(
        args.data_dir,
        files=files,
        window_frames=window_frames,
        hop_frames=hop_frames,
        max_windows_per_file=args.max_windows_per_file,
        random_windows_per_file=0,
        smooth_target_frames=smooth_target_frames,
        seed=int(cfg.get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_envelope,
    )
    model = build_model_from_checkpoint(checkpoint, device)
    variants = ["normal", "zero_landmarks", "reverse_time", "mismatch_batch"]
    results = {variant: eval_variant(model, loader, device, checkpoint, loss_args, variant) for variant in variants}
    normal = results["normal"]["loss"]

    print(f"[data] files={len(files)} windows={len(dataset)} split={args.split}")
    print(f"[window] frames={window_frames} hop={hop_frames} max_windows_per_file={args.max_windows_per_file}")
    print("variant              loss      delta     std_r     del_r     alpha")
    for variant in variants:
        row = results[variant]
        print(
            f"{variant:<16} {row['loss']:9.6f} {row['loss'] - normal:+9.6f} "
            f"{row.get('std_ratio', 0.0):9.3f} {row.get('delta_ratio', 0.0):9.3f} "
            f"{row.get('residual_alpha', 0.0):9.3f}"
        )


if __name__ == "__main__":
    run(parse_args())
