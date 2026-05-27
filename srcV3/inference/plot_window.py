from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from srcV3.data import WindowedMelDataset, collate_windows, split_cache_files
from srcV3.inference.overlap_add import build_model_from_checkpoint
from srcV3.utils import batch_to_device, get_device


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(ckpt, device)
    model.eval()

    limit_files = args.limit_files if args.limit_files > 0 else None
    files, _ = split_cache_files(args.data_dir, val_ratio=0.0, seed=args.seed, limit_files=limit_files)
    dataset = WindowedMelDataset(
        args.data_dir,
        files=files,
        window_frames=args.window_frames,
        hop_frames=args.hop_frames,
        max_windows_per_file=args.max_windows_per_file,
        random_windows_per_file=0,
        seed=args.seed,
    )
    idx = max(0, min(args.window_index, len(dataset) - 1))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_windows)
    batch = None
    for i, item in enumerate(loader):
        if i == idx:
            batch = item
            break
    if batch is None:
        raise RuntimeError("Could not load requested window.")

    batch = batch_to_device(batch, device)
    pred = model(batch, target_len=batch["mel"].shape[1]).float()[0].cpu()
    target = batch["mel"].float()[0].cpu()
    mel_len = int(batch["mel_lengths"][0].item())
    pred = pred[:mel_len]
    target = target[:mel_len]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(batch["paths"][0]).stem
    out_pt = output_dir / f"{stem}_window{idx:04d}_pred.pt"
    torch.save(
        {
            "pred_mel": pred,
            "target_mel": target,
            "path": batch["paths"][0],
            "window_start": int(batch["window_starts"][0].item()),
            "window_end": int(batch["window_ends"][0].item()),
            "checkpoint": str(args.checkpoint),
        },
        out_pt,
    )
    print(f"[saved] {out_pt}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].imshow(target.T.numpy(), aspect="auto", origin="lower")
        axes[0].set_title("target")
        axes[1].imshow(pred.T.numpy(), aspect="auto", origin="lower")
        axes[1].set_title("prediction")
        axes[2].imshow((pred - target).abs().T.numpy(), aspect="auto", origin="lower")
        axes[2].set_title("abs error")
        fig.tight_layout()
        out_png = output_dir / f"{stem}_window{idx:04d}_pred_vs_target.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"[saved] {out_png}")
    except Exception as exc:
        print(f"[plot] skipped: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one srcV3 window prediction against target mel.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="inference_srcV3_window")
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--hop-frames", type=int, default=10)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=1)
    parser.add_argument("--max-windows-per-file", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

