from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from srcV9.data import LandmarkEnvelopeDataset, collate_envelope, split_cache_files
from srcV9.inference.reconstruct_griffinlim import griffin_lim, save_wav
from srcV9.inference.synthesize_vocoder import logmel_to_linear_mag
from srcV9.models import VideoParametricVocoderModel
from srcV9.training.train_parametric_vocoder import compute_targets, parse_layers
from srcV9.utils import batch_to_device, get_device


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> VideoParametricVocoderModel:
    cfg = checkpoint.get("config", {})
    model = VideoParametricVocoderModel(
        num_points=int(cfg.get("num_landmark_points", 40)),
        dim=int(cfg.get("dim", 384)),
        n_mels=int(cfg.get("n_mels", 80)),
        source_bands=int(cfg.get("source_bands", 8)),
        visual_width=int(cfg.get("visual_width", 24)),
        visual_layers=parse_layers(cfg.get("visual_layers", "1,1,1,1")),
        visual_temporal_layers=int(cfg.get("visual_temporal_layers", 1)),
        landmark_tcn_layers=int(cfg.get("landmark_tcn_layers", 6)),
        landmark_transformer_layers=int(cfg.get("landmark_transformer_layers", 2)),
        nhead=int(cfg.get("nhead", 6)),
        decoder_layers=int(cfg.get("decoder_layers", 4)),
        dropout=0.0,
        source_scale_init=float(cfg.get("source_scale_init", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def choose_file(args) -> Path:
    if args.cache_file:
        return Path(args.cache_file)
    train_files, val_files = split_cache_files(
        args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_files=args.limit_files if args.limit_files > 0 else None,
    )
    files = val_files if args.split == "val" and val_files else train_files
    if not files:
        raise RuntimeError("No cache file available.")
    return files[max(0, min(int(args.sample_index), len(files) - 1))]


def synthesize_logmel(logmel: torch.Tensor, sample_rate: int, hop_length: int, n_fft: int, win_length: int, n_iter: int, seed: int) -> torch.Tensor:
    mag = torch.from_numpy(logmel_to_linear_mag(logmel.detach().cpu().float().numpy(), sample_rate, n_fft, logmel.shape[-1]))
    return griffin_lim(
        mag,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=n_iter,
        length=int(logmel.shape[0] * hop_length),
        seed=seed,
    )


@torch.no_grad()
def run(args) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    device = get_device(args.device)
    cache_file = choose_file(args)
    ds = LandmarkEnvelopeDataset(
        args.data_dir,
        files=[cache_file],
        window_frames=args.window_frames or int(cfg.get("window_frames", 45)),
        hop_frames=args.hop_frames or int(cfg.get("hop_frames", 15)),
        max_windows_per_file=1,
        smooth_target_frames=int(cfg.get("smooth_target_frames", 7)),
        seed=args.seed,
    )
    batch = batch_to_device(next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_envelope))), device)
    model = build_model_from_checkpoint(checkpoint, device)
    out = model(batch, target_len=batch["mel"].shape[1])
    targets = compute_targets(batch, int(cfg.get("n_mels", 80)), int(cfg.get("source_bands", 8)), model)
    length = int(batch["mel_lengths"][0].item())
    sample_rate = int(args.sample_rate or batch["sample_rates"][0].item())
    hop_length = int(args.hop_length or batch["hop_lengths"][0].item())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        "pred_parametric_griffinlim.wav": out["mel"][0, :length].detach().cpu(),
        "pred_filter_griffinlim.wav": (out["energy"] + out["filter"])[0, :length].detach().cpu(),
        "target_full_griffinlim.wav": batch["mel"][0, :length].detach().cpu(),
        "target_filter_griffinlim.wav": (targets["energy"] + targets["filter"])[0, :length].detach().cpu(),
    }
    for name, logmel in tensors.items():
        wav = synthesize_logmel(logmel, sample_rate, hop_length, args.n_fft, args.win_length, args.n_iter, args.seed)
        save_wav(output_dir / name, wav, sample_rate)

    torch.save(
        {
            "pred": out["mel"][0, :length].detach().cpu(),
            "pred_filter": out["filter"][0, :length].detach().cpu(),
            "pred_energy": out["energy"][0, :length].detach().cpu(),
            "pred_voicing": out["voicing"][0, :length].detach().cpu(),
            "pred_source": out["source"][0, :length].detach().cpu(),
            "target": batch["mel"][0, :length].detach().cpu(),
            "target_filter": targets["filter"][0, :length].detach().cpu(),
            "target_energy": targets["energy"][0, :length].detach().cpu(),
            "target_voicing": targets["voicing"][0, :length].detach().cpu(),
            "target_broad": targets["broad_mel"][0, :length].detach().cpu(),
            "cache_file": str(cache_file),
        },
        output_dir / "parametric_debug.pt",
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = [
            ("Target full log-mel", batch["mel"][0, :length].detach().cpu().T),
            ("Target filter shape", targets["filter"][0, :length].detach().cpu().T),
            ("Target broad source", targets["broad_mel"][0, :length].detach().cpu().T),
            ("Predicted final log-mel", out["mel"][0, :length].detach().cpu().T),
            ("Predicted filter shape", out["filter"][0, :length].detach().cpu().T),
            ("Predicted source", out["source"][0, :length].detach().cpu().T),
        ]
        fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 2.8 * (len(panels) + 1)), sharex=False)
        for ax, (title, image) in zip(axes[:-1], panels):
            ax.imshow(image.numpy(), aspect="auto", origin="lower")
            ax.set_title(title)
        axes[-1].plot(targets["voicing"][0, :length, 0].detach().cpu().numpy(), label="target voicing")
        axes[-1].plot(out["voicing"][0, :length, 0].detach().cpu().numpy(), label="pred voicing")
        axes[-1].legend(loc="upper right")
        axes[-1].set_ylim(0.0, 1.0)
        axes[-1].set_title("Voicing")
        fig.tight_layout()
        fig.savefig(output_dir / "parametric_compare.png", dpi=140)
        plt.close(fig)

    print(f"[input] {cache_file}")
    print(f"[out] {output_dir / 'pred_parametric_griffinlim.wav'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize srcV9 parametric vocoder checkpoint with Griffin-Lim.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="parametric_vocoder_synthesis")
    parser.add_argument("--cache-file", default="")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--window-frames", type=int, default=0)
    parser.add_argument("--hop-frames", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=0)
    parser.add_argument("--hop-length", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-iter", type=int, default=64)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
