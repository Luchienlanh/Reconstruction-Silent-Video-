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
from srcV9.models import LandmarkSourceFilterVocoderModel, VideoSourceFilterVocoderModel
from srcV9.utils import batch_to_device, get_device


def _parse_layers(value) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(x) for x in value)  # type: ignore[return-value]
    parts = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    return tuple(parts) if len(parts) == 4 else (1, 1, 1, 1)  # type: ignore[return-value]


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get("config", {})
    if checkpoint.get("model_type") == "video_source_filter_vocoder":
        model = VideoSourceFilterVocoderModel(
            num_points=int(cfg.get("num_landmark_points", 40)),
            dim=int(cfg.get("dim", 384)),
            n_mels=int(cfg.get("n_mels", 80)),
            source_bands=int(cfg.get("source_bands", 16)),
            visual_width=int(cfg.get("visual_width", 24)),
            visual_layers=_parse_layers(cfg.get("visual_layers", (1, 1, 1, 1))),
            visual_temporal_layers=int(cfg.get("visual_temporal_layers", 1)),
            landmark_tcn_layers=int(cfg.get("landmark_tcn_layers", 6)),
            landmark_transformer_layers=int(cfg.get("landmark_transformer_layers", 2)),
            nhead=int(cfg.get("nhead", 6)),
            decoder_layers=int(cfg.get("decoder_layers", 6)),
            dropout=0.0,
            output_bias_init=float(checkpoint.get("mel_mean", torch.tensor([-4.0])).float().mean().item()),
            source_scale_init=float(cfg.get("source_scale_init", 0.6)),
        ).to(device)
    else:
        model = LandmarkSourceFilterVocoderModel(
            num_points=int(cfg.get("num_landmark_points", 40)),
            dim=int(cfg.get("dim", 384)),
            n_mels=int(cfg.get("n_mels", 80)),
            source_bands=int(cfg.get("source_bands", 16)),
            tcn_layers=int(cfg.get("tcn_layers", 6)),
            transformer_layers=int(cfg.get("transformer_layers", 2)),
            nhead=int(cfg.get("nhead", 6)),
            decoder_layers=int(cfg.get("decoder_layers", 6)),
            dropout=0.0,
            output_bias_init=float(checkpoint.get("mel_mean", torch.tensor([-4.0])).float().mean().item()),
            source_scale_init=float(cfg.get("source_scale_init", 0.6)),
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
        raise RuntimeError("No cache file available for synthesis.")
    return files[max(0, min(int(args.sample_index), len(files) - 1))]


def synthesize_logmel_griffinlim(
    logmel: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    win_length: int,
    n_iter: int,
    seed: int,
) -> torch.Tensor:
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
        smooth_target_frames=int(cfg.get("smooth_target_frames", 5)),
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_envelope)
    batch = batch_to_device(next(iter(loader)), device)
    model = build_model_from_checkpoint(checkpoint, device)
    out = model(batch, target_len=batch["mel"].shape[1])
    length = int(batch["mel_lengths"][0].item())
    pred = out["mel"][0, :length].detach().cpu()
    pred_env = out["envelope"][0, :length].detach().cpu()
    pred_source = out["source"][0, :length].detach().cpu()
    target = batch["mel"][0, :length].detach().cpu()
    target_env = batch["target_mel"][0, :length].detach().cpu()
    target_source = target - target_env
    sample_rate = int(args.sample_rate or batch["sample_rates"][0].item())
    hop_length = int(args.hop_length or batch["hop_lengths"][0].item())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, logmel in (
        ("pred_source_filter_griffinlim.wav", pred),
        ("pred_envelope_griffinlim.wav", pred_env),
        ("target_full_griffinlim.wav", target),
        ("target_envelope_griffinlim.wav", target_env),
    ):
        wav = synthesize_logmel_griffinlim(
            logmel,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=args.n_fft,
            win_length=args.win_length,
            n_iter=args.n_iter,
            seed=args.seed,
        )
        save_wav(output_dir / name, wav, sample_rate)

    torch.save(
        {
            "pred": pred,
            "pred_env": pred_env,
            "pred_source": pred_source,
            "target": target,
            "target_env": target_env,
            "target_source": target_source,
            "cache_file": str(cache_file),
            "source_scale": out["source_scale"].detach().cpu(),
        },
        output_dir / "source_filter_debug.pt",
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = [
            ("Target full log-mel", target.T),
            ("Target filter/envelope", target_env.T),
            ("Target source = full - envelope", target_source.T),
            ("Predicted final log-mel", pred.T),
            ("Predicted filter/envelope", pred_env.T),
            ("Predicted source", pred_source.T),
        ]
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.8 * len(panels)), sharex=True)
        for ax, (title, image) in zip(axes, panels):
            ax.imshow(image.numpy(), aspect="auto", origin="lower")
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / "source_filter_compare.png", dpi=140)
        plt.close(fig)

    print(f"[input] {cache_file}")
    print(f"[out] {output_dir / 'pred_source_filter_griffinlim.wav'}")
    print(f"[scale] {float(out['source_scale'].detach().cpu()):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize source-filter srcV9 checkpoint with inverse-mel Griffin-Lim.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="source_filter_synthesis")
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
