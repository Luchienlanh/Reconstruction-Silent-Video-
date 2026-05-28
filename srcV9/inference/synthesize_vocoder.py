from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from srcV9.data import LandmarkEnvelopeDataset, collate_envelope, split_cache_files
from srcV9.models import LandmarkEnvelopeVocoderModel
from srcV9.utils import batch_to_device, get_device


def _mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    try:
        import torchaudio

        fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=0.0,
            f_max=float(sample_rate // 2),
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm=None,
            mel_scale="htk",
        )
        return fbanks.T.cpu().numpy().astype(np.float32)
    except Exception:
        import librosa

        return librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sample_rate // 2,
            htk=True,
            norm=None,
        ).astype(np.float32)


def logmel_to_linear_mag(logmel: np.ndarray, sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    basis = _mel_basis(sample_rate, n_fft, n_mels)
    inv_basis = np.linalg.pinv(basis)
    mel_amp = np.exp(np.clip(logmel, -12.0, 5.0)).T
    mag = inv_basis @ mel_amp
    return np.maximum(mag, 1e-5).astype(np.float32)


def make_carrier(length: int, sample_rate: int, f0: float, voiced_mix: float, noise_mix: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(max(1, int(length)), dtype=np.float32) / float(sample_rate)
    max_harm = max(1, min(80, int((sample_rate * 0.5) // max(float(f0), 1.0))))
    buzz = np.zeros_like(t)
    for k in range(1, max_harm + 1):
        buzz += np.sin(2.0 * np.pi * float(f0) * k * t) / float(k)
    buzz /= max(1e-6, np.max(np.abs(buzz)))
    noise = rng.standard_normal(t.shape).astype(np.float32)
    noise /= max(1e-6, np.max(np.abs(noise)))
    carrier = float(voiced_mix) * buzz + float(noise_mix) * noise
    carrier /= max(1e-6, np.max(np.abs(carrier)))
    return carrier.astype(np.float32)


def synthesize_from_logmel(
    logmel: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 256,
    n_fft: int = 1024,
    win_length: int = 1024,
    f0: float = 140.0,
    voiced_mix: float = 0.85,
    noise_mix: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    target_mag = logmel_to_linear_mag(logmel, sample_rate, n_fft, logmel.shape[-1])
    frames = int(logmel.shape[0])
    carrier_len = int((frames + 2) * hop_length + n_fft)
    carrier = make_carrier(carrier_len, sample_rate, f0, voiced_mix, noise_mix, seed)
    window = torch.hann_window(int(win_length))
    carrier_t = torch.from_numpy(carrier)
    carrier_stft_t = torch.stft(
        carrier_t,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        center=True,
        return_complex=True,
    )
    carrier_stft = carrier_stft_t[:, :frames].cpu().numpy()
    if carrier_stft.shape[1] < frames:
        carrier_stft = np.pad(carrier_stft, ((0, 0), (0, frames - carrier_stft.shape[1])), mode="edge")
    carrier_mag = np.maximum(np.abs(carrier_stft), 1e-5)

    basis = _mel_basis(sample_rate, n_fft, logmel.shape[-1])
    inv_basis = np.linalg.pinv(basis)
    carrier_env = np.maximum(inv_basis @ (basis @ carrier_mag), 1e-5)
    vocoder_stft = torch.from_numpy(carrier_stft * (target_mag / carrier_env))
    wav_t = torch.istft(
        vocoder_stft,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        center=True,
        length=int(frames * hop_length),
    )
    wav = wav_t.cpu().numpy()
    wav = np.nan_to_num(wav.astype(np.float32))
    wav /= max(1e-6, np.max(np.abs(wav))) * 1.05
    return wav.astype(np.float32)


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
        dropout=float(cfg.get("dropout", 0.0)),
        output_bias_init=float(checkpoint.get("mel_mean", torch.tensor([-4.0])).float().mean().item()),
        residual_alpha_init=float(cfg.get("residual_alpha_init", 0.25)),
        enable_residual=bool(has_residual and not cfg.get("disable_residual", False)),
    ).to(device)
    model.load_state_dict(state, strict=False)
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


@torch.no_grad()
def run(args) -> None:
    import soundfile as sf

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**checkpoint.get("config", {}))
    cache_file = choose_file(args)
    ds = LandmarkEnvelopeDataset(
        args.data_dir,
        files=[cache_file],
        window_frames=args.window_frames or int(getattr(cfg, "window_frames", 45)),
        hop_frames=args.hop_frames or int(getattr(cfg, "hop_frames", 15)),
        max_windows_per_file=1,
        smooth_target_frames=int(getattr(cfg, "smooth_target_frames", 3)),
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_envelope)
    batch = batch_to_device(next(iter(loader)), device)
    model = build_model_from_checkpoint(checkpoint, device)
    out = model(batch, target_len=batch["target_mel"].shape[1])
    length = int(batch["mel_lengths"][0].item())
    pred_logmel = out["mel"][0, :length].detach().cpu().float().numpy()
    pred_env_logmel = out.get("envelope", out["mel"])[0, :length].detach().cpu().float().numpy()
    target_env_logmel = batch["target_mel"][0, :length].detach().cpu().float().numpy()
    target_full_logmel = batch["mel"][0, :length].detach().cpu().float().numpy()
    sample_rate = int(args.sample_rate or batch["sample_rates"][0].item())
    hop_length = int(args.hop_length or batch["hop_lengths"][0].item())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_wav = synthesize_from_logmel(
        pred_logmel,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=args.n_fft,
        win_length=args.win_length,
        f0=args.f0,
        voiced_mix=args.voiced_mix,
        noise_mix=args.noise_mix,
        seed=args.seed,
    )
    target_wav = synthesize_from_logmel(
        target_env_logmel,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=args.n_fft,
        win_length=args.win_length,
        f0=args.f0,
        voiced_mix=args.voiced_mix,
        noise_mix=args.noise_mix,
        seed=args.seed,
    )
    pred_env_wav = synthesize_from_logmel(
        pred_env_logmel,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=args.n_fft,
        win_length=args.win_length,
        f0=args.f0,
        voiced_mix=args.voiced_mix,
        noise_mix=args.noise_mix,
        seed=args.seed,
    )
    target_full_wav = synthesize_from_logmel(
        target_full_logmel,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=args.n_fft,
        win_length=args.win_length,
        f0=args.f0,
        voiced_mix=args.voiced_mix,
        noise_mix=args.noise_mix,
        seed=args.seed,
    )
    sf.write(output_dir / "pred_vocoder.wav", pred_wav, sample_rate)
    sf.write(output_dir / "pred_envelope_vocoder.wav", pred_env_wav, sample_rate)
    sf.write(output_dir / "target_envelope_vocoder.wav", target_wav, sample_rate)
    sf.write(output_dir / "target_full_vocoder.wav", target_full_wav, sample_rate)
    torch.save(
        {
            "pred_logmel": torch.from_numpy(pred_logmel),
            "pred_env_logmel": torch.from_numpy(pred_env_logmel),
            "target_env_logmel": torch.from_numpy(target_env_logmel),
            "target_full_logmel": torch.from_numpy(target_full_logmel),
            "cache_file": str(cache_file),
            "sample_rate": sample_rate,
            "hop_length": hop_length,
        },
        output_dir / "synthesis_debug.pt",
    )
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        axes[0].imshow(target_full_logmel.T, aspect="auto", origin="lower")
        axes[0].set_title("Target full log-mel")
        axes[1].imshow(target_env_logmel.T, aspect="auto", origin="lower")
        axes[1].set_title("Target envelope log-mel")
        axes[2].imshow(pred_logmel.T, aspect="auto", origin="lower")
        axes[2].set_title("Predicted final log-mel")
        axes[3].imshow(pred_env_logmel.T, aspect="auto", origin="lower")
        axes[3].set_title("Predicted envelope log-mel")
        fig.tight_layout()
        fig.savefig(output_dir / "mel_compare.png", dpi=140)
        plt.close(fig)
    print(f"[input] {cache_file}")
    print(f"[out] {output_dir / 'pred_vocoder.wav'}")
    print(f"[out] {output_dir / 'pred_envelope_vocoder.wav'}")
    print(f"[out] {output_dir / 'target_envelope_vocoder.wav'}")
    print(f"[out] {output_dir / 'target_full_vocoder.wav'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize audio from srcV9 predicted envelope with a classical vocoder.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="vocoder_srcV9_synthesis")
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
    parser.add_argument("--f0", type=float, default=140.0)
    parser.add_argument("--voiced-mix", type=float, default=0.85)
    parser.add_argument("--noise-mix", type=float, default=0.15)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
