from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from srcV2.utils.audio import load_waveform
from srcV9.data.landmark_dataset import load_text_cache, split_cache_files
from srcV9.inference.synthesize_vocoder import logmel_to_linear_mag, synthesize_from_logmel


def extract_audio_with_ffmpeg(input_path: Path, wav_path: Path, sample_rate: int, ffmpeg_bin: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()}")
    if not wav_path.is_file() or wav_path.stat().st_size <= 0:
        raise RuntimeError("ffmpeg did not create a valid wav file")


def choose_cache_file(args: argparse.Namespace) -> Path:
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
        raise RuntimeError("No cache files available.")
    return files[max(0, min(int(args.sample_index), len(files) - 1))]


def resolve_audio_source(item: dict, args: argparse.Namespace) -> Path | None:
    for value in (args.audio_file, args.video_file, item.get("source_audio", ""), item.get("source_video", "")):
        if value:
            path = Path(str(value))
            if path.is_file():
                return path
    return None


def load_audio_from_source(item: dict, args: argparse.Namespace, sample_rate: int) -> tuple[torch.Tensor, int] | None:
    source = resolve_audio_source(item, args)
    if source is None:
        return None
    suffix = source.suffix.lower()
    if suffix in {".wav", ".flac", ".ogg", ".mp3"}:
        return load_waveform(source, sample_rate)
    with tempfile.TemporaryDirectory(prefix="griffinlim_audio_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio_with_ffmpeg(source, wav_path, sample_rate, args.ffmpeg_bin)
        return load_waveform(wav_path, sample_rate)


def stft_mag(wav: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    wav = wav.float().view(-1)
    window = torch.hann_window(int(win_length), dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs().clamp_min(1e-7)


def griffin_lim(
    magnitude: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int = 64,
    length: int | None = None,
    seed: int = 42,
) -> torch.Tensor:
    magnitude = magnitude.float().clamp_min(1e-7)
    generator = torch.Generator(device=magnitude.device)
    generator.manual_seed(int(seed))
    phase = torch.rand(magnitude.shape, generator=generator, device=magnitude.device, dtype=magnitude.dtype)
    phase = torch.exp(2j * torch.pi * phase)
    window = torch.hann_window(int(win_length), device=magnitude.device, dtype=magnitude.dtype)
    complex_spec = magnitude.to(torch.complex64) * phase.to(torch.complex64)
    for _ in range(max(1, int(n_iter))):
        wav = torch.istft(
            complex_spec,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=window,
            center=True,
            length=length,
        )
        rebuilt = torch.stft(
            wav,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=window,
            center=True,
            return_complex=True,
        )
        if rebuilt.shape[-1] != magnitude.shape[-1]:
            if rebuilt.shape[-1] > magnitude.shape[-1]:
                rebuilt = rebuilt[..., : magnitude.shape[-1]]
            else:
                rebuilt = torch.nn.functional.pad(rebuilt, (0, magnitude.shape[-1] - rebuilt.shape[-1]))
        phase = rebuilt / rebuilt.abs().clamp_min(1e-7)
        complex_spec = magnitude.to(torch.complex64) * phase.to(torch.complex64)
    wav = torch.istft(
        complex_spec,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        center=True,
        length=length,
    )
    wav = torch.nan_to_num(wav.float())
    wav = wav / wav.abs().max().clamp_min(1e-6) * 0.95
    return wav.cpu()


def save_wav(path: Path, wav: torch.Tensor | np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = wav.detach().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav)
    arr = np.nan_to_num(arr.astype(np.float32))
    sf.write(path, arr, int(sample_rate))


def plot_compare(output_dir: Path, item: dict, mel_mag: torch.Tensor | None, stft_mag_t: torch.Tensor | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if "mel" in item:
        panels.append(("Cached log-mel", item["mel"].float().T.cpu().numpy()))
    if mel_mag is not None:
        panels.append(("Inverse-mel linear magnitude", np.log(mel_mag.cpu().numpy() + 1e-7)))
    if stft_mag_t is not None:
        panels.append(("True STFT log magnitude", torch.log(stft_mag_t + 1e-7).cpu().numpy()))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3 * len(panels)), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, aspect="auto", origin="lower")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "griffinlim_compare.png", dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cache_file = choose_cache_file(args)
    item = load_text_cache(cache_file)
    sample_rate = int(args.sample_rate or item.get("sample_rate") or 16000)
    hop_length = int(args.hop_length or item.get("hop_length") or 256)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = load_audio_from_source(item, args, sample_rate)
    stft_mag_t = None
    if audio is not None:
        wav, sr = audio
        wav = wav.view(-1).float()
        save_wav(output_dir / "original.wav", wav, sr)
        stft_mag_t = stft_mag(wav, args.n_fft, hop_length, args.win_length)
        length = int(wav.numel())
        wav_stft_gl = griffin_lim(
            stft_mag_t,
            n_fft=args.n_fft,
            hop_length=hop_length,
            win_length=args.win_length,
            n_iter=args.n_iter,
            length=length,
            seed=args.seed,
        )
        save_wav(output_dir / "stft_griffinlim.wav", wav_stft_gl, sr)
    else:
        print("[warn] No source audio/video found. Skipping original.wav and stft_griffinlim.wav.")

    mel_mag_t = None
    if "mel" in item:
        logmel = item["mel"].float().cpu().numpy()
        mel_mag = logmel_to_linear_mag(logmel, sample_rate, args.n_fft, logmel.shape[-1])
        mel_mag_t = torch.from_numpy(mel_mag)
        mel_length = int(logmel.shape[0] * hop_length)
        wav_mel_gl = griffin_lim(
            mel_mag_t,
            n_fft=args.n_fft,
            hop_length=hop_length,
            win_length=args.win_length,
            n_iter=args.n_iter,
            length=mel_length,
            seed=args.seed,
        )
        save_wav(output_dir / "mel_griffinlim.wav", wav_mel_gl, sample_rate)
        wav_carrier = synthesize_from_logmel(
            logmel,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=args.n_fft,
            win_length=args.win_length,
            f0=args.f0,
            voiced_mix=args.voiced_mix,
            noise_mix=args.noise_mix,
            seed=args.seed,
        )
        save_wav(output_dir / "mel_carrier_vocoder.wav", wav_carrier, sample_rate)
    else:
        print("[warn] Cache has no mel. Skipping mel_griffinlim.wav and mel_carrier_vocoder.wav.")

    if args.plot:
        plot_compare(output_dir, item, mel_mag_t, stft_mag_t)

    print(f"[input] {cache_file}")
    print(f"[out] {output_dir}")
    print("[files] original.wav, stft_griffinlim.wav, mel_griffinlim.wav, mel_carrier_vocoder.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-learning Griffin-Lim/source-filter reconstruction from cache/audio.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--cache-file", default="")
    parser.add_argument("--audio-file", default="", help="Optional wav/flac/mp3 override.")
    parser.add_argument("--video-file", default="", help="Optional mp4 override with embedded audio.")
    parser.add_argument("--output-dir", default="griffinlim_reconstruction")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-rate", type=int, default=0)
    parser.add_argument("--hop-length", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-iter", type=int, default=64)
    parser.add_argument("--f0", type=float, default=140.0)
    parser.add_argument("--voiced-mix", type=float, default=0.85)
    parser.add_argument("--noise-mix", type=float, default=0.15)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
