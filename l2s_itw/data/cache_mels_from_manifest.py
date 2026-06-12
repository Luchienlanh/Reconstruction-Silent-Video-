from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from l2s_itw.data.manifest import read_manifest, write_manifest
from l2s_itw.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache log-mel targets from videos without copying the videos.")
    parser.add_argument("--input-manifest", required=True, help="JSONL rows with video_path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", default="", help="Defaults to <output-dir>/manifest.jsonl.")
    parser.add_argument("--video-key", default="video_path")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument(
        "--normalize-mel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each linear mel by its sample max before log; matches the existing HiFi-GAN-compatible cache.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def resolve_row_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else manifest_dir / path


def load_waveform_with_torchaudio(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.ndim != 2:
        raise ValueError(f"Expected waveform [channels, time], got {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        wav = torchaudio.functional.resample(wav, int(sr), int(sample_rate))
        sr = int(sample_rate)
    return wav.clamp(-1.0, 1.0), int(sr)


def load_waveform_with_ffmpeg(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError(f"ffmpeg decoded empty audio from {path}")
    return torch.from_numpy(audio).view(1, -1).clamp(-1.0, 1.0), int(sample_rate)


def load_waveform(path: Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    try:
        return load_waveform_with_torchaudio(path, sample_rate)
    except Exception:
        return load_waveform_with_ffmpeg(path, sample_rate)


def log_mel_from_waveform(wav: torch.Tensor, sample_rate: int, args: argparse.Namespace) -> torch.Tensor:
    def finalize_mel(mel: torch.Tensor) -> torch.Tensor:
        if bool(args.normalize_mel):
            mel = mel / mel.amax().clamp_min(1e-5)
        return torch.log(mel.clamp_min(1e-5))

    try:
        import torchaudio

        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(args.n_fft),
            win_length=int(args.win_length),
            hop_length=int(args.hop_length),
            f_min=float(args.f_min),
            f_max=float(args.f_max),
            n_mels=int(args.n_mels),
            power=1.0,
            center=True,
        )
        mel = mel_fn(wav).squeeze(0).transpose(0, 1).contiguous()
        return finalize_mel(mel)
    except Exception:
        import librosa

        mel_np = librosa.feature.melspectrogram(
            y=wav.squeeze(0).cpu().numpy(),
            sr=sample_rate,
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            win_length=int(args.win_length),
            fmin=float(args.f_min),
            fmax=float(args.f_max),
            n_mels=int(args.n_mels),
            power=1.0,
            center=True,
        )
        return finalize_mel(torch.from_numpy(mel_np.T).float())


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    manifest_dir = input_manifest.parent
    rows = read_manifest(input_manifest, limit=args.limit)
    if not rows:
        raise ValueError(f"Manifest is empty: {input_manifest}")

    output_dir = ensure_dir(args.output_dir)
    mel_dir = ensure_dir(output_dir / "mels")
    output_manifest = Path(args.output_manifest) if args.output_manifest else output_dir / "manifest.jsonl"

    output_rows = []
    failed = []
    for index, row in enumerate(tqdm(rows, desc="cache mels")):
        if args.video_key not in row:
            raise KeyError(f"Missing '{args.video_key}' in manifest row {index}")
        sample_id = str(row.get("id") or f"sample_{index:06d}")
        video_path = resolve_row_path(str(row[args.video_key]), manifest_dir)
        mel_path = mel_dir / f"{sample_id}.mel.pt"

        try:
            if args.overwrite or not mel_path.exists():
                wav, sr = load_waveform(video_path, int(args.sample_rate))
                mel = log_mel_from_waveform(wav, sr, args).float()
                torch.save(
                    {
                        "mel": mel,
                        "source_video_path": str(video_path.resolve()),
                        "sample_rate": int(sr),
                        "n_fft": int(args.n_fft),
                        "win_length": int(args.win_length),
                        "hop_length": int(args.hop_length),
                        "n_mels": int(args.n_mels),
                        "normalize_mel": bool(args.normalize_mel),
                    },
                    mel_path,
                )
            else:
                cached = torch.load(mel_path, map_location="cpu")
                mel = cached["mel"] if isinstance(cached, dict) and "mel" in cached else cached

            new_row = dict(row)
            new_row["mel_path"] = str(mel_path.relative_to(output_dir))
            new_row["sample_rate"] = int(args.sample_rate)
            new_row["n_fft"] = int(args.n_fft)
            new_row["win_length"] = int(args.win_length)
            new_row["hop_length"] = int(args.hop_length)
            new_row["n_mels"] = int(args.n_mels)
            new_row["mel_frames"] = int(mel.shape[0])
            output_rows.append(new_row)
        except Exception as exc:
            if not args.skip_errors:
                raise
            failed.append((sample_id, str(exc)))

    write_manifest(output_rows, output_manifest)
    print(f"wrote {output_manifest}")
    print(f"rows: {len(output_rows)}")
    if failed:
        print(f"failed: {len(failed)}")
        for sample_id, error in failed[:5]:
            print(f"{sample_id}: {error}")


if __name__ == "__main__":
    main()
