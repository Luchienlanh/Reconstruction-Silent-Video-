from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch
from tqdm.auto import tqdm

from srcV2.data.build_cache import LipLandmarkExtractor, decode_video_with_mouth_crops
from srcV2.utils.audio import log_mel_from_audio
from srcV2.utils.common import seed_everything


def parse_transcript(path: Path | None) -> tuple[str, dict[str, str]]:
    if path is None or not path.is_file():
        return "", {}
    meta: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip()
    text = meta.get("text") or " ".join(line.strip() for line in lines if line.strip())
    return text.strip(), meta


def scan_lrs2(root: str | Path, splits: list[str], require_text: bool = True) -> list[tuple[Path, Path | None, str]]:
    root = Path(root)
    pairs: list[tuple[Path, Path | None, str]] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for video_path in sorted(split_dir.rglob("*.mp4")):
            text_path = video_path.with_suffix(".txt")
            if require_text and not text_path.is_file():
                continue
            pairs.append((video_path, text_path if text_path.is_file() else None, split))
    return pairs


def cache_name(video_path: Path, split: str) -> str:
    speaker = video_path.parent.name
    digest = hashlib.sha1(str(video_path).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{split}_{speaker}_{video_path.stem}_{digest}.pt"


def extract_audio_with_ffmpeg(video_path: Path, wav_path: Path, sample_rate: int, ffmpeg_bin: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
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


def process_lrs2_clip(
    video_path: Path,
    text_path: Path | None,
    split: str,
    output_path: Path,
    args: argparse.Namespace,
    extractor: LipLandmarkExtractor,
) -> dict:
    video, landmarks, valid_mask, crop_boxes, video_times, fps = decode_video_with_mouth_crops(
        video_path,
        frame_size=args.frame_size,
        margin=args.margin,
        extractor=extractor,
        force_fps=args.force_fps,
    )
    transcript, transcript_meta = parse_transcript(text_path)

    with tempfile.TemporaryDirectory(prefix="lrs2_audio_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio_with_ffmpeg(video_path, wav_path, args.sample_rate, args.ffmpeg_bin)
        mel = log_mel_from_audio(
            wav_path,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            n_mels=args.n_mels,
        )

    mel_times = (torch.arange(mel.shape[0], dtype=torch.float32) + 0.5) * (args.hop_length / args.sample_rate)
    payload = {
        "format": "r2inr_v1",
        "dataset": "lrs2",
        "split": split,
        "video": video.contiguous(),
        "landmarks": landmarks.contiguous(),
        "mel": mel.contiguous(),
        "video_len": int(video.shape[1]),
        "mel_len": int(mel.shape[0]),
        "fps": float(fps),
        "sample_rate": int(args.sample_rate),
        "hop_length": int(args.hop_length),
        "video_times": video_times.contiguous(),
        "mel_times": mel_times.contiguous(),
        "mouth_valid_mask": valid_mask.contiguous(),
        "crop_boxes": crop_boxes.contiguous(),
        "transcript_text": transcript,
        "transcript_meta": transcript_meta,
        "source_video": str(video_path),
        "source_text": str(text_path) if text_path is not None else "",
    }
    torch.save(payload, output_path)
    return {
        "file": str(output_path),
        "source_video": str(video_path),
        "source_text": str(text_path) if text_path is not None else "",
        "split": split,
        "video_len": int(video.shape[1]),
        "mel_len": int(mel.shape[0]),
        "valid_ratio": float(valid_mask.float().mean().item()),
        "transcript": transcript,
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    root = Path(args.lrs2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = scan_lrs2(root, args.splits, require_text=args.require_text)
    if args.limit is not None:
        pairs = pairs[: max(1, min(int(args.limit), len(pairs)))]
    if not pairs:
        raise RuntimeError(f"No LRS2 mp4/txt pairs found under {root} for splits={args.splits}")

    extractor = LipLandmarkExtractor(enabled=not args.no_mediapipe, model_path=args.face_landmarker_model)
    manifest = []
    failures = []
    try:
        for video_path, text_path, split in tqdm(pairs, desc="lrs2-cache"):
            output_path = output_dir / cache_name(video_path, split)
            if output_path.exists() and not args.overwrite:
                manifest.append({"file": str(output_path), "source_video": str(video_path), "skipped": True})
                continue
            try:
                manifest.append(process_lrs2_clip(video_path, text_path, split, output_path, args, extractor))
            except Exception as exc:
                failures.append(
                    {
                        "video": str(video_path),
                        "text": str(text_path) if text_path is not None else "",
                        "error": repr(exc),
                    }
                )
                print(f"[fail] {video_path}: {exc}")
    finally:
        extractor.close()

    ok = len([item for item in manifest if not item.get("skipped")])
    skipped = len([item for item in manifest if item.get("skipped")])
    summary = {
        "lrs2_dir": str(root),
        "output_dir": str(output_dir),
        "splits": args.splits,
        "total_pairs": len(pairs),
        "ok": ok,
        "skipped": skipped,
        "failed": len(failures),
        "success_rate": (len(pairs) - len(failures)) / max(1, len(pairs)),
        "config": vars(args),
        "items": manifest,
        "failures": failures,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] ok_or_skipped={len(pairs) - len(failures)}/{len(pairs)} success_rate={summary['success_rate']:.3f}")
    print(f"[out] {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build srcV3 r2inr_v1 cache from LRS2 mp4/txt clips.")
    parser.add_argument("--lrs2-dir", default="Report/Data/lrs2_v1/mvlrs_v1")
    parser.add_argument("--output-dir", default="Processed_Data_R2INR_LRS2")
    parser.add_argument("--splits", nargs="+", default=["main"], help="LRS2 split folders to scan, e.g. main pretrain.")
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--margin", type=float, default=1.8)
    parser.add_argument("--force-fps", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true", help="Use center crop fallback for every frame.")
    parser.add_argument("--require-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-landmarker-model", default="face_landmarker_v2_with_blendshapes.task")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
