from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcV2.data.build_cache import LipLandmarkExtractor, decode_video_with_mouth_crops
from srcV2.utils.audio import log_mel_from_audio
from srcV2.utils.common import safe_name, seed_everything


def scan_raw_triplets(raw_dir: str | Path) -> list[tuple[Path, Path, Path]]:
    root = Path(raw_dir)
    triplets = []
    for video_path in sorted(root.rglob("video.mp4")):
        audio_path = video_path.with_name("audio.wav")
        transcript_path = video_path.with_name("transcript.txt")
        if audio_path.is_file() and transcript_path.is_file():
            triplets.append((video_path, audio_path, transcript_path))
    return triplets


def read_transcript(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore").strip()


def process_triplet(
    video_path: Path,
    audio_path: Path,
    transcript_path: Path,
    output_path: Path,
    args,
    extractor: LipLandmarkExtractor,
) -> dict:
    video, landmarks, valid_mask, crop_boxes, video_times, fps = decode_video_with_mouth_crops(
        video_path,
        frame_size=args.frame_size,
        margin=args.margin,
        extractor=extractor,
        force_fps=args.force_fps,
    )
    mel = log_mel_from_audio(
        audio_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
    )
    mel_times = (torch.arange(mel.shape[0], dtype=torch.float32) + 0.5) * (args.hop_length / args.sample_rate)
    transcript = read_transcript(transcript_path)
    payload = {
        "format": "r2inr_text_v1",
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
        "source_video": str(video_path),
        "source_audio": str(audio_path),
        "source_transcript": str(transcript_path),
    }
    torch.save(payload, output_path)
    return {
        "file": str(output_path),
        "video_len": int(video.shape[1]),
        "mel_len": int(mel.shape[0]),
        "transcript_len": len(transcript),
        "valid_ratio": float(valid_mask.float().mean().item()),
    }


def run(args) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    triplets = scan_raw_triplets(args.raw_dir)
    if args.limit is not None:
        triplets = triplets[: max(1, min(int(args.limit), len(triplets)))]
    if not triplets:
        raise RuntimeError(f"No video.mp4/audio.wav/transcript.txt triplets found under {args.raw_dir}")

    extractor = LipLandmarkExtractor(enabled=not args.no_mediapipe, model_path=args.face_landmarker_model)
    manifest = []
    failures = []
    try:
        for video_path, audio_path, transcript_path in tqdm(triplets, desc="build-text-cache"):
            out_name = safe_name(video_path) + ".pt"
            output_path = output_dir / out_name
            if output_path.exists() and not args.overwrite:
                manifest.append({"file": str(output_path), "skipped": True})
                continue
            try:
                manifest.append(process_triplet(video_path, audio_path, transcript_path, output_path, args, extractor))
            except Exception as exc:
                failures.append(
                    {
                        "video": str(video_path),
                        "audio": str(audio_path),
                        "transcript": str(transcript_path),
                        "error": repr(exc),
                    }
                )
                print(f"[fail] {video_path}: {exc}")
    finally:
        extractor.close()

    summary = {
        "raw_dir": str(args.raw_dir),
        "output_dir": str(output_dir),
        "total_triplets": len(triplets),
        "ok": len([x for x in manifest if not x.get("skipped")]),
        "skipped": len([x for x in manifest if x.get("skipped")]),
        "failed": len(failures),
        "success_rate": (len(triplets) - len(failures)) / max(1, len(triplets)),
        "config": vars(args),
        "items": manifest,
        "failures": failures,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] ok_or_skipped={len(triplets) - len(failures)}/{len(triplets)} success_rate={summary['success_rate']:.3f}")
    print(f"[out] {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild r2inr_text_v1 cache from folders containing video.mp4, audio.wav, transcript.txt.")
    parser.add_argument("--raw-dir", default="Dataset_Output_FrontalSpeaking_Filter/kept")
    parser.add_argument("--output-dir", default="Processed_Data_TextV1")
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--margin", type=float, default=1.8)
    parser.add_argument("--force-fps", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true", help="Use center crop fallback for every frame.")
    parser.add_argument("--face-landmarker-model", default="face_landmarker_v2_with_blendshapes.task")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

