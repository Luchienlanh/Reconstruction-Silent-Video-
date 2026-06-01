from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from srcV2.data.build_cache import LipLandmarkExtractor, decode_video_with_mouth_crops, scan_raw_pairs
from srcV2.utils.audio import log_mel_from_audio
from srcV2.utils.common import safe_name, seed_everything
from srcV3.data.window_dataset import mel_indices_for_video_window, window_starts


TEXT_NAMES = (
    "transcript.txt",
    "transcript_clean.txt",
    "text.txt",
    "sentence.txt",
    "label.txt",
)


def read_optional_transcript(video_path: Path) -> str:
    folder = video_path.parent
    for name in TEXT_NAMES:
        path = folder / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore").strip()
    txt_files = sorted(folder.glob("*.txt"))
    if txt_files:
        return txt_files[0].read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def build_v3_window_index(item: dict[str, Any], window_frames: int, hop_frames: int) -> dict[str, Any]:
    starts = window_starts(int(item["video_len"]), int(window_frames), int(hop_frames))
    ranges = []
    for start in starts:
        end = min(int(item["video_len"]), int(start) + int(window_frames))
        mel_idx = mel_indices_for_video_window(item, int(start), int(end))
        if mel_idx.numel() <= 0:
            ranges.append((0, 1))
        else:
            ranges.append((int(mel_idx[0].item()), int(mel_idx[-1].item()) + 1))
    if ranges:
        mel_ranges = torch.tensor(ranges, dtype=torch.long)
    else:
        mel_ranges = torch.zeros(0, 2, dtype=torch.long)
    return {
        "window_frames": int(window_frames),
        "hop_frames": int(hop_frames),
        "starts": torch.tensor([int(x) for x in starts], dtype=torch.long),
        "mel_ranges": mel_ranges,
        "count": int(len(starts)),
    }


def process_pair(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    extractor: LipLandmarkExtractor,
) -> dict[str, Any]:
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
    payload: dict[str, Any] = {
        "format": "r2inr_v1",
        "cache_profile": "v2_v3_merged",
        "video": video.contiguous(),
        "landmarks": landmarks.contiguous(),
        "mel": mel.contiguous(),
        "video_len": int(video.shape[1]),
        "mel_len": int(mel.shape[0]),
        "fps": float(fps),
        "sample_rate": int(args.sample_rate),
        "hop_length": int(args.hop_length),
        "win_length": int(args.win_length),
        "n_fft": int(args.n_fft),
        "n_mels": int(args.n_mels),
        "video_times": video_times.contiguous(),
        "mel_times": mel_times.contiguous(),
        "mouth_valid_mask": valid_mask.contiguous(),
        "crop_boxes": crop_boxes.contiguous(),
        "source_video": str(video_path),
        "source_audio": str(audio_path),
        "transcript": read_optional_transcript(video_path),
        "v2_cache": {
            "compatible": True,
            "full_clip": True,
            "time_aligned": True,
        },
    }
    payload["v3_window_index"] = build_v3_window_index(payload, args.window_frames, args.hop_frames)
    torch.save(payload, output_path)
    return {
        "file": str(output_path),
        "video": str(video_path),
        "audio": str(audio_path),
        "video_len": int(video.shape[1]),
        "mel_len": int(mel.shape[0]),
        "fps": float(fps),
        "valid_ratio": float(valid_mask.float().mean().item()),
        "v3_windows": int(payload["v3_window_index"]["count"]),
        "has_transcript": bool(payload["transcript"]),
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = scan_raw_pairs(args.raw_dir)
    if args.limit is not None:
        pairs = pairs[: max(1, min(int(args.limit), len(pairs)))]
    if not pairs:
        raise RuntimeError(f"No video.mp4/audio.wav pairs found under {args.raw_dir}")

    extractor = LipLandmarkExtractor(
        enabled=not args.no_mediapipe,
        model_path=args.face_landmarker_model,
        auto_download=not args.no_download_face_landmarker,
    )
    manifest = []
    failures = []
    try:
        for video_path, audio_path in tqdm(pairs, desc="build-v2v3-cache"):
            out_name = safe_name(video_path) + ".pt"
            output_path = output_dir / out_name
            if output_path.exists() and not args.overwrite:
                manifest.append({"file": str(output_path), "skipped": True})
                continue
            try:
                manifest.append(process_pair(video_path, audio_path, output_path, args, extractor))
            except Exception as exc:
                failures.append({"video": str(video_path), "audio": str(audio_path), "error": repr(exc)})
                print(f"[fail] {video_path}: {exc}")
    finally:
        extractor.close()

    ok = len([x for x in manifest if not x.get("skipped")])
    skipped = len([x for x in manifest if x.get("skipped")])
    total_windows = sum(int(x.get("v3_windows", 0)) for x in manifest)
    summary = {
        "format": "r2inr_v1",
        "cache_profile": "v2_v3_merged",
        "raw_dir": str(args.raw_dir),
        "output_dir": str(output_dir),
        "total_pairs": len(pairs),
        "ok": ok,
        "skipped": skipped,
        "failed": len(failures),
        "success_rate": (len(pairs) - len(failures)) / max(1, len(pairs)),
        "v3_window_frames": int(args.window_frames),
        "v3_hop_frames": int(args.hop_frames),
        "total_v3_windows": int(total_windows),
        "config": vars(args),
        "items": manifest,
        "failures": failures,
    }
    (output_dir / "manifest_v2v3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] ok_or_skipped={len(pairs) - len(failures)}/{len(pairs)} success_rate={summary['success_rate']:.3f}")
    print(f"[windows] total_v3_windows={total_windows} frames={args.window_frames} hop={args.hop_frames}")
    print(f"[out] {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one cache format usable by srcV2 full-clip and srcV3 window training.")
    parser.add_argument("--raw-dir", default="New_video")
    parser.add_argument("--output-dir", default="Processed_Data_R2INR_NewVideo_V2V3")
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--margin", type=float, default=1.8)
    parser.add_argument("--force-fps", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--window-frames", type=int, default=45)
    parser.add_argument("--hop-frames", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true", help="Use center crop fallback for every frame.")
    parser.add_argument("--face-landmarker-model", default="face_landmarker_v2_with_blendshapes.task")
    parser.add_argument(
        "--no-download-face-landmarker",
        action="store_true",
        help="Do not auto-download the default MediaPipe FaceLandmarker task model when it is missing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
