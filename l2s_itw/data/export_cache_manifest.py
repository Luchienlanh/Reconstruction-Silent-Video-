from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export existing tensor caches into l2s_itw VTP-ready files.")
    parser.add_argument("--cache-dir", required=True, help="Directory containing .pt sample caches.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frame-size", type=int, default=160, help="Video size expected by VTP VideoReader.")
    parser.add_argument("--crop-size", type=int, default=96, help="Centered crop size expected by VTP augmentor.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_video_tensor(video: torch.Tensor) -> np.ndarray:
    video = video.detach().cpu().float()
    if video.ndim == 4 and video.shape[0] in {1, 3}:
        video = video.permute(1, 2, 3, 0)
    elif video.ndim == 3:
        video = video.unsqueeze(-1)
    elif video.ndim != 4:
        raise ValueError(f"Unsupported video shape: {tuple(video.shape)}")

    arr = video.numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got shape {arr.shape}")

    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def center_pad(frames: np.ndarray, frame_size: int, crop_size: int) -> np.ndarray:
    if frames.shape[1] != crop_size or frames.shape[2] != crop_size:
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR))
        frames = np.stack(resized, axis=0)

    canvas = np.zeros((frames.shape[0], frame_size, frame_size, 3), dtype=np.uint8)
    offset = (frame_size - crop_size) // 2
    canvas[:, offset : offset + crop_size, offset : offset + crop_size] = frames
    return canvas


def write_mp4(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    mel_dir = output_dir / "mels"
    manifest_path = output_dir / "manifest.jsonl"

    files = sorted(cache_dir.glob("*.pt"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"No .pt files found in {cache_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in tqdm(files, desc="export cache"):
        sample = torch.load(path, map_location="cpu")
        sample_id = path.stem
        video_path = video_dir / f"{sample_id}.mp4"
        mel_path = mel_dir / f"{sample_id}.mel.pt"

        if args.overwrite or not video_path.exists():
            frames = normalize_video_tensor(sample["video"])
            frames = center_pad(frames, frame_size=int(args.frame_size), crop_size=int(args.crop_size))
            write_mp4(video_path, frames, fps=float(sample.get("fps", args.fps)))

        if args.overwrite or not mel_path.exists():
            mel = sample["mel"].detach().cpu().float()
            torch.save({"mel": mel}, mel_path)

        rows.append(
            {
                "id": sample_id,
                "video_path": str(video_path.relative_to(output_dir)),
                "mel_path": str(mel_path.relative_to(output_dir)),
                "text": str(sample.get("transcript_text", "")),
                "source_cache": str(path.resolve()),
                "source_video": str(sample.get("source_video", "")),
                "fps": float(sample.get("fps", args.fps)),
                "sample_rate": int(sample.get("sample_rate", 16000)),
                "hop_length": int(sample.get("hop_length", 160)),
            }
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
