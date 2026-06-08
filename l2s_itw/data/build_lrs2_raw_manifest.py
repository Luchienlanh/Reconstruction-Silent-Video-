from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm

from l2s_itw.data.manifest import write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JSONL manifest that points directly to raw LRS2 mp4 files.")
    parser.add_argument("--data-dir", required=True, help="Root containing LRS2 .mp4/.txt pairs.")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--skip-missing-text", action="store_true")
    return parser.parse_args()


def read_lrs2_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("text:"):
            return stripped.split(":", 1)[1].strip()
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def make_id(video_path: Path, data_dir: Path) -> str:
    rel = video_path.relative_to(data_dir).with_suffix("")
    parts = list(rel.parts)
    lowered = [part.lower() for part in parts]
    start = 0
    for marker in ("main", "pretrain", "trainval", "test"):
        if marker in lowered:
            start = lowered.index(marker)
            break
    raw_id = "_".join(parts[start:])
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id)


def infer_split(video_path: Path) -> str:
    lowered = [part.lower() for part in video_path.parts]
    for marker in ("main", "pretrain", "trainval", "test"):
        if marker in lowered:
            return marker
    return ""


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    videos = sorted(data_dir.rglob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise ValueError(f"No .mp4 files found in {data_dir}")

    rows = []
    missing_text = 0
    for video_path in tqdm(videos, desc="raw LRS2 manifest"):
        text_path = video_path.with_suffix(".txt")
        if not text_path.exists():
            missing_text += 1
            if args.skip_missing_text:
                continue
            raise FileNotFoundError(f"Missing transcript for {video_path}: {text_path}")

        rows.append(
            {
                "id": make_id(video_path, data_dir),
                "video_path": str(video_path.resolve()),
                "text": read_lrs2_text(text_path),
                "speaker_id": video_path.parent.name,
                "utterance_id": video_path.stem,
                "split": infer_split(video_path),
                "source_text_path": str(text_path.resolve()),
                "fps": float(args.fps),
                "sample_rate": int(args.sample_rate),
            }
        )

    write_manifest(rows, args.output_manifest)
    print(f"wrote {args.output_manifest}")
    print(f"rows: {len(rows)}")
    if missing_text:
        print(f"missing_text: {missing_text}")


if __name__ == "__main__":
    main()
