from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from l2s_itw.data.manifest import read_manifest


PATH_KEYS = [
    "visual_feature_path",
    "mel_path",
    "speaker_embedding_path",
    "video_path",
    "source_video_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a JSONL manifest into train and validation manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def rebase_path_fields(rows: list[dict], source_dir: Path) -> list[dict]:
    rebased = []
    for row in rows:
        new_row = dict(row)
        for key in PATH_KEYS:
            if key not in new_row or not new_row[key]:
                continue
            path = Path(str(new_row[key]))
            if not path.is_absolute():
                new_row[key] = str((source_dir / path).resolve())
        rebased.append(new_row)
    return rebased


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    rows = rebase_path_fields(read_manifest(manifest_path), manifest_path.parent)
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows to split a manifest.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_count = max(1, int(round(len(rows) * args.val_ratio)))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    if not train_rows:
        train_rows, val_rows = rows[:-1], rows[-1:]

    output_dir = Path(args.output_dir)
    write_rows(train_rows, output_dir / "train.jsonl")
    write_rows(val_rows, output_dir / "val.jsonl")
    print(f"train: {len(train_rows)} -> {output_dir / 'train.jsonl'}")
    print(f"val: {len(val_rows)} -> {output_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
