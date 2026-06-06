from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm.auto import tqdm

from srcV11.data import load_feature_cache, normalize_english, text_to_ids


def iter_feature_files(feature_dir: str | Path, limit_files: int = 0) -> list[Path]:
    files = sorted(Path(feature_dir).glob("*.pt"))
    if limit_files and int(limit_files) > 0:
        files = files[: max(1, min(int(limit_files), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt feature files found under {feature_dir}")
    return files


def check_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    item = load_feature_cache(path)
    text = normalize_english(str(item.get("transcript_text", "")))
    ids = text_to_ids(text)
    feature_len = int(item.get("feature_len", item["features"].shape[0]))
    effective_input = feature_len * max(1, int(args.input_length_factor))
    required = max(1, int(round(len(ids) * float(args.min_input_target_ratio))))
    if not text or not ids:
        return {"path": str(path), "ok": False, "reason": "empty_transcript"}
    if args.max_target_len > 0 and len(ids) > args.max_target_len:
        return {"path": str(path), "ok": False, "reason": f"target_too_long:{len(ids)}>{args.max_target_len}"}
    if args.max_feature_len > 0 and feature_len > args.max_feature_len:
        return {"path": str(path), "ok": False, "reason": f"feature_too_long:{feature_len}>{args.max_feature_len}"}
    if effective_input < required:
        return {"path": str(path), "ok": False, "reason": f"ctc_too_short:{effective_input}<{required}"}
    return {
        "path": str(path),
        "ok": True,
        "feature_len": feature_len,
        "target_len": len(ids),
        "text": text,
    }


def split_files(files: list[Path], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n = len(files)
    test_count = int(round(n * max(0.0, min(0.9, float(test_ratio)))))
    val_count = int(round(n * max(0.0, min(0.9, float(val_ratio)))))
    if n > 1 and test_ratio > 0:
        test_count = max(1, test_count)
    if n - test_count > 1 and val_ratio > 0:
        val_count = max(1, val_count)
    if test_count + val_count >= n:
        if test_count > 0:
            test_count = max(0, test_count - 1)
        if test_count + val_count >= n and val_count > 0:
            val_count = max(0, val_count - 1)
    test = sorted(files[:test_count])
    val = sorted(files[test_count : test_count + val_count])
    train = sorted(files[test_count + val_count :])
    if not train:
        train = val or test
        val = []
    return train, val, test


def manifest_line(path: Path, feature_dir: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(feature_dir.resolve()))
    except ValueError:
        return path.name


def write_manifest(path: Path, files: list[Path], feature_dir: Path, absolute: bool) -> None:
    text = "\n".join(manifest_line(file, feature_dir, absolute) for file in files)
    path.write_text((text + "\n") if text else "", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_feature_files(feature_dir, args.limit_files)
    usable: list[Path] = []
    skipped: list[dict[str, Any]] = []
    lengths = []
    for path in tqdm(files, desc="build-manifests"):
        try:
            result = check_file(path, args)
            if result.get("ok"):
                usable.append(path)
                lengths.append({"feature_len": result["feature_len"], "target_len": result["target_len"]})
            else:
                skipped.append(result)
        except Exception as exc:
            skipped.append({"path": str(path), "ok": False, "reason": repr(exc)})
    if not usable:
        raise RuntimeError("No usable feature files after manifest filtering.")

    train, val, test = split_files(usable, args.val_ratio, args.test_ratio, args.seed)
    write_manifest(output_dir / "all_manifest.txt", sorted(usable), feature_dir, args.absolute)
    write_manifest(output_dir / "train_manifest.txt", train, feature_dir, args.absolute)
    write_manifest(output_dir / "val_manifest.txt", val, feature_dir, args.absolute)
    write_manifest(output_dir / "test_manifest.txt", test, feature_dir, args.absolute)

    reason_counts = Counter(str(item.get("reason", "unknown")).split(":", 1)[0] for item in skipped)
    summary = {
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "total_files": len(files),
        "usable": len(usable),
        "skipped": len(skipped),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "reason_counts": dict(sorted(reason_counts.items())),
        "config": vars(args),
        "lengths": lengths,
        "skipped_items": skipped[:200],
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] usable={len(usable)} train={len(train)} val={len(val)} test={len(test)} skipped={len(skipped)}")
    print(f"[out] {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test manifests for srcV11 feature CTC training.")
    parser.add_argument("--feature-dir", default="Processed_Data_VisualFeatures_LRS2_10k")
    parser.add_argument("--output-dir", default="manifests_srcV11_lrs2")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--min-input-target-ratio", type=float, default=1.05)
    parser.add_argument("--input-length-factor", type=int, default=2)
    parser.add_argument("--max-target-len", type=int, default=0)
    parser.add_argument("--max-feature-len", type=int, default=0)
    parser.add_argument("--absolute", action="store_true", help="Write absolute paths instead of paths relative to --feature-dir.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
