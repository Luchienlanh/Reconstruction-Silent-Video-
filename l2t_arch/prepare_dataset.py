from __future__ import annotations

import argparse
import json
import random
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen-VTP architecture manifests from existing VTP cache.")
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-conf", type=int, default=0)
    parser.add_argument("--min-seconds", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def read_manifest(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def parse_lrs2_text(path: str | Path) -> tuple[str, int | None]:
    text_path = Path(path)
    raw = text_path.read_text(encoding="utf-8", errors="replace")
    text = ""
    conf = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("text:"):
            text = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("conf:"):
            match = re.search(r"\d+", stripped)
            conf = int(match.group(0)) if match else None
    if not text:
        for line in raw.splitlines():
            if line.strip():
                text = line.strip()
                break
    return text, conf


def seconds(row: dict[str, Any]) -> float | None:
    if row.get("mel_frames") and row.get("hop_length") and row.get("sample_rate"):
        return float(row["mel_frames"]) * float(row["hop_length"]) / float(row["sample_rate"])
    return None


def resolve_path(path_value: str, base_dir: Path) -> str:
    path = Path(path_value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _process_row(row: dict[str, Any], cache_dir: Path, min_conf: int, min_seconds: float) -> tuple[dict[str, Any] | None, str | None]:
    visual_path = Path(resolve_path(str(row["visual_feature_path"]), cache_dir))
    if not visual_path.exists():
        return None, "missing_visual"
    if not row.get("source_text_path"):
        return None, "missing_gt"
    gt_text, conf = parse_lrs2_text(row["source_text_path"])
    if not gt_text:
        return None, "missing_gt"
    if min_conf and (conf is None or conf < min_conf):
        return None, "low_conf"
    sample_seconds = seconds(row)
    if min_seconds and sample_seconds is not None and sample_seconds < min_seconds:
        return None, "short"

    res = {
        "id": row["id"],
        "visual_feature_path": str(visual_path),
        "vtp_text": str(row.get("text", "")),
        "gt_text": gt_text,
        "conf": conf,
        "split": row.get("split", ""),
        "seconds": sample_seconds,
        "source_text_path": row.get("source_text_path", ""),
        "source_video_path": row.get("source_video_path") or row.get("video_path", ""),
    }
    return res, None


def main() -> None:
    args = parse_args()
    cache_manifest = Path(args.cache_manifest)
    cache_dir = cache_manifest.parent
    rows = read_manifest(cache_manifest, limit=args.limit)
    prepared = []
    skipped = {"missing_visual": 0, "missing_gt": 0, "low_conf": 0, "short": 0}

    # Tối ưu cho CPU AMD Ryzen 5 5600X (12 threads) và RAM 16GB:
    # Sử dụng ThreadPoolExecutor thay vì ProcessPoolExecutor để tránh copy dữ liệu (IPC) gây tốn RAM.
    # Vì quá trình này chủ yếu là I/O bound (đọc ổ đĩa), ta có thể dùng số luồng (threads) gấp đôi số luồng CPU (24 threads) để tối đa hoá tốc độ đọc đĩa.
    max_workers = min((os.cpu_count() or 6) * 2, 24)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_row, row, cache_dir, args.min_conf, args.min_seconds)
            for row in rows
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"prepare l2t_arch (w={max_workers})"):
            res, skip_reason = future.result()
            if skip_reason:
                skipped[skip_reason] += 1
            elif res is not None:
                prepared.append(res)

    if len(prepared) < 3:
        raise ValueError(f"Need at least 3 usable rows, got {len(prepared)}. skipped={skipped}")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val-ratio and test-ratio must be non-negative and sum to < 1.")

    rng = random.Random(args.seed)
    rng.shuffle(prepared)
    test_count = max(1, int(round(len(prepared) * args.test_ratio)))
    val_count = max(1, int(round(len(prepared) * args.val_ratio)))
    test_rows = prepared[:test_count]
    val_rows = prepared[test_count : test_count + val_count]
    train_rows = prepared[test_count + val_count :]

    output_dir = Path(args.output_dir)
    write_rows(train_rows, output_dir / "train.jsonl")
    write_rows(val_rows, output_dir / "val.jsonl")
    write_rows(test_rows, output_dir / "test.jsonl")
    summary = {
        "cache_manifest": str(cache_manifest),
        "rows_in": len(rows),
        "rows_out": len(prepared),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "min_conf": args.min_conf,
        "min_seconds": args.min_seconds,
        "skipped": skipped,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

