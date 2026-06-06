from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_eval(path: str | Path) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        return dict(data.get("summary", {})), list(data.get("rows", []))
    if isinstance(data, list):
        return {}, list(data)
    raise ValueError(f"Unsupported eval JSON format: {path}")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            feature_file = str(row.get("feature_file", "")).strip()
            if feature_file:
                f.write(feature_file + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(value: int, total: int) -> str:
    return f"{value / max(1, total) * 100:.1f}%"


def run(args: argparse.Namespace) -> None:
    summary, rows = load_eval(args.eval_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in rows if str(r.get("feature_file", "")).strip()]
    rows.sort(key=lambda r: (float(r.get("cer", 999.0)), float(r.get("wer", 999.0))))
    clean = [
        r
        for r in rows
        if float(r.get("cer", 999.0)) <= args.clean_cer and float(r.get("wer", 999.0)) <= args.clean_wer
    ]
    medium = [
        r
        for r in rows
        if r not in clean
        and float(r.get("cer", 999.0)) <= args.review_cer
        and float(r.get("wer", 999.0)) <= args.review_wer
    ]
    noisy = [r for r in rows if r not in clean and r not in medium]

    write_manifest(output_dir / "clean_manifest.txt", clean)
    write_manifest(output_dir / "review_manifest.txt", medium)
    write_manifest(output_dir / "usable_manifest.txt", clean + medium)
    write_jsonl(output_dir / "clean.jsonl", clean)
    write_jsonl(output_dir / "review.jsonl", medium)
    write_jsonl(output_dir / "noisy.jsonl", noisy)

    bins = {}
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5):
        bins[f"cer<={threshold}"] = sum(1 for r in rows if float(r.get("cer", 999.0)) <= threshold)
    out = {
        "source_summary": summary,
        "total": len(rows),
        "clean": len(clean),
        "review": len(medium),
        "noisy": len(noisy),
        "clean_ratio": pct(len(clean), len(rows)),
        "usable_ratio": pct(len(clean) + len(medium), len(rows)),
        "thresholds": {
            "clean_cer": args.clean_cer,
            "clean_wer": args.clean_wer,
            "review_cer": args.review_cer,
            "review_wer": args.review_wer,
        },
        "bins": bins,
        "outputs": {
            "clean_manifest": str(output_dir / "clean_manifest.txt"),
            "usable_manifest": str(output_dir / "usable_manifest.txt"),
            "noisy_jsonl": str(output_dir / "noisy.jsonl"),
        },
    }
    (output_dir / "filter_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean/noisy manifests from official AV-HuBERT VSR eval JSON.")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-cer", type=float, default=0.20)
    parser.add_argument("--clean-wer", type=float, default=0.35)
    parser.add_argument("--review-cer", type=float, default=0.35)
    parser.add_argument("--review-wer", type=float, default=0.70)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
