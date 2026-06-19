from __future__ import annotations

import argparse
from pathlib import Path

from l2t_vtp.data import read_lrs2_text, read_manifest, sample_seconds, write_json, write_jsonl
from l2t_vtp.metrics import edit_distance
from l2t_vtp.text import chars, normalize_text, words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached VTP lip-to-text predictions with WER/CER.")
    parser.add_argument("--pred-manifest", required=True, help="VTP cache manifest. Its text field is the prediction.")
    parser.add_argument("--gt-manifest", help="Optional manifest with ground-truth text matched by id.")
    parser.add_argument("--output-dir", default="reports_l2t_vtp_eval")
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--pred-text-key", default="text")
    parser.add_argument("--gt-text-key", default="text")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-seconds", type=float, default=0.0)
    parser.add_argument("--split", default="", help="Optional split filter, e.g. main or pretrain.")
    parser.add_argument("--examples", type=int, default=50)
    return parser.parse_args()


def build_gt_map(path: str | None, id_key: str, text_key: str) -> dict[str, str]:
    if not path:
        return {}
    return {str(row[id_key]): str(row.get(text_key, "")) for row in read_manifest(path)}


def ground_truth_for(row: dict, gt_by_id: dict[str, str], id_key: str, text_key: str) -> str:
    sample_id = str(row.get(id_key, ""))
    if sample_id in gt_by_id:
        return gt_by_id[sample_id]
    if row.get("source_text_path"):
        return read_lrs2_text(row["source_text_path"])
    if row.get("gt_text"):
        return str(row["gt_text"])
    return str(row.get(text_key, ""))


def main() -> None:
    args = parse_args()
    pred_rows = read_manifest(args.pred_manifest, limit=args.limit)
    gt_by_id = build_gt_map(args.gt_manifest, args.id_key, args.gt_text_key)

    total_word_edits = 0
    total_words = 0
    total_char_edits = 0
    total_chars = 0
    exact = 0
    evaluated = 0
    skipped = 0
    examples = []

    for row in pred_rows:
        if args.split and str(row.get("split", "")) != args.split:
            skipped += 1
            continue
        seconds = sample_seconds(row)
        if args.min_seconds and seconds is not None and seconds < args.min_seconds:
            skipped += 1
            continue

        pred_text = str(row.get(args.pred_text_key, ""))
        gt_text = ground_truth_for(row, gt_by_id, args.id_key, args.gt_text_key)

        ref_words = words(gt_text)
        hyp_words = words(pred_text)
        ref_chars = chars(gt_text)
        hyp_chars = chars(pred_text)
        if not ref_words and not ref_chars:
            skipped += 1
            continue

        word_edits = edit_distance(ref_words, hyp_words)
        char_edits = edit_distance(ref_chars, hyp_chars)
        total_word_edits += word_edits
        total_words += max(len(ref_words), 1)
        total_char_edits += char_edits
        total_chars += max(len(ref_chars), 1)
        evaluated += 1

        gt_norm = normalize_text(gt_text)
        pred_norm = normalize_text(pred_text)
        if gt_norm == pred_norm:
            exact += 1

        wer = word_edits / max(len(ref_words), 1)
        cer = char_edits / max(len(ref_chars), 1)
        examples.append(
            {
                "id": row.get(args.id_key),
                "split": row.get("split"),
                "seconds": seconds,
                "wer": wer,
                "cer": cer,
                "gt": gt_text,
                "pred": pred_text,
                "gt_norm": gt_norm,
                "pred_norm": pred_norm,
            }
        )

    summary = {
        "pred_manifest": str(Path(args.pred_manifest)),
        "gt_manifest": str(Path(args.gt_manifest)) if args.gt_manifest else None,
        "evaluated": evaluated,
        "skipped": skipped,
        "wer": total_word_edits / max(total_words, 1),
        "cer": total_char_edits / max(total_chars, 1),
        "exact_match": exact / max(evaluated, 1),
        "total_word_edits": total_word_edits,
        "total_words": total_words,
        "total_char_edits": total_char_edits,
        "total_chars": total_chars,
        "min_seconds": float(args.min_seconds),
        "split": args.split or None,
    }

    examples.sort(key=lambda item: (item["wer"], item["cer"]), reverse=True)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "worst_examples.jsonl", examples[: max(0, int(args.examples))])

    print(f"evaluated={evaluated} skipped={skipped}")
    print(f"WER={summary['wer']:.4f} CER={summary['cer']:.4f} exact={summary['exact_match']:.4f}")
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'worst_examples.jsonl'}")


if __name__ == "__main__":
    main()

