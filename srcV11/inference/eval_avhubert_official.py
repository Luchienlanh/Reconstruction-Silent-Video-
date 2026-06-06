from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm.auto import tqdm

from srcV8.data.video_dataset import load_text_cache
from srcV8.models import AVHubertVisualFeatureExtractor
from srcV11.data import cer, load_feature_cache, normalize_english, split_feature_files, wer
from srcV11.inference.infer_avhubert_official import configure_task, decode_ctc, decode_s2s, make_source
from srcV11.utils import get_device


def select_feature_files(args: argparse.Namespace) -> list[Path]:
    train_files, val_files = split_feature_files(
        args.feature_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_files=args.limit_files if args.limit_files > 0 else None,
    )
    files = val_files if args.split == "val" and val_files else train_files
    if args.max_samples > 0:
        files = files[: int(args.max_samples)]
    if not files:
        raise RuntimeError("No feature files selected.")
    return files


def load_item_from_feature(feature_path: Path):
    feature = load_feature_cache(feature_path)
    source_cache = str(feature.get("source_cache", ""))
    if not source_cache:
        raise ValueError(f"{feature_path} does not contain source_cache.")
    item = load_text_cache(source_cache)
    ref = normalize_english(str(feature.get("transcript_text", item.get("transcript_text", ""))))
    return item, ref


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = select_feature_files(args)

    extractor = AVHubertVisualFeatureExtractor(
        checkpoint=args.checkpoint,
        avhubert_dir=args.avhubert_dir,
        freeze=True,
        normalize_mode=args.normalize_mode,
        crop_size=args.crop_size,
    ).to(device)
    model = extractor.model.to(device).eval()
    task = configure_task(extractor.fairseq_task, extractor.fairseq_cfg, ["video"])
    use_s2s = hasattr(model, "decoder") and hasattr(model, "encoder")

    rows = []
    total_cer = 0.0
    total_wer = 0.0
    for feature_path in tqdm(files, desc="official-vsr-eval"):
        item, ref = load_item_from_feature(feature_path)
        source, padding_mask = make_source(extractor, item, device)
        if use_s2s:
            hyp, meta = decode_s2s(model, task, source, padding_mask, args)
        else:
            hyp, meta = decode_ctc(model, task, source, padding_mask)
        hyp_norm = normalize_english(hyp)
        c = cer(ref, hyp_norm)
        w = wer(ref, hyp_norm)
        row = {
            "feature_file": str(feature_path),
            "source_cache": str(load_feature_cache(feature_path).get("source_cache", "")),
            "source_video": str(item.get("source_video", "")),
            "ref": ref,
            "hyp": hyp_norm,
            "raw_hyp": hyp,
            "cer": c,
            "wer": w,
            **meta,
        }
        rows.append(row)
        total_cer += c
        total_wer += w
        print(f"[{len(rows):04d}] cer={c:.3f} wer={w:.3f} ref={ref[:90]} hyp={hyp_norm[:90]}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "feature_dir": str(args.feature_dir),
        "split": args.split,
        "count": len(rows),
        "cer": total_cer / max(1, len(rows)),
        "wer": total_wer / max(1, len(rows)),
        "preprocess": {"crop_size": int(args.crop_size), "normalize_mode": str(args.normalize_mode)},
        "decode_mode": rows[0].get("decode_mode", "") if rows else "",
    }
    (output_dir / "official_vsr_eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official AV-HuBERT VSR decoder on cached V11 feature files.")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--avhubert-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="eval_avhubert_official")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam", type=int, default=20)
    parser.add_argument("--lenpen", type=float, default=1.0)
    parser.add_argument("--max-len-a", type=float, default=1.0)
    parser.add_argument("--max-len-b", type=int, default=20)
    parser.add_argument("--normalize-mode", choices=["avhubert", "per_frame", "none"], default="avhubert")
    parser.add_argument("--crop-size", type=int, default=88)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
