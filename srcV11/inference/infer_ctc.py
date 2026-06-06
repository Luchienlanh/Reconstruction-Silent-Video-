from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from srcV11.data import CHAR_EN_VOCAB, cer, greedy_decode_with_confidence, load_feature_cache, normalize_english, split_feature_files, wer
from srcV11.models import build_model_from_config
from srcV11.utils import get_device


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    vocab = ckpt.get("vocab", CHAR_EN_VOCAB)
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    if args.feature_file:
        sample = Path(args.feature_file)
    else:
        train_files, val_files = split_feature_files(
            args.feature_dir,
            val_ratio=args.val_ratio,
            seed=int(config.get("seed", args.seed)),
            limit_files=args.limit_files if args.limit_files > 0 else None,
        )
        files = val_files if args.split == "val" and val_files else train_files
        sample = files[max(0, min(int(args.sample_index), len(files) - 1))]

    item = load_feature_cache(sample)
    features = item["features"].float().unsqueeze(0).to(device)
    feature_len = int(item.get("feature_len", features.shape[1]))
    feature_mask = torch.arange(features.shape[1], device=device).unsqueeze(0) < feature_len
    logits = model(features, feature_mask)
    probs = torch.softmax(logits.float(), dim=-1)
    max_probs, pred_ids = probs.max(dim=-1)
    out_len = int(min(model.output_lengths(torch.tensor([feature_len], device=device))[0].item(), pred_ids.shape[1]))
    pred_text, confidence = greedy_decode_with_confidence(
        pred_ids[0, :out_len].detach().cpu().tolist(),
        max_probs[0, :out_len].detach().cpu().tolist(),
        vocab,
    )
    ref_text = normalize_english(str(item.get("transcript_text", "")))
    metrics = {}
    if ref_text:
        metrics = {"cer": cer(ref_text, pred_text), "wer": wer(ref_text, pred_text)}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sample.stem
    (output_dir / f"{stem}_predicted_text.txt").write_text(pred_text + "\n", encoding="utf-8")
    meta = {
        "checkpoint": str(args.checkpoint),
        "feature_file": str(sample),
        "source_video": item.get("source_video", ""),
        "source_cache": item.get("source_cache", ""),
        "feature_len": feature_len,
        "output_len": out_len,
        "reference_text": ref_text,
        "predicted_text": pred_text,
        "confidence": float(confidence),
        **metrics,
    }
    (output_dir / f"{stem}_confidence.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_logits:
        torch.save({"logits": logits.detach().cpu(), "pred_ids": pred_ids.detach().cpu(), "meta": meta}, output_dir / f"{stem}_logits.pt")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer srcV11 lip-to-text CTC from cached AV-HuBERT features.")
    parser.add_argument("--feature-dir", "--data-dir", dest="feature_dir", default="Processed_Data_AVHubertFeatures_LRS2_10k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="infer_srcV11_ctc")
    parser.add_argument("--feature-file", default="")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
