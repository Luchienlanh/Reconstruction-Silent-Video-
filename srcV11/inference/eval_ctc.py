from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV11.data import (
    AVFeatureTextDataset,
    CHAR_EN_VOCAB,
    cer,
    collate_feature_text,
    greedy_decode_with_confidence,
    split_feature_files,
    wer,
)
from srcV11.models import build_model_from_config
from srcV11.utils import batch_to_device, get_device


def read_manifest(path: str | Path) -> list[Path]:
    manifest = Path(path)
    files = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        files.append(Path(line))
    if not files:
        raise RuntimeError(f"Manifest is empty: {manifest}")
    return files


def select_files(args: argparse.Namespace, config: dict) -> list[Path]:
    if args.manifest:
        return read_manifest(args.manifest)
    train_files, val_files = split_feature_files(
        args.feature_dir,
        val_ratio=args.val_ratio,
        seed=int(config.get("seed", args.seed)),
        limit_files=args.limit_files if args.limit_files > 0 else None,
    )
    if args.split == "train":
        return train_files
    if args.split == "val":
        return val_files or train_files
    return sorted(train_files + val_files)


def ctc_loss(logits: torch.Tensor, batch: dict, model: torch.nn.Module, criterion: torch.nn.CTCLoss) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1).contiguous()
    input_lengths = model.output_lengths(batch["feature_lengths"].to(log_probs.device)).clamp_max(log_probs.shape[0])
    target_lengths = batch["target_lengths"].to(log_probs.device).long()
    target_2d = batch["target_ids"].to(log_probs.device).long()
    targets = torch.cat([row[: int(length.item())] for row, length in zip(target_2d, target_lengths)], dim=0)
    return criterion(log_probs, targets, input_lengths, target_lengths)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    vocab = ckpt.get("vocab", CHAR_EN_VOCAB)
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    files = select_files(args, config)
    ds = AVFeatureTextDataset(
        args.feature_dir,
        files=files,
        vocab=vocab,
        min_input_target_ratio=args.min_input_target_ratio,
        input_length_factor=int(config.get("upsample_factor", args.input_length_factor)),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_feature_text,
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_conf = 0.0
    count = 0
    samples = []
    per_sample = []

    print(f"[device] {device}")
    print(f"[data] items={len(ds)} feature_dir={args.feature_dir}")
    for batch in tqdm(loader, desc="eval-ctc"):
        batch = batch_to_device(batch, device)
        logits = model(batch["features"], batch["feature_mask"])
        loss = ctc_loss(logits, batch, model, criterion)
        probs = torch.softmax(logits.float(), dim=-1)
        max_probs, pred_ids = probs.max(dim=-1)
        pred_ids = pred_ids.detach().cpu()
        max_probs = max_probs.detach().cpu()
        out_lengths = model.output_lengths(batch["feature_lengths"]).detach().cpu()
        total_loss += float(loss.detach().cpu())
        for i, ref in enumerate(batch["transcript_texts"]):
            length = int(min(out_lengths[i].item(), pred_ids.shape[1]))
            hyp, conf = greedy_decode_with_confidence(
                pred_ids[i, :length].tolist(),
                max_probs[i, :length].tolist(),
                vocab,
            )
            c = cer(ref, hyp)
            w = wer(ref, hyp)
            row = {
                "path": batch["paths"][i],
                "source_video": batch["source_videos"][i],
                "reference_text": ref,
                "predicted_text": hyp,
                "cer": c,
                "wer": w,
                "confidence": conf,
            }
            per_sample.append(row)
            if len(samples) < args.print_samples:
                samples.append(row)
            total_cer += c
            total_wer += w
            total_conf += conf
            count += 1

    result = {
        "checkpoint": str(args.checkpoint),
        "feature_dir": str(args.feature_dir),
        "manifest": str(args.manifest),
        "split": args.split,
        "items": count,
        "loss": total_loss / max(1, len(loader)),
        "cer": total_cer / max(1, count),
        "wer": total_wer / max(1, count),
        "confidence": total_conf / max(1, count),
        "samples": samples,
        "per_sample": per_sample,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ctc_eval.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, ensure_ascii=False, indent=2))
    print(f"[out] {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a srcV11 lip-to-text CTC checkpoint over a feature split.")
    parser.add_argument("--feature-dir", "--data-dir", dest="feature_dir", default="Processed_Data_VisualFeatures_LRS2_10k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="eval_srcV11_ctc")
    parser.add_argument("--manifest", default="", help="Optional feature manifest. Relative paths are resolved against --feature-dir.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-input-target-ratio", type=float, default=1.05)
    parser.add_argument("--input-length-factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-samples", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
