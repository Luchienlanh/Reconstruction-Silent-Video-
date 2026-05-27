from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV6.data import FullClipCTCDataset, cer, collate_ctc, greedy_decode, split_cache_files, wer
from srcV6.models import build_model_from_config
from srcV6.training.train_ctc import ctc_loss, sanitize_batch
from srcV6.utils import batch_to_device, get_device


def clone_batch(batch: dict) -> dict:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def apply_variant(batch: dict, variant: str) -> dict:
    batch = clone_batch(batch)
    if variant in ("zero_video", "zero_both"):
        batch["video"].zero_()
    if variant in ("zero_landmarks", "zero_both"):
        batch["landmarks"].zero_()
    if variant == "reverse_time":
        batch["video"] = batch["video"].flip(dims=[2])
        batch["landmarks"] = batch["landmarks"].flip(dims=[1])
    if variant == "mismatch_batch" and batch["video"].shape[0] > 1:
        perm = torch.roll(torch.arange(batch["video"].shape[0], device=batch["video"].device), shifts=1)
        batch["video"] = batch["video"][perm]
        batch["landmarks"] = batch["landmarks"][perm]
    return batch


@torch.no_grad()
def eval_variant(model, loader, criterion, device, vocab: dict[str, int], variant: str, max_samples: int) -> dict:
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_count = 0
    samples = []
    for batch in tqdm(loader, desc=variant, leave=False):
        batch = sanitize_batch(batch_to_device(batch, device))
        batch = apply_variant(batch, variant)
        logits = model(batch)
        loss = ctc_loss(logits, batch, criterion)
        pred_ids = logits.float().argmax(dim=-1).detach().cpu()
        for i, ref in enumerate(batch["transcript_texts"]):
            length = int(batch["video_lengths"][i].detach().cpu().item())
            hyp = greedy_decode(pred_ids[i, :length].tolist(), vocab)
            c = cer(ref, hyp)
            w = wer(ref, hyp)
            total_cer += c
            total_wer += w
            total_count += 1
            if len(samples) < max_samples:
                samples.append({"ref": ref, "hyp": hyp, "cer": c, "wer": w})
        total_loss += float(loss.detach().cpu())
    return {
        "loss": total_loss / max(1, len(loader)),
        "cer": total_cer / max(1, total_count),
        "wer": total_wer / max(1, total_count),
        "samples": samples,
    }


def safe_print(text: str) -> None:
    import sys

    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate srcV6 CTC checkpoint to verify input dependence.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=0, help="Default: use checkpoint config.")
    parser.add_argument("--min-input-target-ratio", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print-samples", type=int, default=1)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    vocab = {str(k): int(v) for k, v in ckpt["vocab"].items()}
    model = build_model_from_config(cfg, len(vocab)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    safe_print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        safe_print(f"[checkpoint] missing sample: {missing[:8]}")
    if unexpected:
        safe_print(f"[checkpoint] unexpected sample: {unexpected[:8]}")
    model.eval()
    frame_stride = int(args.frame_stride or cfg.get("frame_stride", 1))
    text_unit = str(cfg.get("text_unit", "char"))
    limit_files = args.limit_files if args.limit_files > 0 else None
    files, _ = split_cache_files(args.data_dir, val_ratio=0.0, seed=int(cfg.get("seed", 42)), limit_files=limit_files)
    dataset = FullClipCTCDataset(
        args.data_dir,
        vocab=vocab,
        files=files,
        frame_stride=frame_stride,
        min_input_target_ratio=args.min_input_target_ratio,
        text_unit=text_unit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_ctc,
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    variants = ["normal", "zero_video", "zero_landmarks", "zero_both", "reverse_time", "mismatch_batch"]
    results = {variant: eval_variant(model, loader, criterion, device, vocab, variant, args.print_samples) for variant in variants}
    normal = results["normal"]
    safe_print(f"[data] items={len(dataset)} text_unit={text_unit} frame_stride={frame_stride}")
    safe_print("variant              loss      cer      wer  delta_cer  delta_wer")
    for variant in variants:
        row = results[variant]
        safe_print(
            f"{variant:<16} {row['loss']:9.4f} {row['cer']:8.4f} {row['wer']:8.4f} "
            f"{row['cer'] - normal['cer']:+10.4f} {row['wer'] - normal['wer']:+10.4f}"
        )
    for sample in normal["samples"][: args.print_samples]:
        safe_print(f"[normal ref] {sample['ref'][:180]}")
        safe_print(f"[normal hyp] {sample['hyp'][:180]}")


if __name__ == "__main__":
    run(parse_args())

