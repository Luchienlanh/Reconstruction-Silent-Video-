from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from srcV6.data import FullClipCTCDataset, build_char_vocab_from_files, cer, collate_ctc, greedy_decode, wer
from srcV6.models import LipreadingCTCModel
from srcV6.training.train_ctc import ctc_loss, parse_layers, sanitize_batch
from srcV6.utils import batch_to_device, get_device, seed_everything, write_json


def build_model(args: argparse.Namespace, device: torch.device, vocab_size: int) -> LipreadingCTCModel:
    return LipreadingCTCModel(
        vocab_size=vocab_size,
        dim=args.dim,
        num_landmark_points=args.num_landmark_points,
        fusion_type=args.fusion_type,
        encoder_width=args.encoder_width,
        resnet_layers=args.resnet_layers,
        visual_temporal_layers=args.visual_temporal_layers,
        landmark_temporal_layers=args.landmark_temporal_layers,
        dropout=args.dropout,
        blank_bias_init=args.blank_bias_init,
    ).to(device)


@torch.no_grad()
def eval_one(model, batch: dict, criterion, vocab: dict[str, int]) -> dict:
    model.eval()
    logits = model(batch)
    loss = ctc_loss(logits, batch, criterion)
    pred_ids = logits.float().argmax(dim=-1)[0, : int(batch["video_lengths"][0].item())].detach().cpu().tolist()
    ref = batch["transcript_texts"][0]
    hyp = greedy_decode(pred_ids, vocab)
    return {
        "loss": float(loss.detach().cpu()),
        "cer": cer(ref, hyp),
        "wer": wer(ref, hyp),
        "ref": ref,
        "hyp": hyp,
    }


def safe_print(text: str) -> None:
    import sys

    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit one srcV6 CTC sample.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--sample-path", default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", default="overfit_srcV6_ctc_one")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-input-target-ratio", type=float, default=1.0)
    parser.add_argument("--text-unit", choices=["char", "char_nodiac", "syllable_nodiac"], default="char")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--encoder-width", type=int, default=16)
    parser.add_argument("--resnet-layers", type=parse_layers, default=(1, 1, 1, 1))
    parser.add_argument("--visual-temporal-layers", type=int, default=1)
    parser.add_argument("--landmark-temporal-layers", type=int, default=1)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--fusion-type", choices=["landmark_first", "concat", "gated", "visual_only", "landmark_only"], default="landmark_first")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--blank-bias-init", type=float, default=-2.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--pass-cer", type=float, default=0.05)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_path:
        sample = Path(args.sample_path)
    else:
        files = sorted(Path(args.data_dir).glob("*.pt"))
        if not files:
            raise RuntimeError(f"No .pt files found under {args.data_dir}")
        sample = files[int(args.sample_index)]
    vocab = build_char_vocab_from_files([sample], min_freq=1, text_unit=args.text_unit)
    dataset = FullClipCTCDataset(
        args.data_dir,
        vocab=vocab,
        files=[sample],
        frame_stride=args.frame_stride,
        min_input_target_ratio=args.min_input_target_ratio,
        text_unit=args.text_unit,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_ctc)
    batch = sanitize_batch(batch_to_device(next(iter(loader)), device))
    model = build_model(args, device, len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    amp_enabled = device.type == "cuda" and args.amp
    safe_print(f"[device] {device}")
    safe_print(f"[sample] {sample}")
    safe_print(
        f"[shape] video={tuple(batch['video'].shape)} target_len={int(batch['target_lengths'][0].item())} "
        f"vocab={len(vocab)} text_unit={args.text_unit}"
    )
    safe_print(f"[ref] {batch['transcript_texts'][0]}")

    history = []
    best = float("inf")
    best_row = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(batch)
        with torch.amp.autocast("cuda", enabled=False):
            loss = ctc_loss(logits, batch, criterion)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite overfit CTC loss: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        should_print = epoch == 1 or epoch % max(1, args.print_every) == 0 or epoch == args.epochs
        if should_print:
            row = eval_one(model, batch, criterion, vocab)
            row["epoch"] = epoch
            history.append(row)
            if row["cer"] < best:
                best = float(row["cer"])
                best_row = row
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": vars(args),
                        "vocab": vocab,
                        "vocab_size": len(vocab),
                        "best": best,
                        "sample": str(sample),
                    },
                    output_dir / "best_model.pth",
                )
            tag = " best" if best_row is row else ""
            safe_print(
                f"[epoch {epoch:04d}] loss={row['loss']:.4f} cer={row['cer']:.4f} "
                f"wer={row['wer']:.4f} best_cer={best:.4f}{tag}"
            )
            safe_print(f"  hyp: {row['hyp'][:180]}")
            write_json(output_dir / "history.json", {"history": history, "config": vars(args)})
            if best <= args.pass_cer:
                break

    final = best_row or eval_one(model, batch, criterion, vocab)
    verdict = "pass" if float(final["cer"]) <= args.pass_cer else "fail"
    safe_print(f"[final] best_cer={best:.4f} pass_cer={args.pass_cer:.4f} verdict={verdict}")
    safe_print(f"[best_hyp] {str(final['hyp'])[:240]}")


if __name__ == "__main__":
    run(parse_args())
