from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV9.data import (
    LandmarkCTCDataset,
    build_vocab_from_files,
    cer,
    collate_landmark_ctc,
    greedy_decode,
    split_cache_files,
    wer,
)
from srcV9.models import LandmarkCTCModel
from srcV9.utils import batch_to_device, get_device, seed_everything, unwrap_model, write_json


def make_loader(args, files, vocab, batch_size: int, shuffle: bool) -> DataLoader:
    ds = LandmarkCTCDataset(
        args.data_dir,
        vocab=vocab,
        files=files,
        frame_stride=args.frame_stride,
        text_unit=args.text_unit,
        min_input_target_ratio=args.min_input_target_ratio,
    )
    if ds.skipped:
        print(f"[data] skipped_ctc_too_short={len(ds.skipped)}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_landmark_ctc,
    )


def build_model(args, vocab_size: int, device: torch.device) -> torch.nn.Module:
    model = LandmarkCTCModel(
        vocab_size=vocab_size,
        num_points=args.num_landmark_points,
        dim=args.dim,
        tcn_layers=args.tcn_layers,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        blank_bias_init=args.blank_bias_init,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu and args.batch_size >= torch.cuda.device_count():
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def ctc_loss(logits: torch.Tensor, batch: dict, criterion: torch.nn.CTCLoss) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1).contiguous()
    input_lengths = batch["landmark_lengths"].to(log_probs.device).long().clamp_max(log_probs.shape[0])
    target_lengths = batch["target_lengths"].to(log_probs.device).long()
    target_2d = batch["target_ids"].to(log_probs.device).long()
    targets = torch.cat([row[: int(length.item())] for row, length in zip(target_2d, target_lengths)], dim=0)
    return criterion(log_probs, targets, input_lengths, target_lengths)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args) -> float:
    model.train()
    amp_enabled = device.type == "cuda" and args.amp
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="train-ctc", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(batch)
        with torch.amp.autocast("cuda", enabled=False):
            loss = ctc_loss(logits, batch, criterion)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite CTC loss at paths={batch.get('paths', [])[:4]}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


@torch.no_grad()
def evaluate(model, loader, criterion, device, vocab: dict[str, int], args) -> dict:
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_count = 0
    samples = []
    for batch in tqdm(loader, desc="eval-ctc", leave=False):
        batch = batch_to_device(batch, device)
        logits = model(batch)
        loss = ctc_loss(logits, batch, criterion)
        pred_ids = logits.float().argmax(dim=-1).detach().cpu()
        for i, ref in enumerate(batch["transcript_texts"]):
            length = int(batch["landmark_lengths"][i].detach().cpu().item())
            hyp = greedy_decode(pred_ids[i, :length].tolist(), vocab)
            c = cer(ref, hyp)
            w = wer(ref, hyp)
            total_cer += c
            total_wer += w
            total_count += 1
            if len(samples) < int(args.print_samples):
                samples.append({"ref": ref, "hyp": hyp, "cer": f"{c:.3f}", "wer": f"{w:.3f}"})
        total_loss += float(loss.detach().cpu())
    return {
        "loss": total_loss / max(1, len(loader)),
        "cer": total_cer / max(1, total_count),
        "wer": total_wer / max(1, total_count),
        "samples": samples,
    }


def safe_print(text: str) -> None:
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(str(text).encode(enc, errors="replace").decode(enc, errors="replace") + "\n")


def save_checkpoint(path, model, optimizer, epoch, best, args, vocab):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best": float(best),
            "config": vars(args),
            "vocab": vocab,
            "vocab_size": len(vocab),
        },
        path,
    )


def run(args) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = split_cache_files(
        args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_files=args.limit_files if args.limit_files > 0 else None,
    )
    vocab = build_vocab_from_files(train_files, text_unit=args.text_unit, min_freq=args.min_token_freq)
    train_loader = make_loader(args, train_files, vocab, args.batch_size, shuffle=True)
    val_loader = make_loader(args, val_files, vocab, args.val_batch_size or args.batch_size, shuffle=False) if val_files else None
    model = build_model(args, len(vocab), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    print(f"[device] {device}")
    print(f"[data] train_files={len(train_files)} val_files={len(val_files)} train_items={len(train_loader.dataset)} vocab={len(vocab)}")
    print(f"[ctc] text_unit={args.text_unit} frame_stride={args.frame_stride} min_ratio={args.min_input_target_ratio}")
    print(f"[model] srcV9 landmark_ctc dim={args.dim} tcn={args.tcn_layers} transformer={args.transformer_layers}")

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args)
        train_eval = evaluate(model, train_loader, criterion, device, vocab, args) if args.eval_train else None
        val_eval = evaluate(model, val_loader, criterion, device, vocab, args) if val_loader is not None else None
        score_eval = val_eval or train_eval
        score = float(score_eval["cer"]) if score_eval is not None else train_loss
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, vocab)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, vocab)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval": train_eval,
            "val_eval": val_eval,
            "best": best,
        }
        history.append(row)
        write_json(output_dir / "history.json", {"history": history, "config": vars(args), "vocab": vocab})
        val_txt = "n/a"
        if val_eval is not None:
            val_txt = f"loss={float(val_eval['loss']):.4f} cer={float(val_eval['cer']):.4f} wer={float(val_eval['wer']):.4f}"
        train_eval_txt = ""
        if train_eval is not None:
            train_eval_txt = f" train_cer={float(train_eval['cer']):.4f} train_wer={float(train_eval['wer']):.4f}"
        safe_print(f"[epoch {epoch:04d}] train_loss={train_loss:.4f}{train_eval_txt} val={val_txt} best={best:.4f}{' best' if is_best else ''}")
        sample_source = val_eval or train_eval
        if sample_source is not None:
            for sample in sample_source["samples"][: args.print_samples]:
                safe_print(f"  ref: {sample['ref'][:160]}")
                safe_print(f"  hyp: {sample['hyp'][:160]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight landmark-only CTC lip-reading model.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--output-dir", default="checkpoints_srcV9_landmark_ctc")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-input-target-ratio", type=float, default=1.05)
    parser.add_argument("--text-unit", choices=["char", "char_nodiac", "syllable_nodiac"], default="syllable_nodiac")
    parser.add_argument("--min-token-freq", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--tcn-layers", type=int, default=6)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--blank-bias-init", type=float, default=-3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-samples", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

