from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV6.data import (
    FullClipCTCDataset,
    build_char_vocab_from_files,
    cer,
    collate_ctc,
    greedy_decode,
    split_cache_files,
    wer,
)
from srcV6.models import LipreadingCTCModel
from srcV6.utils import batch_to_device, get_device, seed_everything, unwrap_model, write_json


def safe_print(text: str) -> None:
    import sys

    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def parse_layers(value: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("resnet layers must look like 1,1,1,1 or 2,2,2,2")
    return tuple(parts)  # type: ignore[return-value]


def make_loader(
    data_dir: str | Path,
    files: list[Path],
    vocab: dict[str, int],
    batch_size: int,
    frame_stride: int,
    min_input_target_ratio: float,
    text_unit: str,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    ds = FullClipCTCDataset(
        data_dir,
        vocab=vocab,
        files=files,
        frame_stride=frame_stride,
        min_input_target_ratio=min_input_target_ratio,
        text_unit=text_unit,
    )
    if ds.skipped:
        print(f"[data] skipped_ctc_too_short={len(ds.skipped)}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_ctc,
    )


def build_model(args: argparse.Namespace, device: torch.device, vocab_size: int) -> torch.nn.Module:
    model = LipreadingCTCModel(
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
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu and args.batch_size >= torch.cuda.device_count():
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    raw = unwrap_model(model)
    return torch.optim.AdamW(
        [
            {"params": raw.visual.parameters(), "lr": args.visual_lr or args.lr * 0.5},
            {"params": raw.landmarks.parameters(), "lr": args.landmark_lr or args.lr},
            {"params": raw.fusion.parameters(), "lr": args.fusion_lr or args.lr},
            {"params": raw.ctc_head.parameters(), "lr": args.ctc_lr or args.lr},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )


def sanitize_batch(batch: dict) -> dict:
    for key in ("video", "landmarks", "video_times"):
        val = batch.get(key)
        if torch.is_tensor(val) and not torch.isfinite(val).all():
            batch[key] = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return batch


def ctc_loss(logits: torch.Tensor, batch: dict, criterion: torch.nn.CTCLoss) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1).contiguous()
    input_lengths = batch["video_lengths"].to(log_probs.device).long().clamp_max(log_probs.shape[0])
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
        batch = sanitize_batch(batch_to_device(batch, device))
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
def evaluate(model, loader, criterion, device, vocab: dict[str, int], args) -> dict[str, float | list[dict[str, str]]]:
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_count = 0
    samples = []
    for batch in tqdm(loader, desc="eval-ctc", leave=False):
        batch = sanitize_batch(batch_to_device(batch, device))
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
            if len(samples) < int(args.print_samples):
                samples.append({"ref": ref, "hyp": hyp, "cer": f"{c:.3f}", "wer": f"{w:.3f}"})
        total_loss += float(loss.detach().cpu())
    batches = max(1, len(loader))
    return {
        "loss": total_loss / batches,
        "cer": total_cer / max(1, total_count),
        "wer": total_wer / max(1, total_count),
        "samples": samples,
    }


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train srcV6 video-to-text char CTC model.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--output-dir", default="checkpoints_srcV6_ctc")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-input-target-ratio", type=float, default=1.05)
    parser.add_argument("--text-unit", choices=["char", "char_nodiac", "syllable_nodiac"], default="char")
    parser.add_argument("--min-token-freq", type=int, default=1)
    parser.add_argument("--min-char-freq", type=int, default=0, help="Deprecated alias; use --min-token-freq.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--encoder-width", type=int, default=24)
    parser.add_argument("--resnet-layers", type=parse_layers, default=(1, 1, 1, 1))
    parser.add_argument("--visual-temporal-layers", type=int, default=1)
    parser.add_argument("--landmark-temporal-layers", type=int, default=1)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--fusion-type", choices=["landmark_first", "concat", "gated", "visual_only", "landmark_only"], default="landmark_first")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--blank-bias-init", type=float, default=-2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--visual-lr", type=float, default=0.0)
    parser.add_argument("--landmark-lr", type=float, default=0.0)
    parser.add_argument("--fusion-lr", type=float, default=0.0)
    parser.add_argument("--ctc-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-samples", type=int, default=2)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, limit_files=limit_files)
    min_freq = args.min_token_freq if args.min_token_freq > 0 else max(1, args.min_char_freq)
    vocab = build_char_vocab_from_files(train_files, min_freq=min_freq, text_unit=args.text_unit)
    train_loader = make_loader(
        args.data_dir,
        train_files,
        vocab,
        args.batch_size,
        args.frame_stride,
        args.min_input_target_ratio,
        args.text_unit,
        args.num_workers,
        shuffle=True,
    )
    val_loader = None
    if val_files:
        val_loader = make_loader(
            args.data_dir,
            val_files,
            vocab,
            args.val_batch_size or args.batch_size,
            args.frame_stride,
            args.min_input_target_ratio,
            args.text_unit,
            args.num_workers,
            shuffle=False,
        )
    model = build_model(args, device, len(vocab))
    optimizer = make_optimizer(model, args)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    print(f"[device] {device}")
    print(f"[data] train_files={len(train_files)} val_files={len(val_files)} train_items={len(train_loader.dataset)} vocab={len(vocab)}")
    print(f"[ctc] text_unit={args.text_unit} frame_stride={args.frame_stride} min_input_target_ratio={args.min_input_target_ratio}")
    print(f"[model] srcV6 dim={args.dim} width={args.encoder_width} fusion={args.fusion_type}")

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args)
        train_eval = evaluate(model, train_loader, criterion, device, vocab, args)
        val_eval = evaluate(model, val_loader, criterion, device, vocab, args) if val_loader is not None else None
        score = float(val_eval["cer"] if val_eval is not None else train_eval["cer"])
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
            "best_cer": best,
        }
        history.append(row)
        write_json(output_dir / "history.json", {"history": history, "config": vars(args), "vocab": vocab})
        val_txt = "n/a"
        if val_eval is not None:
            val_txt = f"loss={float(val_eval['loss']):.4f} cer={float(val_eval['cer']):.4f} wer={float(val_eval['wer']):.4f}"
        safe_print(
            f"[epoch {epoch:04d}] train_loss={train_loss:.4f} "
            f"train_cer={float(train_eval['cer']):.4f} train_wer={float(train_eval['wer']):.4f} "
            f"val={val_txt} best_cer={best:.4f}{' best' if is_best else ''}"
        )
        for sample in (val_eval or train_eval)["samples"][: args.print_samples]:
            safe_print(f"  ref: {sample['ref'][:140]}")
            safe_print(f"  hyp: {sample['hyp'][:140]}")


if __name__ == "__main__":
    run(parse_args())
