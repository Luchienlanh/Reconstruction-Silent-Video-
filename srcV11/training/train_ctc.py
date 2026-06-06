from __future__ import annotations

import argparse
from contextlib import nullcontext
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
    CHAR_EN_VOCAB,
    AVFeatureTextDataset,
    cer,
    collate_feature_text,
    greedy_decode,
    greedy_decode_with_confidence,
    split_feature_files,
    wer,
)
from srcV11.models import LipTextCTCModel
from srcV11.utils import batch_to_device, get_device, seed_everything, unwrap_model, write_json


def progress_bar(iterable, desc: str):
    return tqdm(
        iterable,
        desc=desc,
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
    )


def cuda_bf16_supported() -> bool:
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(fn is not None and fn())


def cuda_autocast(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        return amp.autocast("cuda", enabled=True, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=True, dtype=dtype)


def make_grad_scaler(enabled: bool):
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def add_boolean_arg(parser: argparse.ArgumentParser, name: str, default: bool, help: str | None = None) -> None:
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help)
        return
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def configure_amp(args: argparse.Namespace, device: torch.device) -> None:
    args.amp_enabled = False
    args.amp_resolved_dtype = "off"
    args.amp_scaler_enabled = False
    if device.type != "cuda" or not args.amp:
        return
    requested = str(args.amp_dtype).lower()
    bf16_supported = cuda_bf16_supported()
    if requested in {"auto", "bf16"} and bf16_supported:
        args.amp_enabled = True
        args.amp_resolved_dtype = "bf16"
        return
    if requested == "bf16":
        print("[amp] BF16 is not supported on this CUDA device; disabling AMP for stability.")
        return
    if requested == "auto":
        print("[amp] BF16 is not supported; disabling AMP. Use --amp-dtype fp16 to force FP16.")
        return
    args.amp_enabled = True
    args.amp_resolved_dtype = "fp16"
    args.amp_scaler_enabled = True
    print("[amp] WARNING: FP16 AMP can produce non-finite losses; BF16 or no AMP is safer.")


def amp_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.bfloat16 if getattr(args, "amp_resolved_dtype", "off") == "bf16" else torch.float16


def make_loader(args: argparse.Namespace, files: list[Path], batch_size: int, shuffle: bool) -> DataLoader:
    ds = AVFeatureTextDataset(
        args.feature_dir,
        files=files,
        vocab=CHAR_EN_VOCAB,
        min_input_target_ratio=args.min_input_target_ratio,
        input_length_factor=args.upsample_factor,
    )
    if ds.skipped:
        print(f"[data] skipped={len(ds.skipped)} first={ds.skipped[:3]}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_feature_text,
    )


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


def train_val_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    if args.train_manifest:
        train_files = read_manifest(args.train_manifest)
        if args.val_manifest:
            val_files = read_manifest(args.val_manifest)
        else:
            _unused_train, val_files = split_feature_files(
                args.feature_dir,
                val_ratio=args.val_ratio,
                seed=args.seed,
                limit_files=args.limit_files if args.limit_files > 0 else None,
            )
        return train_files, val_files

    return split_feature_files(
        args.feature_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_files=args.limit_files if args.limit_files > 0 else None,
    )


def ctc_loss(logits: torch.Tensor, batch: dict, model: LipTextCTCModel, criterion: torch.nn.CTCLoss) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1).contiguous()
    input_lengths = model.output_lengths(batch["feature_lengths"].to(log_probs.device)).clamp_max(log_probs.shape[0])
    target_lengths = batch["target_lengths"].to(log_probs.device).long()
    target_2d = batch["target_ids"].to(log_probs.device).long()
    targets = torch.cat([row[: int(length.item())] for row, length in zip(target_2d, target_lengths)], dim=0)
    return criterion(log_probs, targets, input_lengths, target_lengths)


def logits_are_finite(logits: torch.Tensor) -> bool:
    return bool(torch.isfinite(logits).all())


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args) -> float:
    model.train()
    raw = unwrap_model(model)
    amp_enabled = bool(getattr(args, "amp_enabled", False))
    autocast_dtype = amp_torch_dtype(args)
    total = 0.0
    count = 0

    def forward_loss(batch: dict, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor, bool]:
        with cuda_autocast(use_amp, autocast_dtype):
            logits = model(batch["features"], batch["feature_mask"])
        loss = ctc_loss(logits, batch, raw, criterion)
        return logits, loss, logits_are_finite(logits)

    for batch in progress_bar(loader, "train-ctc"):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss, finite_logits = forward_loss(batch, amp_enabled)
        if amp_enabled and (not finite_logits or not torch.isfinite(loss)):
            print(
                f"[amp] non-finite logits/loss with dtype={getattr(args, 'amp_resolved_dtype', 'off')}; "
                "retrying this batch in FP32 and disabling AMP."
            )
            args.amp_enabled = False
            args.amp_resolved_dtype = "off"
            args.amp_scaler_enabled = False
            amp_enabled = False
            if hasattr(scaler, "_enabled"):
                scaler._enabled = False
            optimizer.zero_grad(set_to_none=True)
            logits, loss, finite_logits = forward_loss(batch, False)
        if (not finite_logits) or (not torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite CTC loss loss={float(loss.detach().nan_to_num().cpu())} "
                f"logits_finite={finite_logits} amp={getattr(args, 'amp_resolved_dtype', 'off')} "
                f"paths={batch.get('paths', [])[:4]}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if not bool(torch.isfinite(grad_norm)):
                print(f"[warn] non-finite grad_norm at paths={batch.get('paths', [])[:4]}; skipping step.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


@torch.no_grad()
def evaluate(model, loader, criterion, device, args) -> dict:
    model.eval()
    raw = unwrap_model(model)
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_conf = 0.0
    total_count = 0
    samples = []
    for batch in progress_bar(loader, "eval-ctc"):
        batch = batch_to_device(batch, device)
        logits = model(batch["features"], batch["feature_mask"])
        loss = ctc_loss(logits, batch, raw, criterion)
        probs = torch.softmax(logits.float(), dim=-1)
        max_probs, pred_ids = probs.max(dim=-1)
        pred_ids = pred_ids.detach().cpu()
        max_probs = max_probs.detach().cpu()
        out_lengths = raw.output_lengths(batch["feature_lengths"]).detach().cpu()
        for i, ref in enumerate(batch["transcript_texts"]):
            length = int(min(out_lengths[i].item(), pred_ids.shape[1]))
            hyp, conf = greedy_decode_with_confidence(
                pred_ids[i, :length].tolist(),
                max_probs[i, :length].tolist(),
                CHAR_EN_VOCAB,
            )
            c = cer(ref, hyp)
            w = wer(ref, hyp)
            total_cer += c
            total_wer += w
            total_conf += conf
            total_count += 1
            if len(samples) < int(args.print_samples):
                samples.append({"ref": ref, "hyp": hyp, "cer": c, "wer": w, "confidence": conf})
        total_loss += float(loss.detach().cpu())
    return {
        "loss": total_loss / max(1, len(loader)),
        "cer": total_cer / max(1, total_count),
        "wer": total_wer / max(1, total_count),
        "confidence": total_conf / max(1, total_count),
        "samples": samples,
    }


def save_checkpoint(path, model, optimizer, epoch, best, args, input_dim: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["input_dim"] = int(input_dim)
    config["vocab_size"] = len(CHAR_EN_VOCAB)
    config["text_unit"] = "char_en"
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "best": float(best),
            "config": config,
            "vocab": CHAR_EN_VOCAB,
            "model_type": "srcV11_lip_text_ctc",
        },
        path,
    )


def safe_print(text: str) -> None:
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(str(text).encode(enc, errors="replace").decode(enc, errors="replace") + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train srcV11 lip-to-text CTC on cached AV-HuBERT features.")
    parser.add_argument("--feature-dir", "--data-dir", dest="feature_dir", default="Processed_Data_AVHubertFeatures_LRS2_10k")
    parser.add_argument("--output-dir", default="checkpoints_srcV11_lrs2_char_ctc")
    parser.add_argument("--train-manifest", default="", help="Optional text file with one training feature path per line.")
    parser.add_argument("--val-manifest", default="", help="Optional text file with one validation feature path per line.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--min-input-target-ratio", type=float, default=1.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--tcn-layers", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--upsample-factor", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--blank-bias-init", type=float, default=-3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    add_boolean_arg(parser, "--multi-gpu", default=True)
    add_boolean_arg(parser, "--eval-train", default=True)
    parser.add_argument("--print-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    configure_amp(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = train_val_files(args)
    train_loader = make_loader(args, train_files, args.batch_size, shuffle=True)
    val_loader = (
        make_loader(args, val_files, args.val_batch_size or args.batch_size, shuffle=False)
        if val_files
        else None
    )
    first = train_loader.dataset[0]
    input_dim = int(first["features"].shape[-1])
    model = LipTextCTCModel(
        input_dim=input_dim,
        vocab_size=len(CHAR_EN_VOCAB),
        dim=args.dim,
        tcn_layers=args.tcn_layers,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        upsample_factor=args.upsample_factor,
        blank_bias_init=args.blank_bias_init,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu and args.batch_size >= torch.cuda.device_count():
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = make_grad_scaler(bool(getattr(args, "amp_scaler_enabled", False)))

    print(f"[device] {device}")
    print(f"[amp] enabled={args.amp_enabled} dtype={args.amp_resolved_dtype} scaler={args.amp_scaler_enabled}")
    print(
        f"[data] train_files={len(train_files)} val_files={len(val_files)} "
        f"train_items={len(train_loader.dataset)} vocab={len(CHAR_EN_VOCAB)}"
    )
    print(
        f"[model] srcV11 lip-text-ctc input_dim={input_dim} dim={args.dim} "
        f"tcn={args.tcn_layers} transformer={args.transformer_layers} upsample={args.upsample_factor}"
    )

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args)
        train_eval = evaluate(model, train_loader, criterion, device, args) if args.eval_train else None
        val_eval = evaluate(model, val_loader, criterion, device, args) if val_loader is not None else None
        score_eval = val_eval or train_eval
        score = float(score_eval["cer"]) if score_eval is not None else train_loss
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, input_dim)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, input_dim)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval": train_eval,
            "val_eval": val_eval,
            "best": best,
        }
        history.append(row)
        write_json(output_dir / "history.json", {"history": history, "config": vars(args), "vocab": CHAR_EN_VOCAB})
        (output_dir / "vocab.json").write_text(json.dumps(CHAR_EN_VOCAB, ensure_ascii=False, indent=2), encoding="utf-8")
        val_txt = "n/a"
        if val_eval is not None:
            val_txt = (
                f"loss={float(val_eval['loss']):.4f} cer={float(val_eval['cer']):.4f} "
                f"wer={float(val_eval['wer']):.4f} conf={float(val_eval['confidence']):.3f}"
            )
        train_txt = ""
        if train_eval is not None:
            train_txt = f" train_cer={float(train_eval['cer']):.4f} train_wer={float(train_eval['wer']):.4f}"
        safe_print(f"[epoch {epoch:04d}] train_loss={train_loss:.4f}{train_txt} val={val_txt} best={best:.4f}{' best' if is_best else ''}")
        sample_source = val_eval or train_eval
        if sample_source is not None:
            for sample in sample_source["samples"][: args.print_samples]:
                safe_print(f"  ref: {sample['ref'][:160]}")
                safe_print(f"  hyp: {sample['hyp'][:160]}")
                safe_print(f"  cer={sample['cer']:.3f} wer={sample['wer']:.3f} conf={sample['confidence']:.3f}")


if __name__ == "__main__":
    run(parse_args())
