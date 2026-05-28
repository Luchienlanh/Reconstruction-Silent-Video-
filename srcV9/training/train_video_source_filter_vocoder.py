from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from srcV9.data import split_cache_files
from srcV9.models import VideoSourceFilterVocoderModel
from srcV9.training.train_source_filter_vocoder import (
    compute_mel_stats,
    evaluate,
    make_loader,
    mean_baseline,
    train_one_epoch,
)
from srcV9.utils import get_device, seed_everything, unwrap_model, write_json


def parse_layers(value: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("visual layers must look like 1,1,1,1 or 2,2,2,2")
    return tuple(parts)  # type: ignore[return-value]


def build_model(args, device: torch.device, mel_mean: torch.Tensor) -> torch.nn.Module:
    model = VideoSourceFilterVocoderModel(
        num_points=args.num_landmark_points,
        dim=args.dim,
        n_mels=args.n_mels,
        source_bands=args.source_bands,
        visual_width=args.visual_width,
        visual_layers=args.visual_layers,
        visual_temporal_layers=args.visual_temporal_layers,
        landmark_tcn_layers=args.landmark_tcn_layers,
        landmark_transformer_layers=args.landmark_transformer_layers,
        nhead=args.nhead,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        output_bias_init=float(mel_mean.mean().item()),
        source_scale_init=args.source_scale_init,
    ).to(device)
    unwrap_model(model).set_output_bias(mel_mean.to(device))
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.multi_gpu and args.batch_size >= torch.cuda.device_count():
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)
    return model


def make_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    raw = unwrap_model(model)
    return torch.optim.AdamW(
        [
            {"params": raw.visual.parameters(), "lr": args.visual_lr or args.lr * 0.5},
            {"params": raw.landmarks.parameters(), "lr": args.landmark_lr or args.lr},
            {
                "params": list(raw.env_in.parameters())
                + list(raw.src_in.parameters())
                + list(raw.env_decoder.parameters())
                + list(raw.src_decoder.parameters())
                + list(raw.env_norm.parameters())
                + list(raw.src_norm.parameters())
                + list(raw.envelope_head.parameters())
                + list(raw.source_head.parameters())
                + list(raw.source_gate.parameters())
                + [raw.source_scale_logit],
                "lr": args.lr,
            },
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )


def save_checkpoint(path, model, optimizer, epoch, best, args, mel_mean, mel_std):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best": float(best),
            "config": vars(args),
            "mel_mean": mel_mean.detach().cpu(),
            "mel_std": mel_std.detach().cpu(),
            "model_type": "video_source_filter_vocoder",
        },
        path,
    )


def run(args) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit_files = args.limit_files if args.limit_files > 0 else None
    train_files, val_files = split_cache_files(args.data_dir, args.val_ratio, args.seed, limit_files=limit_files)
    train_loader = make_loader(args, train_files, args.batch_size, shuffle=True, random_windows=args.random_windows_per_file)
    stats_loader = make_loader(args, train_files, args.batch_size, shuffle=False, random_windows=0)
    val_loader = make_loader(args, val_files, args.val_batch_size or args.batch_size, shuffle=False, random_windows=0) if val_files else None
    mel_mean, mel_std = compute_mel_stats(stats_loader, device)
    model = build_model(args, device, mel_mean)
    optimizer = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    mean_train = mean_baseline(stats_loader, device, mel_mean, mel_std, args)
    mean_val = mean_baseline(val_loader, device, mel_mean, mel_std, args) if val_loader is not None else None
    print(f"[device] {device}")
    print(f"[data] files train={len(train_files)} val={len(val_files)} windows train={len(train_loader.dataset)}")
    print(
        f"[model] srcV9 video_source_filter dim={args.dim} visual_width={args.visual_width} "
        f"source_bands={args.source_bands} scale_init={args.source_scale_init:.3f}"
    )
    print(f"[baseline] mean_train={mean_train:.6f} mean_val={mean_val if mean_val is not None else 'n/a'}")

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train = train_one_epoch(model, train_loader, optimizer, scaler, device, mel_mean, mel_std, args, epoch)
        train_eval = evaluate(model, stats_loader, device, mel_mean, mel_std, args)
        val_eval = evaluate(model, val_loader, device, mel_mean, mel_std, args) if val_loader is not None else None
        score = float((val_eval or train_eval)["loss"])
        is_best = score < best
        if is_best:
            best = score
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, epoch, best, args, mel_mean, mel_std)
        history.append({"epoch": epoch, "train": train, "train_eval": train_eval, "val_eval": val_eval, "best": best})
        write_json(output_dir / "history.json", {"history": history, "config": vars(args)})
        stats = (val_eval or train_eval)["stats"]
        val_txt = f"{float(val_eval['loss']):.6f}" if val_eval is not None else "n/a"
        print(
            f"[epoch {epoch:04d}] train={train['loss']:.6f} env={train['env']:.4f} src={train['source']:.4f} "
            f"final={train['final']:.4f} train_eval={train_eval['loss']:.6f} val={val_txt} best={best:.6f} "
            f"std_r={stats.get('std_ratio', 0.0):.3f} del_r={stats.get('delta_ratio', 0.0):.3f} "
            f"src_std={stats.get('source_std', 0.0):.3f} scale={stats.get('source_scale', 0.0):.3f} "
            f"gate={stats.get('source_gate', 0.0):.3f}{' best' if is_best else ''}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train video+landmark source-filter vocoder target, synthesize with Griffin-Lim.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_Frontal_v2")
    parser.add_argument("--output-dir", default="checkpoints_srcV9_video_source_filter")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--window-frames", type=int, default=45)
    parser.add_argument("--hop-frames", type=int, default=15)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-windows-per-file", type=int, default=0)
    parser.add_argument("--random-windows-per-file", type=int, default=0)
    parser.add_argument("--smooth-target-frames", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--source-bands", type=int, default=16)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--visual-width", type=int, default=24)
    parser.add_argument("--visual-layers", type=parse_layers, default=(1, 1, 1, 1))
    parser.add_argument("--visual-temporal-layers", type=int, default=1)
    parser.add_argument("--landmark-tcn-layers", type=int, default=6)
    parser.add_argument("--landmark-transformer-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--visual-lr", type=float, default=0.0)
    parser.add_argument("--landmark-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-env", type=float, default=0.7)
    parser.add_argument("--lambda-source", type=float, default=0.8)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--lambda-final-delta", type=float, default=0.35)
    parser.add_argument("--lambda-source-delta", type=float, default=0.15)
    parser.add_argument("--lambda-energy", type=float, default=0.05)
    parser.add_argument("--source-scale-init", type=float, default=0.6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-last", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
