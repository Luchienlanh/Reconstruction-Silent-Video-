from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from srcV7.data import R2CacheDataset, collate_r2cache
from srcV7.models import MaskedMelLoss
from srcV7.training.common import (
    build_model,
    compute_mel_stats,
    init_decoder_output_bias,
    masked_stats,
    model_inputs,
    save_checkpoint,
    unwrap_model,
)
from srcV7.utils.common import batch_to_device, get_device, seed_everything
from srcV7.utils.plotting import save_mel_comparison


def run(args) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = R2CacheDataset(args.data_dir, max_frames=args.max_frames, random_crop=False, seed=args.seed, limit=args.limit)
    sample_idx = max(0, min(args.sample_index, len(ds) - 1))
    one = Subset(ds, [sample_idx])
    loader = DataLoader(one, batch_size=1, shuffle=False, collate_fn=collate_r2cache)
    batch_cpu = next(iter(loader))
    batch = batch_to_device(batch_cpu, device)

    stats_loader = DataLoader(one, batch_size=1, shuffle=False, collate_fn=collate_r2cache)
    mel_mean, mel_std = compute_mel_stats(stats_loader, device)
    criterion = MaskedMelLoss(
        mel_mean,
        mel_std,
        lambda_mel=args.lambda_mel,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
        shift_window=args.shift_window,
    ).to(device)
    model = build_model(device, args)
    init_decoder_output_bias(model, mel_mean)
    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.encoder.parameters(), "lr": args.lr * args.encoder_lr_scale},
            {"params": raw_model.decoder.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    print(f"[device] {device}")
    print(f"[sample] {batch_cpu['paths'][0]}")
    print(f"[shape] video={tuple(batch['video'].shape)} landmarks={tuple(batch['landmarks'].shape)} mel={tuple(batch['mel'].shape)}")
    print(
        f"[model] r2plus1d_{args.decoder_type} dim={args.dim} spatial_tokens={args.spatial_tokens} "
        f"decoder_channels={args.decoder_channels or args.dim} layers={args.decoder_layers}"
    )

    best = float("inf")
    history = []
    amp_enabled = device.type == "cuda" and args.amp
    for epoch in range(1, args.epochs + 1):
        freeze_encoder = epoch <= args.freeze_encoder_epochs
        for param in raw_model.encoder.parameters():
            param.requires_grad = not freeze_encoder
        for _ in range(args.steps_per_epoch):
            model.train()
            if freeze_encoder:
                raw_model.encoder.eval()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(model_inputs(batch))
            with torch.amp.autocast("cuda", enabled=False):
                loss = criterion(pred.float(), batch["mel"].float(), batch["mel_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite overfit loss: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        with torch.no_grad():
            pred_eval = model(model_inputs(batch))
            eval_loss = criterion(pred_eval.float(), batch["mel"].float(), batch["mel_mask"])
            stats = masked_stats(pred_eval, batch["mel"], batch["mel_mask"])
        best = min(best, float(eval_loss.detach().cpu()))
        row = {"epoch": epoch, "loss": float(eval_loss.detach().cpu()), "best": best, **stats}
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:04d}] loss={row['loss']:.6f} best={best:.6f} "
                f"std_r={row.get('std_ratio', 0):.3f} del_r={row.get('delta_ratio', 0):.3f} "
                f"eng_r={row.get('energy_ratio', 0):.3f} {'encoder=frozen' if freeze_encoder else 'encoder=train'}"
            )
        if epoch % args.plot_every == 0 or epoch == args.epochs:
            save_mel_comparison(pred_eval, batch["mel"], output_dir / f"mel_epoch_{epoch:04d}.png", title=f"overfit epoch {epoch}")

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "config": vars(args)}, f, ensure_ascii=False, indent=2)
    save_checkpoint(output_dir / "last_model.pth", model, optimizer, args.epochs, best, args, mel_mean, mel_std)
    verdict = "pass" if best <= args.pass_loss else "fail"
    print(f"[final] best={best:.6f} pass_loss={args.pass_loss:.6f} verdict={verdict}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fast overfit one srcV7 cache sample.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR")
    parser.add_argument("--output-dir", default="overfit_srcV7_cnn_plain")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--multi-gpu", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.05)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=100.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-tokens", type=int, default=4)
    parser.add_argument("--upsample-mode", default="conv_transpose", choices=["linear", "conv_transpose"])
    parser.add_argument("--decoder-type", default="cnn_plain", choices=["cnn_plain", "cnn_film"])
    parser.add_argument("--decoder-channels", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--decoder-kernel-size", type=int, default=5)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-mel", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.25)
    parser.add_argument("--lambda-delta2", type=float, default=0.05)
    parser.add_argument("--lambda-energy", type=float, default=0.05)
    parser.add_argument("--shift-window", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--pass-loss", type=float, default=0.06)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
