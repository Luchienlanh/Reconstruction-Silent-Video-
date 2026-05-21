"""
main.py
=======
Entry point for the modular silent-video-to-mel reconstruction pipeline.
Supports full CLI arguments for easy configuration, direct end-to-end training,
and seamless switching between encoder and decoder types.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from tqdm.auto import tqdm

# Ensure parent and src directories are in sys.path to resolve modular imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import modular components
from data.dataset import VNLipDatasetV2, collate_pad_v2
from models.encoders.factory import build_encoder, VisualLandmarkEncoderV2
from models.decoders.siren import TFiLMSIRENDecoder
from models.decoders.wire import TFiLMWIREDecoder
from models.decoders.finer import TFiLMFINERDecoder
from models.decoders.dual import DualDecoder, DualWrapDecoder
from models.decoders.wrap import TFiLMWrapFISINDecoder, TFiLMWrapFIWIDecoder
from models.decoders.upsample import MelTemporalUpsampleDecoder
from models.loss import MelReconstructionLoss


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def default_data_dir() -> str:
    full_frame = PROJECT_ROOT / "FullFrame_test"
    if full_frame.is_dir():
        return "Processed_Data_Mel_HiFiGAN_FullFrame"
    return "Processed_Data_Mel_HiFiGAN"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_snn_if_needed(module: torch.nn.Module, encoder_type: str) -> None:
    if encoder_type != "snn":
        return
    from spikingjelly.activation_based import functional
    functional.reset_net(module)


def build_base_decoder(decoder_type: str):
    decoder_type = decoder_type.lower()
    common = dict(hidden_dim=256, out_dim=80, num_layers=4, use_conv=True)
    if decoder_type == "siren":
        return TFiLMSIRENDecoder(**common, output_activation=None)
    if decoder_type == "wire":
        return TFiLMWIREDecoder(**common)
    if decoder_type == "finer":
        return TFiLMFINERDecoder(**common)
    if decoder_type == "dual":
        return DualDecoder(**common)
    if decoder_type in {"dual_wrap", "dualwrap"}:
        return DualWrapDecoder(**common)
    if decoder_type in {"wrap_siren", "wrap_fisin", "wrap"}:
        return TFiLMWrapFISINDecoder(**common)
    if decoder_type in {"wrap_wire", "wrap_fiwi"}:
        return TFiLMWrapFIWIDecoder(**common)
    raise ValueError(f"Unknown decoder_type: {decoder_type}")


def build_models(device: torch.device, encoder_type: str, decoder_type: str, num_landmark_points: int):
    visual_encoder = build_encoder(encoder_type).to(device)
    encoder = VisualLandmarkEncoderV2(
        visual_encoder,
        num_landmark_points=num_landmark_points,
        z_dim=512,
    ).to(device)

    base_decoder = build_base_decoder(decoder_type).to(device)
    decoder = MelTemporalUpsampleDecoder(
        base_decoder,
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    return encoder, decoder


def make_criterion(args: argparse.Namespace, device: torch.device):
    return MelReconstructionLoss(
        lambda_mel=1.0,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
    ).to(device)


def unpack_batch(batch, device: torch.device):
    if len(batch) == 5:
        video, landmarks, target, lengths, paths = batch
    elif len(batch) == 4 and torch.is_tensor(batch[1]) and batch[1].dim() == 4:
        video, landmarks, target, lengths = batch
        paths = None
    else:
        raise ValueError("Dataset batch shape is not recognized. Ensure VNLipDatasetV2 returns correct outputs.")

    return (
        video.to(device, non_blocking=True),
        landmarks.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        lengths.to(device, non_blocking=True),
        paths,
    )


def save_mel_plot(pred_mel: torch.Tensor, target_mel: torch.Tensor, output_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = pred_mel.detach().cpu()[0].float().transpose(0, 1).numpy()
    target = target_mel.detach().cpu()[0].float().transpose(0, 1).numpy()
    diff = pred - target
    vmin = min(float(pred.min()), float(target.min()))
    vmax = max(float(pred.max()), float(target.max()))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    gt_img = axes[0].imshow(target, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    axes[1].imshow(pred, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    diff_img = axes[2].imshow(diff, origin="lower", aspect="auto", interpolation="nearest", cmap="coolwarm")

    fig.suptitle(title)
    axes[0].set_title("Ground truth mel")
    axes[1].set_title("Predicted mel")
    axes[2].set_title("Prediction - ground truth")
    axes[2].set_xlabel("Mel frame")
    for ax in axes:
        ax.set_ylabel("Mel bin")

    fig.colorbar(gt_img, ax=axes[:2], fraction=0.02, pad=0.02)
    fig.colorbar(diff_img, ax=axes[2], fraction=0.02, pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": vars(args),
        },
        path,
    )


def load_resume(
    resume_path: Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
    return start_epoch, best_val


def train_one_epoch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    encoder.train()
    decoder.train()
    amp_enabled = device.type == "cuda" and args.amp
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        video, landmarks, target, lengths, _ = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        reset_snn_if_needed(encoder, args.encoder_type)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            z = encoder(video, landmarks)
            pred = decoder(z, target_len=target.shape[1])

        with torch.amp.autocast("cuda", enabled=False):
            loss = criterion(pred.float(), target.float(), lengths)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite train loss: {float(loss.detach().cpu())}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        reset_snn_if_needed(encoder, args.encoder_type)

        total_loss += float(loss.detach().cpu())
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("No training batches were processed.")
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    args: argparse.Namespace,
    plot_path: Optional[Path] = None,
    epoch: Optional[int] = None,
) -> float:
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    num_batches = 0
    plotted = False

    for batch in tqdm(loader, desc="val", leave=False):
        video, landmarks, target, lengths, paths = unpack_batch(batch, device)
        reset_snn_if_needed(encoder, args.encoder_type)
        z = encoder(video, landmarks)
        pred = decoder(z, target_len=target.shape[1])
        reset_snn_if_needed(encoder, args.encoder_type)

        loss = criterion(pred.float(), target.float(), lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite val loss: {float(loss.detach().cpu())}")

        total_loss += float(loss.detach().cpu())
        num_batches += 1

        if plot_path is not None and not plotted:
            sample_name = Path(paths[0]).name if paths else "validation sample"
            title = f"Epoch {epoch} | {sample_name}"
            save_mel_plot(pred, target, plot_path, title)
            plotted = True

    if num_batches == 0:
        raise RuntimeError("No validation batches were processed.")
    return total_loss / num_batches


def create_loaders(args: argparse.Namespace):
    data_dir = resolve_path(args.data_dir)
    dataset_output_dir = resolve_path(args.dataset_output_dir)
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")

    dataset = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=True,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(dataset_output_dir or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
        force_full_frame=args.force_full_frame,
    )

    split_dataset = dataset
    if args.limit is not None:
        limit = max(1, min(int(args.limit), len(dataset)))
        split_dataset = Subset(dataset, list(range(limit)))

    total = len(split_dataset)
    val_count = max(1, int(round(total * args.val_ratio))) if total > 1 else 0
    train_count = total - val_count
    if train_count <= 0:
        raise RuntimeError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    if val_count > 0:
        train_set, val_set = random_split(split_dataset, [train_count, val_count], generator=generator)
    else:
        train_set, val_set = split_dataset, None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad_v2,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.val_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_pad_v2,
        )

    return dataset, train_loader, val_loader, train_count, val_count


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "checkpoints_modular"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, train_loader, val_loader, train_count, val_count = create_loaders(args)
    encoder, decoder = build_models(device, args.encoder_type, args.decoder_type, dataset.landmark_num_points)
    criterion = make_criterion(args, device)

    # Implement parameter groups learning rate split from the optimization plan:
    # Scale learning rate up for discrete SNN gradients, scale down for sensitive FINER decoders.
    is_snn = args.encoder_type.lower() == "snn"
    is_finer = args.decoder_type.lower() == "finer"

    encoder_lr = args.lr * 2.5 if is_snn else args.lr
    decoder_lr = args.lr * 0.5 if is_finer else args.lr

    optimizer_groups = [
        {"params": encoder.parameters(), "lr": encoder_lr},
        {"params": decoder.parameters(), "lr": decoder_lr}
    ]

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    start_epoch = 1
    best_val_loss = float("inf")
    resume_path = resolve_path(args.resume)
    if resume_path is not None:
        start_epoch, best_val_loss = load_resume(resume_path, encoder, decoder, optimizer, scaler, device)
        print(f"[resume] {safe_text(resume_path)} -> start_epoch={start_epoch}")

    print(f"[device] {device}")
    print(f"[data] {safe_text(resolve_path(args.data_dir))}")
    print(f"[split] train={train_count} val={val_count}")
    print(f"[model] encoder={args.encoder_type} decoder={args.decoder_type} landmarks={dataset.landmark_num_points}")
    print(f"[lr] encoder_lr={encoder_lr:.2e} decoder_lr={decoder_lr:.2e}")
    print(
        "[loss] mel=1.0 delta={} delta2={} energy={}".format(
            args.lambda_delta,
            args.lambda_delta2,
            args.lambda_energy,
        )
    )

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            args=args,
        )

        val_loss = None
        if val_loader is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            plot_path = None
            if epoch % args.plot_every == 0 or epoch == args.epochs:
                plot_path = output_dir / "plots" / f"val_pred_vs_gt_epoch_{epoch:04d}.png"
            val_loss = evaluate(
                encoder=encoder,
                decoder=decoder,
                loader=val_loader,
                criterion=criterion,
                device=device,
                args=args,
                plot_path=plot_path,
                epoch=epoch,
            )

        score = val_loss if val_loss is not None else train_loss
        is_best = score < best_val_loss
        if is_best:
            best_val_loss = score
            save_checkpoint(
                output_dir / "best_model.pth",
                encoder,
                decoder,
                optimizer,
                scaler,
                epoch,
                train_loss,
                val_loss,
                best_val_loss,
                args,
            )

        save_checkpoint(
            output_dir / "last_model.pth",
            encoder,
            decoder,
            optimizer,
            scaler,
            epoch,
            train_loss,
            val_loss,
            best_val_loss,
            args,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pth",
                encoder,
                decoder,
                optimizer,
                scaler,
                epoch,
                train_loss,
                val_loss,
                best_val_loss,
                args,
            )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": best_val_loss}
        history.append(row)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"history": history, "config": vars(args)}, f, indent=2)

        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        mark = " best" if is_best else ""
        print(f"[epoch {epoch:04d}] train={train_loss:.6f} val={val_text} best={best_val_loss:.6f}{mark}")

    print(f"[done] output={safe_text(output_dir)}")
    print(f"[best] {safe_text(output_dir / 'best_model.pth')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full mel reconstruction in modular structure.")
    parser.add_argument("--data-dir", default=default_data_dir())
    parser.add_argument("--dataset-output-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default="checkpoints_modular")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "nonsnn", "cnn_transformer", "snn"])
    parser.add_argument(
        "--decoder-type",
        default="siren",
        choices=["siren", "wire", "finer", "dual", "dual_wrap", "wrap_siren", "wrap_fisin", "wrap", "wrap_wire", "wrap_fiwi"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--limit", type=int, default=None, help="Use first N samples for dry runs.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.0)
    parser.add_argument("--lambda-delta2", type=float, default=0.0)
    parser.add_argument("--lambda-energy", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-full-frame", action="store_true")
    parser.add_argument("--disable-fallback", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
