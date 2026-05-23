"""
main.py
=======
Entry point for the modular silent-video-to-mel reconstruction pipeline.
Supports full CLI arguments for easy configuration, direct end-to-end training,
seamless switching between encoder and decoder types, multi-GPU training (nn.DataParallel),
and gradient accumulation to prevent VRAM OOM errors.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import argparse
import random
import math
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
from models.encoders.factory import build_encoder, VisualLandmarkEncoder, VisualLandmarkEncoderV2, VisualLandmarkEncoderGatedResidual
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
    try:
        from spikingjelly.activation_based import functional
        functional.reset_net(module)
    except ImportError:
        pass


def get_curriculum_frames(epoch: int, total_epochs: int, target_max_frames: int) -> int:
    """
    Calculate the max_frames for curriculum learning based on current epoch.
    Instead of a step function, uses a smooth linear progression from 40% to 100%
    over the first 75% of epochs, and holds at 100% for the remaining 25% of epochs.
    Ensures active max_frames increases smoothly almost every epoch.
    """
    if epoch <= 0:
        return target_max_frames
    
    ramp_end_pct = 0.75
    pct = epoch / total_epochs
    
    if pct >= ramp_end_pct:
        factor = 1.0
    else:
        start_factor = 0.40
        end_factor = 1.00
        factor = start_factor + (end_factor - start_factor) * (pct / ramp_end_pct)
        
    frames = int(round(target_max_frames * factor))
    min_allowed = min(15, target_max_frames)
    return max(min_allowed, frames)


def set_dataset_max_frames(dataset, max_frames: Optional[int]) -> None:
    """Recursively unwrap Subsets to find the actual dataset and set max_frames."""
    if hasattr(dataset, "dataset"):
        set_dataset_max_frames(dataset.dataset, max_frames)
    elif hasattr(dataset, "max_frames"):
        dataset.max_frames = max_frames


def build_base_decoder(decoder_type: str, target_type: str = "mel_hifigan"):
    decoder_type = decoder_type.lower()
    common = dict(hidden_dim=256, out_dim=80, num_layers=4, use_conv=True)
    output_act = None if target_type == "mel_hifigan" else "tanh"
    if decoder_type == "siren":
        return TFiLMSIRENDecoder(**common, output_activation=None)
    if decoder_type == "wire":
        return TFiLMWIREDecoder(**common, output_activation=output_act)
    if decoder_type == "finer":
        return TFiLMFINERDecoder(**common, output_activation=output_act)
    if decoder_type == "dual":
        return DualDecoder(**common, output_activation=output_act)
    if decoder_type in {"dual_wrap", "dualwrap"}:
        return DualWrapDecoder(**common, output_activation=output_act)
    if decoder_type in {"wrap_siren", "wrap_fisin", "wrap"}:
        return TFiLMWrapFISINDecoder(**common, output_activation=output_act)
    if decoder_type in {"wrap_wire", "wrap_fiwi"}:
        return TFiLMWrapFIWIDecoder(**common, output_activation=output_act)
    raise ValueError(f"Unknown decoder_type: {decoder_type}")


def build_models(device: torch.device, encoder_type: str, decoder_type: str, num_landmark_points: int, fusion_type: str = "cross_attn", target_type: str = "mel_hifigan"):
    visual_encoder = build_encoder(encoder_type).to(device)
    if fusion_type == "concat":
        encoder = VisualLandmarkEncoder(
            visual_encoder,
            num_landmark_points=num_landmark_points,
            z_dim=512,
        ).to(device)
    elif fusion_type == "gated_residual":
        encoder = VisualLandmarkEncoderGatedResidual(
            visual_encoder,
            num_landmark_points=num_landmark_points,
            z_dim=512,
        ).to(device)
    else:
        encoder = VisualLandmarkEncoderV2(
            visual_encoder,
            num_landmark_points=num_landmark_points,
            z_dim=512,
        ).to(device)

    base_decoder = build_base_decoder(decoder_type, target_type=target_type).to(device)
    decoder = MelTemporalUpsampleDecoder(
        base_decoder,
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    return encoder, decoder


def load_compatible_state_dict(module: torch.nn.Module, state_dict: dict, tag: str):
    target_state = module.state_dict()
    compatible = {}
    skipped = []
    unexpected = []

    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        if clean_key not in target_state:
            unexpected.append(clean_key)
            continue
        if hasattr(value, "shape") and target_state[clean_key].shape != value.shape:
            skipped.append((clean_key, tuple(value.shape), tuple(target_state[clean_key].shape)))
            continue
        compatible[clean_key] = value

    missing_keys, unexpected_keys = module.load_state_dict(compatible, strict=False)
    print(
        f"[pretrained-ssl] {tag}: loaded={len(compatible)} "
        f"missing={len(missing_keys)} unexpected={len(unexpected) + len(unexpected_keys)} skipped_shape={len(skipped)}"
    )
    if missing_keys:
        print(f"[pretrained-ssl] {tag} missing (first 8): {missing_keys[:8]}")
    combined_unexpected = unexpected + list(unexpected_keys)
    if combined_unexpected:
        print(f"[pretrained-ssl] {tag} unexpected (first 8): {combined_unexpected[:8]}")
    if skipped:
        formatted = [f"{k}: ckpt{src}->model{dst}" for k, src, dst in skipped[:8]]
        print(f"[pretrained-ssl] {tag} skipped shape mismatch (first 8): {formatted}")
    return missing_keys, combined_unexpected, skipped


def make_criterion(args: argparse.Namespace, device: torch.device):
    return MelReconstructionLoss(
        lambda_mel=1.0,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
    ).to(device)


def unpack_batch(batch, device: torch.device, crop_mouth: bool = False, mouth_roi: list[int] = [45, 80, 32, 80]):
    if len(batch) == 5:
        video, landmarks, target, lengths, paths = batch
    elif len(batch) == 4 and torch.is_tensor(batch[1]) and batch[1].dim() == 4:
        video, landmarks, target, lengths = batch
        paths = None
    else:
        raise ValueError("Dataset batch shape is not recognized. Ensure VNLipDatasetV2 returns correct outputs.")

    if crop_mouth:
        y1, y2, x1, x2 = mouth_roi
        video = video[:, :, :, y1:y2, x1:x2]

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


def get_state_dict(model: torch.nn.Module) -> dict:
    """Unwrap DataParallel model before extracting state_dict to keep checkpoints clean."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


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
            "encoder_state_dict": get_state_dict(encoder),
            "decoder_state_dict": get_state_dict(decoder),
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
    
    # Resolve underlying modules if using nn.DataParallel
    enc_module = encoder.module if isinstance(encoder, torch.nn.DataParallel) else encoder
    dec_module = decoder.module if isinstance(decoder, torch.nn.DataParallel) else decoder

    enc_module.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    dec_module.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    
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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> float:
    encoder.train()
    decoder.train()
    amp_enabled = device.type == "cuda" and args.amp
    total_loss = 0.0
    num_batches = 0
    accum_steps = max(1, args.accum_steps)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        crop_mouth = getattr(args, "crop_mouth", False)
        mouth_roi = getattr(args, "mouth_roi", [45, 80, 32, 80])
        video, landmarks, target, lengths, _ = unpack_batch(batch, device, crop_mouth=crop_mouth, mouth_roi=mouth_roi)
        reset_snn_if_needed(encoder, args.encoder_type)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            z = encoder(video, landmarks)
            pred = decoder(z, target_len=target.shape[1])

        with torch.amp.autocast("cuda", enabled=False):
            loss = criterion(pred.float(), target.float(), lengths)
            # Scale loss to support gradient accumulation
            loss = loss / accum_steps

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite train loss: {float(loss.detach().cpu()) * accum_steps}")

        scaler.scale(loss).backward()

        # Step weights only after accumulating gradients for configured steps
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    args.max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and args.lr_scheduler == "onecycle":
                scheduler.step()

        reset_snn_if_needed(encoder, args.encoder_type)

        total_loss += float(loss.detach().cpu()) * accum_steps
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
        crop_mouth = getattr(args, "crop_mouth", False)
        mouth_roi = getattr(args, "mouth_roi", [45, 80, 32, 80])
        video, landmarks, target, lengths, paths = unpack_batch(batch, device, crop_mouth=crop_mouth, mouth_roi=mouth_roi)
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
    encoder, decoder = build_models(
        device,
        args.encoder_type,
        args.decoder_type,
        dataset.landmark_num_points,
        fusion_type=getattr(args, "fusion_type", "cross_attn"),
        target_type="mel_hifigan",
    )
    
    # Load Pre-trained SSL Weights if specified
    if getattr(args, "pretrained_ssl", None) is not None:
        pretrained_ssl_path = resolve_path(args.pretrained_ssl)
        if pretrained_ssl_path and pretrained_ssl_path.is_file():
            print(f"[pretrained-ssl] Loading weights from {safe_text(pretrained_ssl_path)}")
            checkpoint = torch.load(pretrained_ssl_path, map_location=device, weights_only=False)
            
            if not hasattr(encoder, "visual_encoder"):
                print("[pretrained-ssl] WARNING: Could not locate visual_encoder in encoder!")
            elif isinstance(checkpoint, dict) and "online_encoder_state_dict" in checkpoint:
                load_compatible_state_dict(
                    encoder.visual_encoder,
                    checkpoint["online_encoder_state_dict"],
                    "visual_encoder from online_encoder_state_dict",
                )
            elif isinstance(checkpoint, dict) and "visual_encoder" in checkpoint:
                load_compatible_state_dict(
                    encoder.visual_encoder,
                    checkpoint["visual_encoder"],
                    "visual_encoder",
                )
            elif isinstance(checkpoint, dict) and "backbone" in checkpoint and checkpoint["backbone"] is not None:
                load_compatible_state_dict(
                    encoder.visual_encoder.backbone,
                    checkpoint["backbone"],
                    "visual_encoder.backbone",
                )
            elif isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
                load_compatible_state_dict(
                    encoder.visual_encoder,
                    checkpoint,
                    "visual_encoder from raw state_dict",
                )
            else:
                print("[pretrained-ssl] WARNING: Unsupported SSL checkpoint format.")
        else:
            print(f"[pretrained-ssl] WARNING: Pretrained SSL path {args.pretrained_ssl} does not exist or is not a file!")

    criterion = make_criterion(args, device)

    # Wrap model in nn.DataParallel to utilize all available GPUs
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Wrapping modules in nn.DataParallel!")
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)

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

    # Initialize LR Scheduler
    scheduler = None
    if args.lr_scheduler == "onecycle":
        steps_per_epoch = int(math.ceil(len(train_loader) / args.accum_steps))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[encoder_lr, decoder_lr],
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 10% warmup
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )
        print(f"[scheduler] OneCycleLR initialized with steps_per_epoch={steps_per_epoch} (total_steps={args.epochs * steps_per_epoch})")
    elif args.lr_scheduler == "cosine":
        def get_lr_lambda(warmup_epochs: int, total_epochs: int):
            def lr_lambda(epoch: int) -> float:
                current_epoch = epoch + 1
                if current_epoch <= warmup_epochs:
                    return float(current_epoch) / float(max(1, warmup_epochs))
                progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                progress = min(1.0, max(0.0, progress))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_lambda

        lr_lambda = get_lr_lambda(args.warmup_epochs, args.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"[scheduler] CosineAnnealing with manual Warmup initialized (warmup={args.warmup_epochs} epochs)")

    start_epoch = 1
    best_val_loss = float("inf")
    resume_path = resolve_path(args.resume)
    if resume_path is not None:
        start_epoch, best_val_loss = load_resume(resume_path, encoder, decoder, optimizer, scaler, device)
        print(f"[resume] {safe_text(resume_path)} -> start_epoch={start_epoch}")

    print(f"[device] {device}")
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[device] Multi-GPU Active: {torch.cuda.device_count()}x GPUs utilized.")
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
    print(f"[optim] accum_steps={args.accum_steps} batch_size={args.batch_size} effective_batch_size={args.batch_size * args.accum_steps}")

    target_max_frames = args.max_frames

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        if args.curriculum:
            current_frames = get_curriculum_frames(epoch, args.epochs, target_max_frames)
            set_dataset_max_frames(dataset, current_frames)
            set_dataset_max_frames(train_loader.dataset, current_frames)
            print(f"[curriculum] Epoch {epoch:04d} -> active max_frames={current_frames}")

        train_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            args=args,
            scheduler=scheduler,
        )

        if args.curriculum:
            # Restore target_max_frames for validation / standard evaluation
            set_dataset_max_frames(dataset, target_max_frames)
            if val_loader is not None:
                set_dataset_max_frames(val_loader.dataset, target_max_frames)

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

        # Clear VRAM cache actively at the end of each epoch to prevent leakage OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best": best_val_loss}
        history.append(row)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"history": history, "config": vars(args)}, f, indent=2)

        if scheduler is not None and args.lr_scheduler == "cosine":
            scheduler.step()

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        lr_text = ", ".join([f"{lr:.2e}" for lr in current_lrs])

        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        mark = " best" if is_best else ""
        print(f"[epoch {epoch:04d}] train={train_loss:.6f} val={val_text} best={best_val_loss:.6f}{mark} | lrs=[{lr_text}]")

    print(f"[done] output={safe_text(output_dir)}")
    print(f"[best] {safe_text(output_dir / 'best_model.pth')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full mel reconstruction in modular structure.")
    parser.add_argument("--data-dir", default=default_data_dir())
    parser.add_argument("--dataset-output-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default="checkpoints_modular")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "nonsnn", "cnn_transformer", "snn", "resnet18_temporal", "resnet2plus1d_spatial_motion"])
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
    parser.add_argument("--accum-steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--force-full-frame", action="store_true")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--crop-mouth", action="store_true", help="Crop mouth region of interest from the input video.")
    parser.add_argument("--mouth-roi", type=int, nargs=4, default=[45, 80, 32, 80], help="Mouth ROI y1, y2, x1, x2")
    parser.add_argument("--fusion-type", default="cross_attn", choices=["concat", "cross_attn", "gated_residual"], help="Landmark fusion type.")
    parser.add_argument("--lr-scheduler", default="constant", choices=["constant", "onecycle", "cosine"], help="Learning rate scheduler type.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs for cosine scheduler.")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (progressive max_frames).")
    parser.add_argument("--pretrained-ssl", default=None, help="Path to self-supervised pre-trained backbone checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
