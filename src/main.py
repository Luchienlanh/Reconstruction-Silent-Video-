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
from torch.utils.data import DataLoader
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
from models.encoders.factory import (
    build_encoder,
    is_flow_encoder_type,
    VisualLandmarkEncoder,
    VisualLandmarkEncoderV2,
    VisualLandmarkEncoderGatedResidual,
    VisualLandmarkEncoderLandmarkFirst,
)
from models.decoders.siren import TFiLMSIRENDecoder
from models.decoders.direct_tcn import DirectTCNMelDecoder
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
    normalized = str(path).replace("\\", "/")
    p = Path(normalized)
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
    """Recursively unwrap dataset wrappers to find the actual dataset and set max_frames."""
    if hasattr(dataset, "dataset"):
        set_dataset_max_frames(dataset.dataset, max_frames)
    elif hasattr(dataset, "max_frames"):
        dataset.max_frames = max_frames


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def get_visual_encoder_module(encoder: torch.nn.Module) -> Optional[torch.nn.Module]:
    enc_module = unwrap_module(encoder)
    return getattr(enc_module, "visual_encoder", None)


def set_module_trainable(module: Optional[torch.nn.Module], trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = trainable


def build_base_decoder(
    decoder_type: str,
    target_type: str = "mel_hifigan",
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.0,
):
    decoder_type = decoder_type.lower()
    common = dict(hidden_dim=hidden_dim, out_dim=80, num_layers=num_layers, use_conv=True)
    output_act = None if target_type == "mel_hifigan" else "tanh"
    if decoder_type == "direct_tcn":
        return DirectTCNMelDecoder(hidden_dim=hidden_dim, out_dim=80, num_layers=num_layers, dropout=dropout)
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


def build_models(
    device: torch.device,
    encoder_type: str,
    decoder_type: str,
    num_landmark_points: int,
    fusion_type: str = "landmark_first",
    target_type: str = "mel_hifigan",
    decoder_hidden_dim: int = 256,
    decoder_num_layers: int = 4,
    decoder_dropout: float = 0.0,
):
    if is_flow_encoder_type(encoder_type):
        encoder = build_encoder(encoder_type, num_landmark_points=num_landmark_points).to(device)
    else:
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
        elif fusion_type == "landmark_first":
            encoder = VisualLandmarkEncoderLandmarkFirst(
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

    base_decoder = build_base_decoder(
        decoder_type,
        target_type=target_type,
        hidden_dim=decoder_hidden_dim,
        num_layers=decoder_num_layers,
        dropout=decoder_dropout,
    ).to(device)
    decoder = MelTemporalUpsampleDecoder(
        base_decoder,
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    return encoder, decoder


def make_criterion(
    args: argparse.Namespace,
    device: torch.device,
    mel_mean: Optional[torch.Tensor] = None,
    mel_std: Optional[torch.Tensor] = None,
):
    return MelReconstructionLoss(
        lambda_mel=1.0,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
        mel_mean=mel_mean if getattr(args, "normalize_mel_loss", True) else None,
        mel_std=mel_std if getattr(args, "normalize_mel_loss", True) else None,
    ).to(device)


def build_optimizer_groups(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    args: argparse.Namespace,
) -> list[dict]:
    enc_module = unwrap_module(encoder)
    visual_encoder = getattr(enc_module, "visual_encoder", None)
    is_snn = args.encoder_type.lower() == "snn"
    is_finer = args.decoder_type.lower() == "finer"

    visual_base_lr = args.lr * (2.5 if is_snn else 1.0)
    visual_lr = visual_base_lr * args.visual_lr_scale
    fusion_lr = args.lr * args.fusion_lr_scale
    decoder_lr = args.lr * args.decoder_lr_scale * (0.5 if is_finer else 1.0)

    groups = []
    if visual_encoder is not None:
        visual_params = list(visual_encoder.parameters())
        visual_param_ids = {id(param) for param in visual_params}
        fusion_params = [param for param in encoder.parameters() if id(param) not in visual_param_ids]
        groups.append({"params": visual_params, "lr": visual_lr, "name": "visual"})
        if fusion_params:
            groups.append({"params": fusion_params, "lr": fusion_lr, "name": "fusion"})
    else:
        groups.append({"params": list(encoder.parameters()), "lr": visual_lr, "name": "encoder"})
    groups.append({"params": list(decoder.parameters()), "lr": decoder_lr, "name": "decoder"})
    return groups


def unpack_batch(batch, device: torch.device, crop_mouth: bool = False, mouth_roi: list[int] = [45, 80, 32, 80]):
    flow = None
    if len(batch) == 6:
        video, flow, landmarks, target, lengths, paths = batch
    elif len(batch) == 5:
        video, landmarks, target, lengths, paths = batch
    elif len(batch) == 4 and torch.is_tensor(batch[1]) and batch[1].dim() == 4:
        video, landmarks, target, lengths = batch
        paths = None
    else:
        raise ValueError("Dataset batch shape is not recognized. Ensure VNLipDatasetV2 returns correct outputs.")

    if crop_mouth and video.dim() == 5:
        y1, y2, x1, x2 = mouth_roi
        video = video[:, :, :, y1:y2, x1:x2]
        if flow is not None:
            flow = flow[:, :, :, y1:y2, x1:x2]

    video = video.to(device, non_blocking=True)
    if flow is not None:
        video = (video, flow.to(device, non_blocking=True))

    return (
        video,
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
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            print(f"[resume] WARNING: optimizer state not loaded ({exc}). Continuing with fresh optimizer.")
    if "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except ValueError as exc:
            print(f"[resume] WARNING: scaler state not loaded ({exc}). Continuing with fresh scaler.")
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
    if getattr(args, "freeze_visual_active", False):
        visual_encoder = get_visual_encoder_module(encoder)
        if visual_encoder is not None:
            visual_encoder.eval()
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


@torch.no_grad()
def evaluate_mean_baseline(
    loader: DataLoader,
    criterion,
    device: torch.device,
    args: argparse.Namespace,
    mel_mean: Optional[torch.Tensor],
) -> Optional[float]:
    if mel_mean is None:
        return None

    total_loss = 0.0
    num_batches = 0
    mean = mel_mean.to(device=device, dtype=torch.float32).view(1, 1, -1)
    for batch in tqdm(loader, desc="mean-baseline", leave=False):
        crop_mouth = getattr(args, "crop_mouth", False)
        mouth_roi = getattr(args, "mouth_roi", [45, 80, 32, 80])
        _, _, target, lengths, _ = unpack_batch(batch, device, crop_mouth=crop_mouth, mouth_roi=mouth_roi)
        pred = mean.expand(target.shape[0], target.shape[1], -1).contiguous()
        loss = criterion(pred.float(), target.float(), lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite mean baseline loss: {float(loss.detach().cpu())}")
        total_loss += float(loss.detach().cpu())
        num_batches += 1

    if num_batches == 0:
        return None
    return total_loss / num_batches


def create_loaders(args: argparse.Namespace):
    data_dir = resolve_path(args.data_dir)
    dataset_output_dir = resolve_path(args.dataset_output_dir)
    flow_cache_dir = resolve_path(args.flow_cache_dir) if getattr(args, "flow_cache_dir", None) else None
    use_optical_flow = bool(getattr(args, "use_optical_flow", False) or is_flow_encoder_type(args.encoder_type))
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")

    base_dataset = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=False,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(dataset_output_dir or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
        force_full_frame=args.force_full_frame,
        use_optical_flow=use_optical_flow,
        flow_cache_dir=str(flow_cache_dir) if flow_cache_dir is not None else None,
        flow_method=args.flow_method,
        flow_scale=args.flow_scale,
    )

    files = list(base_dataset.files)
    if args.limit is not None:
        limit = max(1, min(int(args.limit), len(files)))
        files = files[:limit]

    total = len(files)
    val_count = max(1, int(round(total * args.val_ratio))) if total > 1 else 0
    train_count = total - val_count
    if train_count <= 0:
        raise RuntimeError("Dataset is too small for the requested validation split.")

    rng = random.Random(args.seed)
    shuffled_files = list(files)
    rng.shuffle(shuffled_files)
    if val_count > 0:
        val_files = sorted(shuffled_files[:val_count])
        train_files = sorted(shuffled_files[val_count:])
    else:
        train_files = sorted(shuffled_files)
        val_files = []

    train_set = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=True,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(dataset_output_dir or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
        force_full_frame=args.force_full_frame,
        use_optical_flow=use_optical_flow,
        flow_cache_dir=str(flow_cache_dir) if flow_cache_dir is not None else None,
        flow_method=args.flow_method,
        flow_scale=args.flow_scale,
    )
    train_set.files = train_files

    val_set = None
    if val_files:
        val_set = VNLipDatasetV2(
            data_dir=str(data_dir),
            max_frames=args.max_frames,
            random_crop=False,
            return_path=True,
            target_type="mel_hifigan",
            use_landmarks=True,
            dataset_output_dir=str(dataset_output_dir or PROJECT_ROOT / "Dataset_Output"),
            enable_fallback=not args.disable_fallback,
            force_full_frame=args.force_full_frame,
            use_optical_flow=use_optical_flow,
            flow_cache_dir=str(flow_cache_dir) if flow_cache_dir is not None else None,
            flow_method=args.flow_method,
            flow_scale=args.flow_scale,
        )
        val_set.files = val_files

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

    return train_set, train_loader, val_loader, train_count, val_count


def _target_from_dataset_item(item) -> torch.Tensor:
    if len(item) >= 4 and torch.is_tensor(item[1]) and item[1].dim() == 4:
        return item[3]
    if len(item) >= 3 and torch.is_tensor(item[1]) and item[1].dim() == 3:
        return item[2]
    if len(item) >= 2:
        return item[1]
    raise ValueError("Dataset item does not contain a target tensor.")


def compute_mel_stats(dataset, max_samples: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    limit = len(dataset) if max_samples <= 0 else min(len(dataset), int(max_samples))
    if limit <= 0:
        raise RuntimeError("Cannot compute mel stats from an empty dataset.")

    total = None
    total_sq = None
    count = 0
    for idx in tqdm(range(limit), desc="mel-stats", leave=False):
        target = _target_from_dataset_item(dataset[idx]).float()
        if target.dim() != 2:
            raise ValueError(f"Expected mel target (T,80), got {tuple(target.shape)}")
        total = target.sum(dim=0) if total is None else total + target.sum(dim=0)
        total_sq = target.pow(2).sum(dim=0) if total_sq is None else total_sq + target.pow(2).sum(dim=0)
        count += int(target.shape[0])

    count = max(count, 1)
    mean = total / count
    var = (total_sq / count) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt().clamp_min(0.05)
    return mean.cpu(), std.cpu()


def init_decoder_output_bias_from_mel_mean(decoder: torch.nn.Module, mel_mean: torch.Tensor) -> bool:
    dec_module = unwrap_module(decoder)
    base_decoder = getattr(dec_module, "base_decoder", dec_module)
    for attr in ("output", "final_layer"):
        layer = getattr(base_decoder, attr, None)
        if isinstance(layer, torch.nn.Linear) and layer.bias is not None and layer.bias.numel() == mel_mean.numel():
            with torch.no_grad():
                layer.bias.copy_(mel_mean.to(device=layer.bias.device, dtype=layer.bias.dtype))
            return True
    return False


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "checkpoints_modular"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, train_loader, val_loader, train_count, val_count = create_loaders(args)
    mel_mean = None
    mel_std = None
    if getattr(args, "normalize_mel_loss", True) or getattr(args, "init_decoder_bias_from_data", True):
        mel_mean, mel_std = compute_mel_stats(
            train_loader.dataset,
            max_samples=max(0, int(getattr(args, "mel_stats_max_samples", 0))),
        )
        args.mel_loss_mean = [float(x) for x in mel_mean.tolist()]
        args.mel_loss_std = [float(x) for x in mel_std.tolist()]

    encoder, decoder = build_models(
        device,
        args.encoder_type,
        args.decoder_type,
        dataset.landmark_num_points,
        fusion_type=getattr(args, "fusion_type", "landmark_first"),
        target_type="mel_hifigan",
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        decoder_dropout=args.decoder_dropout,
    )
    if getattr(args, "init_decoder_bias_from_data", True) and mel_mean is not None:
        bias_ok = init_decoder_output_bias_from_mel_mean(decoder, mel_mean)
        print(f"[mel-stats] decoder_bias_from_data={'yes' if bias_ok else 'no'}")
    criterion = make_criterion(args, device, mel_mean=mel_mean, mel_std=mel_std)

    if args.visual_lr_scale is None:
        args.visual_lr_scale = 1.0
    if args.fusion_lr_scale is None:
        args.fusion_lr_scale = 5.0
    if args.decoder_lr_scale is None:
        args.decoder_lr_scale = 2.0 if args.decoder_type.lower() == "direct_tcn" else (3.0 if args.decoder_type.lower() == "siren" else 1.0)
    if args.freeze_visual_epochs is None:
        args.freeze_visual_epochs = 0

    # Wrap model in nn.DataParallel to utilize all available GPUs
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[device] Found {torch.cuda.device_count()} GPUs. Wrapping modules in nn.DataParallel!")
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)

    optimizer_groups = build_optimizer_groups(encoder, decoder, args)

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
            max_lr=[group["lr"] for group in optimizer_groups],
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
    if mel_mean is not None and mel_std is not None:
        print(
            "[mel-stats] normalize_loss={} mean_avg={:.4f} std_avg={:.4f} max_samples={}".format(
                bool(getattr(args, "normalize_mel_loss", True)),
                float(mel_mean.mean()),
                float(mel_std.mean()),
                "all" if args.mel_stats_max_samples <= 0 else args.mel_stats_max_samples,
            )
        )
    if getattr(args, "use_optical_flow", False) or is_flow_encoder_type(args.encoder_type):
        print(
            f"[flow] enabled method={args.flow_method} cache={safe_text(resolve_path(args.flow_cache_dir)) if args.flow_cache_dir else 'none'}"
        )
    print(
        f"[model] encoder={args.encoder_type} decoder={args.decoder_type} "
        f"hidden={args.decoder_hidden_dim} layers={args.decoder_num_layers} "
        f"landmarks={dataset.landmark_num_points}"
    )
    lr_summary = ", ".join(f"{group.get('name', idx)}={group['lr']:.2e}" for idx, group in enumerate(optimizer_groups))
    print(f"[lr] {lr_summary}")
    if args.freeze_visual_epochs > 0:
        print(f"[freeze] visual_encoder frozen for first {args.freeze_visual_epochs} epoch(s)")
    print(
        "[loss] mel=1.0 delta={} delta2={} energy={}".format(
            args.lambda_delta,
            args.lambda_delta2,
            args.lambda_energy,
        )
    )
    print(f"[optim] accum_steps={args.accum_steps} batch_size={args.batch_size} effective_batch_size={args.batch_size * args.accum_steps}")

    mean_train_loss = evaluate_mean_baseline(train_loader, criterion, device, args, mel_mean)
    mean_val_loss = evaluate_mean_baseline(val_loader, criterion, device, args, mel_mean) if val_loader is not None else None
    if mean_train_loss is not None:
        mean_val_text = "n/a" if mean_val_loss is None else f"{mean_val_loss:.6f}"
        print(f"[baseline] mean_train={mean_train_loss:.6f} mean_val={mean_val_text}")

    target_max_frames = args.max_frames

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        freeze_visual = epoch <= args.freeze_visual_epochs
        args.freeze_visual_active = freeze_visual
        set_module_trainable(get_visual_encoder_module(encoder), not freeze_visual)
        if epoch == 1 and freeze_visual:
            print("[freeze] visual_encoder frozen; training landmark/fusion/decoder first")
        elif epoch == args.freeze_visual_epochs + 1 and args.freeze_visual_epochs > 0:
            print("[freeze] visual_encoder unfrozen")

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

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best": best_val_loss,
            "mean_train_loss": mean_train_loss,
            "mean_val_loss": mean_val_loss,
        }
        history.append(row)
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"history": history, "config": vars(args)}, f, indent=2)

        if scheduler is not None and args.lr_scheduler == "cosine":
            scheduler.step()

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        lr_text = ", ".join([f"{lr:.2e}" for lr in current_lrs])

        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        gap_text = ""
        if val_loss is not None and mean_val_loss is not None:
            gap_text = f" gap_vs_mean={val_loss - mean_val_loss:+.6f}"
        mark = " best" if is_best else ""
        print(f"[epoch {epoch:04d}] train={train_loss:.6f} val={val_text} best={best_val_loss:.6f}{gap_text}{mark} | lrs=[{lr_text}]")

    print(f"[done] output={safe_text(output_dir)}")
    print(f"[best] {safe_text(output_dir / 'best_model.pth')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full mel reconstruction in modular structure.")
    parser.add_argument("--data-dir", default=default_data_dir())
    parser.add_argument("--dataset-output-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default="checkpoints_modular")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "nonsnn", "cnn_transformer", "snn", "ephrat_flow_r2plus1d", "flow_r2plus1d", "two_tower_flow"])
    parser.add_argument(
        "--decoder-type",
        default="siren",
        choices=["siren", "direct_tcn", "wire", "finer", "dual", "dual_wrap", "wrap_siren", "wrap_fisin", "wrap", "wrap_wire", "wrap_fiwi"],
    )
    parser.add_argument("--decoder-hidden-dim", type=int, default=256, help="Hidden dimension for the decoder.")
    parser.add_argument("--decoder-num-layers", type=int, default=4, help="Number of decoder layers/blocks.")
    parser.add_argument("--decoder-dropout", type=float, default=0.0, help="Dropout for decoders that support it.")
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
    parser.add_argument("--lambda-delta", type=float, default=0.25)
    parser.add_argument("--lambda-delta2", type=float, default=0.05)
    parser.add_argument("--lambda-energy", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--force-full-frame", action="store_true")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--use-optical-flow", default=False, action=argparse.BooleanOptionalAction, help="Return dense optical flow from the dataset. Automatically enabled by flow encoders.")
    parser.add_argument("--flow-cache-dir", default="flow_cache", help="Directory for cached optical-flow tensors.")
    parser.add_argument("--flow-method", default="farneback", choices=["farneback", "pseudo"], help="Optical-flow backend. pseudo is a no-cv2 fallback.")
    parser.add_argument("--flow-scale", type=float, default=10.0, help="Divide Farneback pixel displacement by this value before training.")
    parser.add_argument("--crop-mouth", action="store_true", help="Crop mouth region of interest from the input video.")
    parser.add_argument("--mouth-roi", type=int, nargs=4, default=[45, 80, 32, 80], help="Mouth ROI y1, y2, x1, x2")
    parser.add_argument("--fusion-type", default="landmark_first", choices=["concat", "cross_attn", "gated_residual", "landmark_first"], help="Landmark fusion type.")
    parser.add_argument("--lr-scheduler", default="constant", choices=["constant", "onecycle", "cosine"], help="Learning rate scheduler type.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs for cosine scheduler.")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (progressive max_frames).")
    parser.add_argument("--visual-lr-scale", type=float, default=None, help="Scale base LR for the visual encoder. Default: 1.0.")
    parser.add_argument("--fusion-lr-scale", type=float, default=None, help="Scale base LR for landmark/fusion layers. Default: 5.0.")
    parser.add_argument("--decoder-lr-scale", type=float, default=None, help="Scale base LR for decoder. Default: 2.0 for direct_tcn, 3.0 for siren, else 1.0.")
    parser.add_argument("--freeze-visual-epochs", type=int, default=None, help="Freeze visual encoder for the first N epochs. Default: 0.")
    parser.add_argument("--normalize-mel-loss", default=True, action=argparse.BooleanOptionalAction, help="Normalize mel/delta losses per mel bin using train-set stats.")
    parser.add_argument("--init-decoder-bias-from-data", default=True, action=argparse.BooleanOptionalAction, help="Initialize decoder output bias with train-set mel mean.")
    parser.add_argument("--mel-stats-max-samples", type=int, default=0, help="Max train samples for mel stats. 0 means all train samples.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
