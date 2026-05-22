"""
pretrain_ssl_mouth_only.py
==========================
Self-Supervised Pre-training (SSL) script for lip-reading visual feature extractors.
Uses aligned full-frame video as input and reconstructs only the fixed mouth patch.
The pretraining decoder is discarded after SSL; the visual encoder weights are loaded
into the main mel-reconstruction system.
"""

import os
import sys
import gc
import time
import argparse
import random
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm.auto import tqdm

# Ensure parent and src directories are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import dataset components
from src.data.dataset import VNLipDatasetV2, collate_pad_v2

# Try to import SpikingJelly components
try:
    from spikingjelly.activation_based import functional
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def reset_snn_net(model: nn.Module) -> None:
    if SNN_AVAILABLE:
        unwrapped = model.module if hasattr(model, "module") else model
        functional.reset_net(unwrapped.backbone)


class MouthPatchDecoder(nn.Module):
    """
    Lightweight SSL-only decoder.
    Takes a sequence representation of shape (B, T, 512) and reconstructs a mouth
    patch of shape (B, 1, T, H, W). This decoder is not used by the main mel model.
    """
    def __init__(
        self,
        z_dim: int = 512,
        out_channels: int = 1,
        out_size: tuple[int, int] = (35, 48),
        base_channels: int = 192,
    ):
        super().__init__()
        self.out_size = out_size
        self.seed_h = 5
        self.seed_w = 6
        self.base_channels = base_channels

        self.seed = nn.Sequential(
            nn.Linear(z_dim, base_channels * self.seed_h * self.seed_w),
            nn.LayerNorm(base_channels * self.seed_h * self.seed_w),
            nn.SiLU(inplace=True),
        )

        self.spatial_upsample = nn.Sequential(
            # (B * T, C, 5, 6) -> (B * T, 128, 10, 12)
            nn.ConvTranspose2d(base_channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),

            # (B * T, 128, 10, 12) -> (B * T, 64, 20, 24)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),

            # (B * T, 64, 20, 24) -> (B * T, 32, 40, 48)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )

        self.temporal_refine = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (B, T, 512)
        B, T, C = z.shape
        x = self.seed(z.reshape(B * T, C))
        x = x.reshape(B * T, self.base_channels, self.seed_h, self.seed_w)
        x = self.spatial_upsample(x)

        if x.shape[-2:] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)

        H, W = self.out_size
        x = x.reshape(B, T, 32, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return self.temporal_refine(x)


class SSLAutoencoder(nn.Module):
    """
    Wrapper module to combine the visual encoder and SSL-only mouth decoder.
    """
    def __init__(self, visual_encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.backbone = visual_encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.visual_encoder(x)  # (B, T, 512)
        return self.decoder(z)      # (B, 1, T, H_roi, W_roi)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_data_dir() -> str:
    full_frame = PROJECT_ROOT / "FullFrame_test"
    if full_frame.is_dir():
        return "Processed_Data_Mel_HiFiGAN_FullFrame"
    return "Processed_Data_Mel_HiFiGAN"


def build_visual_encoder(encoder_type: str) -> nn.Module:
    encoder_type = encoder_type.lower()
    if encoder_type == "snn" and not SNN_AVAILABLE:
        raise ImportError("SpikingJelly is not installed, cannot use SNN visual encoder.")
    from src.models.encoders.factory import build_encoder
    print(f"[visual-encoder] Instantiating full visual encoder: {encoder_type}")
    return build_encoder(encoder_type)


def parse_mouth_roi(values: list[int]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("--mouth-roi expects four integers: y1 y2 x1 x2")
    y1, y2, x1, x2 = [int(v) for v in values]
    if not (0 <= y1 < y2 <= 112 and 0 <= x1 < x2 <= 112):
        raise ValueError(f"Invalid mouth ROI {(y1, y2, x1, x2)} for 112x112 frames.")
    return y1, y2, x1, x2


def crop_mouth(video: torch.Tensor, mouth_roi: tuple[int, int, int, int]) -> torch.Tensor:
    y1, y2, x1, x2 = mouth_roi
    return video[:, :, :, y1:y2, x1:x2]


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    flat = x.reshape(b * c * t, 1, h, w)
    kernel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    kernel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=x.device,
        dtype=x.dtype,
    ).unsqueeze(0)
    grad_x = F.conv2d(flat, kernel_x, padding=1)
    grad_y = F.conv2d(flat, kernel_y, padding=1)
    edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    return edge.reshape(b, c, t, h, w)


def compute_reconstruction_metrics(
    mouth_recon: torch.Tensor,
    mouth_gt: torch.Tensor,
    mse_weight: float,
    temporal_weight: float,
    edge_weight: float,
) -> dict[str, torch.Tensor]:
    loss_l1 = F.l1_loss(mouth_recon, mouth_gt)
    loss_mse = F.mse_loss(mouth_recon, mouth_gt)

    if mouth_gt.shape[2] > 1:
        diff_gt = mouth_gt[:, :, 1:] - mouth_gt[:, :, :-1]
        diff_recon = mouth_recon[:, :, 1:] - mouth_recon[:, :, :-1]
        loss_temporal = F.l1_loss(diff_recon, diff_gt)
        gt_motion = diff_gt.abs().mean()
        recon_motion = diff_recon.abs().mean()
        motion_ratio = recon_motion / gt_motion.clamp_min(1e-6)
    else:
        loss_temporal = mouth_gt.new_tensor(0.0)
        gt_motion = mouth_gt.new_tensor(0.0)
        recon_motion = mouth_gt.new_tensor(0.0)
        motion_ratio = mouth_gt.new_tensor(0.0)

    loss_edge = F.l1_loss(sobel_edges(mouth_recon), sobel_edges(mouth_gt))
    loss = loss_l1 + mse_weight * loss_mse + temporal_weight * loss_temporal + edge_weight * loss_edge

    return {
        "loss": loss,
        "l1": loss_l1,
        "mse": loss_mse,
        "temporal": loss_temporal,
        "edge": loss_edge,
        "gt_motion": gt_motion,
        "recon_motion": recon_motion,
        "motion_ratio": motion_ratio,
    }


def save_reconstruction_plot(
    original: torch.Tensor,
    mouth_gt: torch.Tensor,
    mouth_recon: torch.Tensor,
    output_path: Path,
    epoch: int,
    mouth_roi: tuple[int, int, int, int],
) -> None:
    """
    Save full-frame context plus mouth target/reconstruction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    orig = original.detach().cpu().squeeze(0).numpy()  # (T, 112, 112)
    gt = mouth_gt.detach().cpu().squeeze(0).numpy()
    recon = mouth_recon.detach().cpu().squeeze(0).numpy()
    T = orig.shape[0]
    y1, y2, x1, x2 = mouth_roi

    num_frames_to_plot = min(5, T)
    indices = np.linspace(0, T - 1, num_frames_to_plot, dtype=int)

    fig, axes = plt.subplots(3, num_frames_to_plot, figsize=(3 * num_frames_to_plot, 7), constrained_layout=True)
    
    if num_frames_to_plot == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, frame_idx in enumerate(indices):
        axes[0, col_idx].imshow(orig[frame_idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col_idx].axis("off")
        rect_orig = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--")
        axes[0, col_idx].add_patch(rect_orig)
        if col_idx == 0:
            axes[0, col_idx].set_title("Full frame + ROI", loc="left", fontsize=12, fontweight="bold")
        else:
            axes[0, col_idx].set_title(f"Frame {frame_idx}")

        axes[1, col_idx].imshow(gt[frame_idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col_idx].axis("off")
        if col_idx == 0:
            axes[1, col_idx].set_title("Mouth target", loc="left", fontsize=12, fontweight="bold")

        axes[2, col_idx].imshow(recon[frame_idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, col_idx].axis("off")
        if col_idx == 0:
            axes[2, col_idx].set_title("Mouth reconstructed", loc="left", fontsize=12, fontweight="bold")

    fig.suptitle(f"SSL Mouth Patch Reconstruction (Epoch {epoch})", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    accum_steps: int,
    is_snn: bool,
    amp: bool,
    mouth_roi: tuple[int, int, int, int],
    mse_weight: float,
    temporal_weight: float,
    edge_weight: float,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "mse": 0.0,
        "temporal": 0.0,
        "edge": 0.0,
        "gt_motion": 0.0,
        "recon_motion": 0.0,
        "motion_ratio": 0.0,
    }
    processed_batches = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  SSL Train")
    for batch_idx, batch in enumerate(pbar):
        video, _, _ = batch
        video = video.to(device, non_blocking=True)  # (B, 1, T, 112, 112)

        # Reset SNN states if using SNN
        if is_snn:
            reset_snn_net(model)

        try:
            with torch.amp.autocast("cuda", enabled=amp):
                mouth_recon = model(video)
            mouth_gt = crop_mouth(video, mouth_roi)

            with torch.amp.autocast("cuda", enabled=False):
                metrics = compute_reconstruction_metrics(
                    mouth_recon.float(),
                    mouth_gt.float(),
                    mse_weight=mse_weight,
                    temporal_weight=temporal_weight,
                    edge_weight=edge_weight,
                )
                loss = metrics["loss"] / accum_steps

            # Scale gradients
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for key in totals:
                totals[key] += float(metrics[key].detach().cpu())
            processed_batches += 1
            pbar.set_postfix(
                loss=f"{float(metrics['loss'].detach().cpu()):.5f}",
                l1=f"{float(metrics['l1'].detach().cpu()):.5f}",
                motion=f"{float(metrics['motion_ratio'].detach().cpu()):.3f}",
            )

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Out of memory error in batch index {batch_idx}. Skipping.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise

        finally:
            if is_snn:
                reset_snn_net(model)

    return {key: value / max(1, processed_batches) for key, value in totals.items()}


@torch.no_grad()
def save_train_reconstruction_sample(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_snn: bool,
    epoch: int,
    plot_dir: Path,
    mouth_roi: tuple[int, int, int, int],
) -> None:
    model.eval()
    batch = next(iter(loader))
    video, _, _ = batch
    video = video.to(device, non_blocking=True)

    if is_snn:
        reset_snn_net(model)

    mouth_recon = model(video)
    mouth_gt = crop_mouth(video, mouth_roi)
    plot_path = plot_dir / f"train_reconstruction_epoch_{epoch:04d}.png"
    save_reconstruction_plot(video[0], mouth_gt[0], mouth_recon[0], plot_path, epoch, mouth_roi)

    if is_snn:
        reset_snn_net(model)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_snn: bool,
    epoch: int,
    plot_dir: Path,
    mouth_roi: tuple[int, int, int, int],
    mse_weight: float,
    temporal_weight: float,
    edge_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "mse": 0.0,
        "temporal": 0.0,
        "edge": 0.0,
        "gt_motion": 0.0,
        "recon_motion": 0.0,
        "motion_ratio": 0.0,
    }
    processed_batches = 0

    # Save visual plot for the first sample in the validation set
    plotted = False

    pbar = tqdm(loader, desc="  SSL Val")
    for batch_idx, batch in enumerate(pbar):
        video, _, _ = batch
        video = video.to(device, non_blocking=True)

        if is_snn:
            reset_snn_net(model)

        mouth_recon = model(video)
        mouth_gt = crop_mouth(video, mouth_roi)
        metrics = compute_reconstruction_metrics(
            mouth_recon.float(),
            mouth_gt.float(),
            mse_weight=mse_weight,
            temporal_weight=temporal_weight,
            edge_weight=edge_weight,
        )

        for key in totals:
            totals[key] += float(metrics[key].detach().cpu())
        processed_batches += 1
        pbar.set_postfix(
            loss=f"{float(metrics['loss']):.5f}",
            l1=f"{float(metrics['l1']):.5f}",
            motion=f"{float(metrics['motion_ratio']):.3f}",
        )

        # Save comparative plot for the first validation sample of each epoch
        if not plotted and plot_dir is not None:
            plot_path = plot_dir / f"reconstruction_epoch_{epoch:04d}.png"
            save_reconstruction_plot(video[0], mouth_gt[0], mouth_recon[0], plot_path, epoch, mouth_roi)
            plotted = True

        if is_snn:
            reset_snn_net(model)

    return {key: value / max(1, processed_batches) for key, value in totals.items()}


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] Using device: {device}")

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "ssl_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data path
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    print(f"[data] Data directory resolved to: {safe_text(data_path)}")
    mouth_roi = parse_mouth_roi(args.mouth_roi)
    roi_h = mouth_roi[1] - mouth_roi[0]
    roi_w = mouth_roi[3] - mouth_roi[2]
    print(f"[mouth-roi] y={mouth_roi[0]}:{mouth_roi[1]} x={mouth_roi[2]}:{mouth_roi[3]} size={roi_h}x{roi_w}")

    # 1. Dataset Loading
    # Disable landmarks and targets to save RAM and time
    print("[data] Loading VNLipDatasetV2...")
    dataset = VNLipDatasetV2(
        data_dir=str(data_path),
        max_frames=args.max_frames,
        random_crop=True,
        use_landmarks=False,  # VERY IMPORTANT: Disabling landmarks saves 90% of preprocessing
        return_path=False,
        enable_fallback=not args.disable_fallback,
    )

    if args.limit:
        limit = max(1, min(int(args.limit), len(dataset)))
        print(f"[data] Limiting dataset to first {limit} samples.")
        split_dataset = Subset(dataset, list(range(limit)))
    else:
        split_dataset = dataset

    total = len(split_dataset)
    val_count = max(1, int(round(total * args.val_ratio))) if total > 1 else 0
    train_count = total - val_count
    
    generator = torch.Generator().manual_seed(args.seed)
    if val_count > 0:
        train_set, val_set = random_split(split_dataset, [train_count, val_count], generator=generator)
    else:
        train_set, val_set = split_dataset, None

    print(f"[data] Splits: train={len(train_set)}, val={len(val_set) if val_set else 0}")

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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_pad_v2,
        )

    # 2. Build Models
    visual_encoder = build_visual_encoder(args.encoder_type).to(device)
    decoder = MouthPatchDecoder(
        z_dim=512,
        out_channels=1,
        out_size=(roi_h, roi_w),
        base_channels=args.decoder_base_channels,
    ).to(device)
    model = SSLAutoencoder(visual_encoder, decoder).to(device)

    is_snn = args.encoder_type.lower() == "snn"
    if is_snn:
        print("[SNN] Parametric LIF / LIF Nodes enabled with Surrogate Gradients (ATan).")

    # Multi-GPU support
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[device] Multi-GPU Active: {torch.cuda.device_count()}x GPUs utilized. Wrapping model!")
        model = torch.nn.DataParallel(model)

    # 3. Optimization Setup
    encoder_lr = args.lr if args.lr is not None else args.encoder_lr
    decoder_lr = args.lr if args.lr is not None else args.decoder_lr
    optim_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(
        [
            {"params": optim_model.visual_encoder.parameters(), "lr": encoder_lr},
            {"params": optim_model.decoder.parameters(), "lr": decoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    # 4. Training Loop
    best_val_loss = float("inf")
    print(f"\n--- Starting SSL Mouth-Patch Pre-training for {args.encoder_type.upper()} ({args.epochs} epochs) ---")
    print(f"[lr] encoder_lr={encoder_lr:.2e} decoder_lr={decoder_lr:.2e}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            accum_steps=args.accum_steps,
            is_snn=is_snn,
            amp=args.amp,
            mouth_roi=mouth_roi,
            mse_weight=args.mse_weight,
            temporal_weight=args.temporal_weight,
            edge_weight=args.edge_weight,
            max_grad_norm=args.max_grad_norm,
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                is_snn=is_snn,
                epoch=epoch,
                plot_dir=plot_dir,
                mouth_roi=mouth_roi,
                mse_weight=args.mse_weight,
                temporal_weight=args.temporal_weight,
                edge_weight=args.edge_weight,
            )
        elif args.plot_train_every > 0 and (epoch % args.plot_train_every == 0 or epoch == 1 or epoch == args.epochs):
            save_train_reconstruction_sample(
                model=model,
                loader=train_loader,
                device=device,
                is_snn=is_snn,
                epoch=epoch,
                plot_dir=plot_dir,
                mouth_roi=mouth_roi,
            )

        scheduler.step()
        elapsed = time.time() - epoch_start
        
        train_loss_value = train_loss["loss"]
        val_loss_value = val_metrics["loss"] if val_metrics is not None else None
        val_text = f"{val_loss_value:.6f}" if val_loss_value is not None else "n/a"
        print(
            f"[epoch {epoch:03d}] train={train_loss_value:.6f} val={val_text} "
            f"train_l1={train_loss['l1']:.5f} train_motion={train_loss['motion_ratio']:.3f} "
            f"| time={elapsed:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if val_metrics is not None:
            print(
                f"  [val-metrics] l1={val_metrics['l1']:.5f} temporal={val_metrics['temporal']:.5f} "
                f"edge={val_metrics['edge']:.5f} motion_ratio={val_metrics['motion_ratio']:.3f}"
            )

        # Save checkpoints
        score = val_loss_value if val_loss_value is not None else train_loss_value
        is_best = score < best_val_loss
        if is_best:
            best_val_loss = score
            
            unwrapped_model = model.module if hasattr(model, "module") else model
            visual_state = unwrapped_model.visual_encoder.state_dict()
            backbone_state = None
            if hasattr(unwrapped_model.visual_encoder, "backbone"):
                backbone_state = unwrapped_model.visual_encoder.backbone.state_dict()
            
            checkpoint_name = f"pretrain_visual_encoder_{args.encoder_type.lower()}_ssl_mouth_patch.pth"
            checkpoint_path = output_dir / checkpoint_name
            torch.save({
                "epoch": epoch,
                "visual_encoder": visual_state,
                "backbone": backbone_state,
                "mouth_decoder": unwrapped_model.decoder.state_dict(),
                "autoencoder_state_dict": unwrapped_model.state_dict(),
                "val_loss": val_loss_value,
                "train_loss": train_loss_value,
                "train_metrics": train_loss,
                "val_metrics": val_metrics,
                "mouth_roi": mouth_roi,
                "config": vars(args),
            }, checkpoint_path)
            print(f"  -> Best mouth-patch model checkpoint saved to: {safe_text(checkpoint_path)}")

        # Clean cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[done] SSL Mouth-Patch Pre-training successfully completed! Best loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Supervised Pre-training for Silent Video Encoder (Aligned Mouth Patch).")
    parser.add_argument("--data-dir", default=default_data_dir(), help="Data directory containing .pt files.")
    parser.add_argument("--output-dir", default="checkpoints_modular", help="Checkpoint output directory.")
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "nonsnn", "snn"], help="Backbone encoder to pretrain.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=20, help="Number of SSL epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per forward pass.")
    parser.add_argument("--accum-steps", type=int, default=1, help="Steps for gradient accumulation.")
    parser.add_argument("--lr", type=float, default=None, help="Optional shared learning rate override.")
    parser.add_argument("--encoder-lr", type=float, default=1e-4, help="Learning rate for the visual encoder.")
    parser.add_argument("--decoder-lr", type=float, default=5e-4, help="Learning rate for the SSL mouth decoder.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm. Use 0 to disable.")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum video frames per sample.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for dry-run.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mouth-roi", nargs=4, type=int, default=[45, 80, 32, 80], metavar=("Y1", "Y2", "X1", "X2"))
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--temporal-weight", type=float, default=10.0)
    parser.add_argument("--edge-weight", type=float, default=0.5)
    parser.add_argument("--decoder-base-channels", type=int, default=192)
    parser.add_argument("--plot-train-every", type=int, default=1, help="When no val split exists, save a train reconstruction plot every N epochs.")
    parser.add_argument("--disable-fallback", default=True, action=argparse.BooleanOptionalAction, help="Disable dataset fallback replacement during aligned SSL pretrain.")

    main(parser.parse_args())
