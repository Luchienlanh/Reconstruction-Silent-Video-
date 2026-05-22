"""
pretrain_ssl.py
===============
Self-Supervised Pre-training (SSL) script for lip-reading visual feature extractors.
Supports pre-training both:
- NonSNN backbone (NonSpikingVidResNet)
- SNN backbone (VidResNet)
Uses a Hybrid 2D/3D Convolutional Decoder to reconstruct raw grayscale video frames.
Saves comparative visualization plots (original vs reconstructed) for visual validation.
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


class VideoDecoder3D(nn.Module):
    """
    Hybrid 2D/3D Video Decoder.
    Takes a continuous sequence representation of shape (B, T, 512) and reconstructs
    the original grayscale lip video of shape (B, 1, T, 112, 112).
    """
    def __init__(self, z_dim: int = 512, out_channels: int = 1):
        super().__init__()
        # Spatial progressive upsampling using 2D Transposed Convolutions
        self.spatial_upsample = nn.Sequential(
            # (B * T, 512, 1, 1) -> (B * T, 256, 7, 7)
            nn.ConvTranspose2d(z_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            # (B * T, 256, 7, 7) -> (B * T, 128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # (B * T, 128, 14, 14) -> (B * T, 64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # (B * T, 64, 28, 28) -> (B * T, 32, 56, 56)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            # (B * T, 32, 56, 56) -> (B * T, 16, 112, 112)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        
        # Temporal smoothing 3D Convolution layer with spatial-temporal integration
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(16, out_channels, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1)),
            nn.Sigmoid()  # Normalize output pixels to [0.0, 1.0]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (B, T, 512)
        B, T, C = z.shape
        
        # Flatten temporal and batch dimensions to treat as 2D frames spatially
        x = z.reshape(B * T, C).unsqueeze(-1).unsqueeze(-1)  # (B * T, 512, 1, 1)
        
        # Upsample spatially to (B * T, 16, 112, 112)
        x = self.spatial_upsample(x)
        
        # Reshape back to 3D video tensor: (B, 16, T, 112, 112)
        x = x.reshape(B, T, 16, 112, 112).permute(0, 2, 1, 3, 4).contiguous()
        
        # Apply 3D Convolution for temporal context and smoothing
        out = self.temporal_conv(x)  # (B, 1, T, 112, 112)
        return out


class SSLAutoencoder(nn.Module):
    """
    Wrapper module to combine visual backbone and VideoDecoder3D.
    """
    def __init__(self, backbone: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)  # (B, T, 512)
        return self.decoder(z)  # (B, 1, T, 112, 112)


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


def build_backbone(encoder_type: str) -> nn.Module:
    encoder_type = encoder_type.lower()
    if encoder_type == "snn":
        if not SNN_AVAILABLE:
            raise ImportError("SpikingJelly is not installed, cannot use SNN backbone.")
        from src.models.encoders.snn import VidResNet
        print("[backbone] Instantiating Spiking VidResNet (SNN)")
        return VidResNet()
    elif encoder_type in {"non_snn", "nonsnn", "cnn_transformer"}:
        from src.models.encoders.non_snn import NonSpikingVidResNet
        print("[backbone] Instantiating NonSpikingVidResNet (Non-SNN)")
        return NonSpikingVidResNet()
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


def save_reconstruction_plot(original: torch.Tensor, reconstructed: torch.Tensor, output_path: Path, epoch: int) -> None:
    """
    Save comparative plot showing original vs reconstructed frames side-by-side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ensure tensors are on CPU and detached
    # shapes: (1, T, 112, 112)
    orig = original.detach().cpu().squeeze(0).numpy()  # (T, 112, 112)
    recon = reconstructed.detach().cpu().squeeze(0).numpy()  # (T, 112, 112)
    T = orig.shape[0]

    # Select up to 5 frames uniformly spaced
    num_frames_to_plot = min(5, T)
    indices = np.linspace(0, T - 1, num_frames_to_plot, dtype=int)

    fig, axes = plt.subplots(2, num_frames_to_plot, figsize=(3 * num_frames_to_plot, 6), constrained_layout=True)
    
    if num_frames_to_plot == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, frame_idx in enumerate(indices):
        # Plot original
        axes[0, col_idx].imshow(orig[frame_idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col_idx].axis("off")
        if col_idx == 0:
            axes[0, col_idx].set_title("Original", loc="left", fontsize=12, fontweight="bold")
        else:
            axes[0, col_idx].set_title(f"Frame {frame_idx}")

        # Plot reconstructed
        axes[1, col_idx].imshow(recon[frame_idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col_idx].axis("off")
        if col_idx == 0:
            axes[1, col_idx].set_title("Reconstructed", loc="left", fontsize=12, fontweight="bold")

    fig.suptitle(f"SSL Pixel Reconstruction (Epoch {epoch})", fontsize=14, fontweight="bold")
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
) -> float:
    model.train()
    total_loss = 0.0
    processed_batches = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  SSL Train")
    for batch_idx, batch in enumerate(pbar):
        # batch is (video_batches, target_batches, target_lengths) since landmarks are disabled
        video, _, _ = batch
        video = video.to(device, non_blocking=True)  # (B, 1, T, 112, 112)

        # Reset SNN states if using SNN
        if is_snn:
            reset_snn_net(model)

        try:
            with torch.amp.autocast("cuda", enabled=amp):
                # Reconstruct video pixels
                recon_video = model(video)  # (B, 1, T, 112, 112)
                
                # Combine L1 Loss and MSE Loss with Mouth-focused ROI Weighting + Temporal Difference Loss
                # 1. Mouth ROI Focused Loss (y: 65 to 100, x: 32 to 80) - Tăng trọng số vùng miệng lên gấp 8 lần
                loss_map_mse = (recon_video - video) ** 2
                loss_map_l1 = torch.abs(recon_video - video)
                
                loss_map_mse[:, :, :, 65:100, 32:80] *= 8.0
                loss_map_l1[:, :, :, 65:100, 32:80] *= 8.0
                
                loss_mse = loss_map_mse.mean()
                loss_l1 = loss_map_l1.mean()
                
                # 2. Temporal Difference Loss (bắt buộc học chuyển động động học)
                diff_gt = video[:, :, 1:] - video[:, :, :-1]
                diff_recon = recon_video[:, :, 1:] - recon_video[:, :, :-1]
                loss_temporal = F.l1_loss(diff_recon, diff_gt)
                
                loss = loss_mse + 0.5 * loss_l1 + 3.0 * loss_temporal
                
                # Normalize loss to account for gradient accumulation
                loss = loss / accum_steps

            # Scale gradients
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().cpu()) * accum_steps
            processed_batches += 1
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()) * accum_steps:.6f}")

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Out of memory error in batch index {batch_idx}. Skipping.")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise

        finally:
            if is_snn:
                reset_snn_net(model)

    return total_loss / max(1, processed_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_snn: bool,
    epoch: int,
    plot_dir: Path,
) -> float:
    model.eval()
    total_loss = 0.0
    processed_batches = 0

    # We will save visual plot for the first sample in the validation set
    plotted = False

    pbar = tqdm(loader, desc="  SSL Val")
    for batch_idx, batch in enumerate(pbar):
        video, _, _ = batch
        video = video.to(device, non_blocking=True)

        if is_snn:
            reset_snn_net(model)

        recon_video = model(video)
        
        # Weighted L1 & MSE + Temporal Diff
        loss_map_mse = (recon_video - video) ** 2
        loss_map_l1 = torch.abs(recon_video - video)
        
        loss_map_mse[:, :, :, 65:100, 32:80] *= 8.0
        loss_map_l1[:, :, :, 65:100, 32:80] *= 8.0
        
        loss_mse = loss_map_mse.mean()
        loss_l1 = loss_map_l1.mean()
        
        diff_gt = video[:, :, 1:] - video[:, :, :-1]
        diff_recon = recon_video[:, :, 1:] - recon_video[:, :, :-1]
        loss_temporal = F.l1_loss(diff_recon, diff_gt)
        
        loss = loss_mse + 0.5 * loss_l1 + 3.0 * loss_temporal

        total_loss += float(loss.detach().cpu())
        processed_batches += 1
        pbar.set_postfix(loss=f"{float(loss):.6f}")

        # Save comparative plot for the first validation sample of each epoch
        if not plotted and plot_dir is not None:
            plot_path = plot_dir / f"reconstruction_epoch_{epoch:04d}.png"
            save_reconstruction_plot(video[0], recon_video[0], plot_path, epoch)
            plotted = True

        if is_snn:
            reset_snn_net(model)

    return total_loss / max(1, processed_batches)


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

    # 1. Dataset Loading
    # Disable landmarks and targets to save RAM and time
    print("[data] Loading VNLipDatasetV2...")
    dataset = VNLipDatasetV2(
        data_dir=str(data_path),
        max_frames=args.max_frames,
        random_crop=True,
        use_landmarks=False,  # VERY IMPORTANT: Disabling landmarks saves 90% of preprocessing
        return_path=False,
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
    backbone = build_backbone(args.encoder_type).to(device)
    decoder = VideoDecoder3D(z_dim=512, out_channels=1).to(device)
    model = SSLAutoencoder(backbone, decoder).to(device)

    is_snn = args.encoder_type.lower() == "snn"
    if is_snn:
        print("[SNN] Parametric LIF / LIF Nodes enabled with Surrogate Gradients (ATan).")

    # Multi-GPU support
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[device] Multi-GPU Active: {torch.cuda.device_count()}x GPUs utilized. Wrapping model!")
        model = torch.nn.DataParallel(model)

    # 3. Optimization Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    # 4. Training Loop
    best_val_loss = float("inf")
    print(f"\n--- Starting SSL Pre-training for {args.encoder_type.upper()} ({args.epochs} epochs) ---")
    
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
        )

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                is_snn=is_snn,
                epoch=epoch,
                plot_dir=plot_dir,
            )

        scheduler.step()
        elapsed = time.time() - epoch_start
        
        val_text = f"{val_loss:.6f}" if val_loss is not None else "n/a"
        print(f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_text} | time={elapsed:.1f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoints
        score = val_loss if val_loss is not None else train_loss
        is_best = score < best_val_loss
        if is_best:
            best_val_loss = score
            
            # Save the backbone weights only for main model loading
            # Handle DataParallel wrapper
            unwrapped_backbone = model.module.backbone if hasattr(model, "module") else model.backbone
            backbone_state = unwrapped_backbone.state_dict()
            
            checkpoint_name = f"pretrain_resnet2plus1d_{args.encoder_type.lower()}_ssl.pth"
            checkpoint_path = output_dir / checkpoint_name
            torch.save({
                "epoch": epoch,
                "backbone": backbone_state,
                "autoencoder_state_dict": model.state_dict() if not hasattr(model, "module") else model.module.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": vars(args),
            }, checkpoint_path)
            print(f"  -> Best model checkpoint saved to: {safe_text(checkpoint_path)}")

        # Clean cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[done] SSL Pre-training successfully completed! Best loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Supervised Pre-training for Silent Video Encoder.")
    parser.add_argument("--data-dir", default=default_data_dir(), help="Data directory containing .pt files.")
    parser.add_argument("--output-dir", default="checkpoints_modular", help="Checkpoint output directory.")
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "nonsnn", "snn"], help="Backbone encoder to pretrain.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=20, help="Number of SSL epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per forward pass.")
    parser.add_argument("--accum-steps", type=int, default=1, help="Steps for gradient accumulation.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum video frames per sample.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for dry-run.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision.")
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
