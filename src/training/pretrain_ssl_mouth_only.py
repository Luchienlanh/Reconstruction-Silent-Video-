"""
pretrain_byol_mouth.py
======================
Bootstrap Your Own Latent (BYOL) pretraining for mouth video encoder (ResNet2+1D).
No labels, no negative pairs, only relies on augmentations of the same clip.
"""

import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Adjust path to import your encoder builder
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your dataset and encoder builder
# If you don't have these modules, replace with direct dataset loading
try:
    from src.data.dataset import VNLipDatasetV2, collate_pad_v2
    from src.models.encoders.factory import build_encoder
    USE_FACTORY = True
except ImportError:
    USE_FACTORY = False
    print("Warning: Could not import from src modules. Using fallback dataset and encoder.")

# ========== Fallback dataset if src not available ==========
class SimpleMouthDataset(Dataset):
    """Fallback dataset: load .pt files, crop mouth ROI, return video tensor."""
    def __init__(self, data_dir, mouth_roi=(45,80,32,80), max_frames=30):
        self.data_dir = Path(data_dir)
        self.mouth_roi = mouth_roi
        self.max_frames = max_frames
        self.files = list(self.data_dir.glob("*.pt"))
        print(f"Loaded {len(self.files)} .pt files from {data_dir}")
        if not self.files:
            raise RuntimeError(f"No .pt files in {data_dir}")

    def _load_video(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        video = data['video'].float()
        # video shape: (C, T, H, W) where C=1, H=W=112
        if video.dim() == 3:
            video = video.unsqueeze(0)
        # Crop mouth ROI
        y1, y2, x1, x2 = self.mouth_roi
        mouth = video[:, :, :, y1:y2, x1:x2]  # (1, T, H_roi, W_roi)
        T = mouth.shape[1]
        if T > self.max_frames:
            start = random.randint(0, T - self.max_frames)
            mouth = mouth[:, start:start+self.max_frames]
        elif T < self.max_frames:
            pad = self.max_frames - T
            mouth = F.pad(mouth, (0,0,0,0,0,pad,0,0))
        return mouth

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        return self._load_video(path)  # (1, max_frames, H_roi, W_roi)

def collate_byol(batch):
    # batch: list of (mouth_video) each shape (1, T, H, W)
    # Stack to (B, 1, T, H, W)
    return torch.stack(batch, dim=0)


# ========== Augmentations for mouth videos ==========
class RandomBrightness(nn.Module):
    def __init__(self, strength=0.1):
        super().__init__()
        self.strength = strength
    def forward(self, x):
        if random.random() < 0.5:
            return x + torch.randn(1, device=x.device) * self.strength
        return x

class RandomContrast(nn.Module):
    def __init__(self, min_factor=0.7, max_factor=1.3):
        super().__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
    def forward(self, x):
        if random.random() < 0.5:
            factor = random.uniform(self.min_factor, self.max_factor)
            return x * factor
        return x

class RandomTimeMask(nn.Module):
    def __init__(self, max_mask_frames=4):
        super().__init__()
        self.max_mask = max_mask_frames
    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        if random.random() < 0.5:
            mask_len = random.randint(1, min(self.max_mask, T//2))
            start = random.randint(0, T - mask_len)
            x = x.clone()
            x[:, :, start:start+mask_len] = 0.0
        return x

def get_byol_augmentation():
    """Augmentation pipeline for BYOL: not too aggressive to preserve lip motion."""
    return nn.Sequential(
        RandomBrightness(strength=0.05),
        RandomContrast(0.8, 1.2),
        RandomTimeMask(max_mask_frames=4),
        # Optional: small Gaussian blur
    )


# ========== BYOL Components ==========
class BYOLProjector(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # no affine for target
        )
    def forward(self, x):
        return self.net(x)

class BYOLPredictor(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, encoder, projection_dim=256, hidden_dim=4096, momentum=0.996):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = self._copy_encoder(encoder)
        self.online_projector = BYOLProjector(512, hidden_dim, projection_dim)
        self.target_projector = BYOLProjector(512, hidden_dim, projection_dim)
        self.predictor = BYOLPredictor(projection_dim, hidden_dim, projection_dim)
        self.momentum = momentum

        # Stop gradients for target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self._update_target(keep_online=True)  # init target = online

    def _copy_encoder(self, encoder):
        """Deep copy encoder without shared parameters."""
        import copy
        copy_enc = copy.deepcopy(encoder)
        return copy_enc

    @torch.no_grad()
    def _update_target(self, keep_online=False):
        """Momentum update for target network."""
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1 - self.momentum)
        for param_online, param_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1 - self.momentum)
        if keep_online:
            for param_online in self.online_encoder.parameters():
                param_online.requires_grad = True

    def forward(self, x1, x2):
        # x1, x2: (B, 1, T, H, W) two augmented views of the same clip
        # Encode and temporal pool
        z1 = self.online_encoder(x1)   # (B, T, 512)
        z2 = self.online_encoder(x2)
        z1 = z1.mean(dim=1)             # (B, 512)
        z2 = z2.mean(dim=1)

        # Project
        p1 = self.online_projector(z1)
        p2 = self.online_projector(z2)
        # Predict
        q1 = self.predictor(p1)
        q2 = self.predictor(p2)

        with torch.no_grad():
            target_z1 = self.target_encoder(x1).mean(dim=1)
            target_z2 = self.target_encoder(x2).mean(dim=1)
            target_p1 = self.target_projector(target_z1)
            target_p2 = self.target_projector(target_z2)

        # Symmetric loss: q1 with target_p2, q2 with target_p1
        loss = 2 - (F.cosine_similarity(q1, target_p2, dim=-1).mean() +
                    F.cosine_similarity(q2, target_p1, dim=-1).mean())
        return loss

    @torch.no_grad()
    def update_target(self):
        self._update_target()


# ========== Training ==========
def train_byol(model, dataloader, optimizer, device, epochs=50,
               max_grad_norm=1.0, checkpoint_dir=None, save_every=5):
    model.train()
    aug = get_byol_augmentation().to(device)
    best_loss = float('inf')

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, video in enumerate(pbar):
            video = video.to(device, non_blocking=True)  # (B, 1, T, H_roi, W_roi)
            # Create two augmented views
            v1 = aug(video)
            v2 = aug(video)

            loss = model(v1, v2)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            model.update_target()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        if checkpoint_dir and (epoch % save_every == 0 or epoch == epochs):
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'online_encoder_state_dict': model.online_encoder.state_dict(),
                'loss': avg_loss,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"byol_epoch_{epoch}.pth"))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, os.path.join(checkpoint_dir, "byol_best.pth"))
                print(f"  -> Best model saved (loss={best_loss:.6f})")

    return model.online_encoder


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="BYOL pretraining for mouth video encoder")
    parser.add_argument("--data-dir", type=str, default="Processed_Data_Mel_HiFiGAN",
                        help="Directory containing .pt files (video frames).")
    parser.add_argument("--output-dir", type=str, default="checkpoints_byol",
                        help="Directory to save checkpoints.")
    parser.add_argument("--encoder-type", type=str, default="non_snn",
                        choices=["non_snn", "snn", "resnet18_temporal"],
                        help="Backbone encoder type (must be (2+1)D).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Number of frames per clip.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.996,
                        help="Momentum for target network update.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mouth-roi", nargs=4, type=int, default=[45,80,32,80],
                        metavar=("Y1","Y2","X1","X2"), help="Mouth crop coordinates in 112x112 frame.")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Dataset
    if USE_FACTORY:
        # Use VNLipDatasetV2 but we only need video, no landmarks/target
        # Note: set use_landmarks=False, random_crop=False to avoid unnecessary data
        dataset = VNLipDatasetV2(
            data_dir=args.data_dir,
            max_frames=args.max_frames,
            random_crop=True,
            use_landmarks=False,
            return_path=False,
            enable_fallback=True,
        )
        # Override __getitem__ to return only mouth crop? Actually VNLipDatasetV2 returns (video, landmarks, target)
        # We'll create a wrapper to crop mouth and ignore other outputs
        class Wrapper(Dataset):
            def __init__(self, ds, roi):
                self.ds = ds
                self.roi = roi
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                video, _, _ = self.ds[idx]  # video: (C, T, H, W)
                y1,y2,x1,x2 = self.roi
                mouth = video[:, :, :, y1:y2, x1:x2]
                return mouth
        dataset = Wrapper(dataset, tuple(args.mouth_roi))
        collate_fn = collate_pad_v2  # this collate expects (video, landmarks, target) but we only have video? Needs adjustment.
        # Simpler: use fallback dataset that only loads video and crops.
        print("Falling back to SimpleMouthDataset because collate mismatch.")
        dataset = SimpleMouthDataset(args.data_dir, tuple(args.mouth_roi), args.max_frames)
        collate_fn = collate_byol
    else:
        dataset = SimpleMouthDataset(args.data_dir, tuple(args.mouth_roi), args.max_frames)
        collate_fn = collate_byol

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    print(f"Dataset size: {len(dataset)}")

    # 2. Build base encoder (ResNet2+1D)
    if USE_FACTORY:
        base_encoder = build_encoder(args.encoder_type).to(device)
    else:
        # If build_encoder not available, you need to define your (2+1)D encoder here.
        # For now, raise error.
        raise ImportError("build_encoder not available. Please ensure src.models.encoders.factory exists or manually define your ResNet2+1D.")
    print(f"Base encoder built: {args.encoder_type}")

    # 3. BYOL model
    byol = BYOL(base_encoder, momentum=args.momentum).to(device)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(byol.parameters(), lr=args.lr, weight_decay=1e-4)

    # 5. Train
    pretrained_encoder = train_byol(
        model=byol,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir,
    )

    # 6. Save final encoder weights only (for downstream mel reconstruction)
    final_encoder_path = os.path.join(args.output_dir, "pretrained_encoder_resnet2plus1d.pth")
    torch.save(pretrained_encoder.state_dict(), final_encoder_path)
    print(f"Pretrained encoder saved to {final_encoder_path}")
    print("BYOL pretraining completed.")


if __name__ == "__main__":
    main()