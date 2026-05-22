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
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# Add project paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import existing dataset and encoder factory
try:
    from src.data.dataset import VNLipDatasetV2
    from src.models.encoders.factory import build_encoder
    USE_FACTORY = True
except ImportError:
    USE_FACTORY = False
    print("[WARN] Could not import src modules. Using fallback dataset and encoder.")


# ========== Fallback dataset ==========
class SimpleMouthDataset(Dataset):
    """Load .pt files, crop mouth ROI, return video tensor (1, T, H_roi, W_roi)."""
    def __init__(self, data_dir, mouth_roi=(45,80,32,80), max_frames=30):
        self.data_dir = Path(data_dir)
        self.mouth_roi = mouth_roi
        self.max_frames = max_frames
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found in {data_dir}")
        print(f"[data] Found {len(self.files)} .pt files.")

    def _load_video(self, path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        video = data['video'].float()
        if video.dim() == 3:
            video = video.unsqueeze(0)  # (1, T, H, W)
        y1, y2, x1, x2 = self.mouth_roi
        mouth = video[:, :, y1:y2, x1:x2]   # (1, T, H_roi, W_roi)
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
        return self._load_video(self.files[idx])

def collate_byol(batch):
    """batch: list of (1, T, H, W) -> (B, 1, T, H, W)"""
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
        B, C, T, H, W = x.shape
        if random.random() < 0.5 and T > self.max_mask:
            mask_len = random.randint(1, min(self.max_mask, T//2))
            start = random.randint(0, T - mask_len)
            x = x.clone()
            x[:, :, start:start+mask_len] = 0.0
        return x

def get_byol_augmentation():
    return nn.Sequential(
        RandomBrightness(strength=0.05),
        RandomContrast(0.8, 1.2),
        RandomTimeMask(max_mask_frames=4),
    )


# ========== BYOL components (using LayerNorm to support batch_size=1) ==========
class BYOLProjector(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)

class BYOLPredictor(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, encoder, projection_dim=256, hidden_dim=4096, momentum=0.996):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.online_projector = BYOLProjector(512, hidden_dim, projection_dim)
        self.target_projector = BYOLProjector(512, hidden_dim, projection_dim)
        self.predictor = BYOLPredictor(projection_dim, hidden_dim, projection_dim)
        self.momentum = momentum

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self._update_target(keep_online=True)

    @torch.no_grad()
    def _update_target(self, keep_online=False):
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1 - self.momentum)
        for param_online, param_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1 - self.momentum)
        if keep_online:
            for param_online in self.online_encoder.parameters():
                param_online.requires_grad = True

    def forward(self, x1, x2):
        # x1, x2: (B, 1, T, H, W)
        z1 = self.online_encoder(x1).mean(dim=1)   # (B, 512)
        z2 = self.online_encoder(x2).mean(dim=1)
        p1 = self.online_projector(z1)
        p2 = self.online_projector(z2)
        q1 = self.predictor(p1)
        q2 = self.predictor(p2)

        with torch.no_grad():
            t1 = self.target_encoder(x1).mean(dim=1)
            t2 = self.target_encoder(x2).mean(dim=1)
            tp1 = self.target_projector(t1)
            tp2 = self.target_projector(t2)

        loss = 2 - (F.cosine_similarity(q1, tp2, dim=-1).mean() +
                    F.cosine_similarity(q2, tp1, dim=-1).mean())
        return loss

    @torch.no_grad()
    def update_target(self):
        self._update_target()


# ========== Training loop ==========
def train_byol(model, dataloader, optimizer, device, epochs, max_grad_norm=1.0,
               checkpoint_dir=None, save_every=5, amp=False):
    model.train()
    aug = get_byol_augmentation().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.type=='cuda')
    best_loss = float('inf')

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for video in pbar:
            video = video.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp and device.type=='cuda'):
                v1 = aug(video)
                v2 = aug(video)
                loss = model(v1, v2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model.update_target()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        if checkpoint_dir and (epoch % save_every == 0 or epoch == epochs):
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = {
                'epoch': epoch,
                'online_encoder_state_dict': model.online_encoder.state_dict(),
                'loss': avg_loss,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, f"byol_epoch_{epoch}.pth"))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(checkpoint_dir, "byol_best.pth"))

    return model.online_encoder


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="BYOL pretraining for mouth video encoder")
    parser.add_argument("--data-dir", type=str, default="Processed_Data_Mel_HiFiGAN",
                        help="Directory containing .pt files")
    parser.add_argument("--output-dir", type=str, default="checkpoints_byol",
                        help="Checkpoint output directory")
    parser.add_argument("--encoder-type", type=str, default="non_snn",
                        choices=["non_snn", "snn", "resnet18_temporal"],
                        help="Backbone encoder type")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--mouth-roi", nargs=4, type=int, default=[45,80,32,80],
                        metavar=("Y1","Y2","X1","X2"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Mixed precision: {args.amp and device.type=='cuda'}")

    # 1. Dataset
    if USE_FACTORY:
        base_ds = VNLipDatasetV2(
            data_dir=args.data_dir,
            max_frames=args.max_frames,
            random_crop=True,
            use_landmarks=False,
            return_path=False,
            enable_fallback=True,
        )
        class MouthWrapper(Dataset):
            def __init__(self, ds, roi):
                self.ds = ds
                self.roi = roi
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                result = self.ds[idx]
                if isinstance(result, tuple):
                    video = result[0]
                else:
                    video = result
                y1,y2,x1,x2 = self.roi
                mouth = video[:, :, y1:y2, x1:x2]
                return mouth
        dataset = MouthWrapper(base_ds, tuple(args.mouth_roi))
        collate_fn = lambda batch: torch.stack(batch, dim=0)
    else:
        dataset = SimpleMouthDataset(args.data_dir, tuple(args.mouth_roi), args.max_frames)
        collate_fn = collate_byol

    if args.limit:
        limit = max(1, min(int(args.limit), len(dataset)))
        dataset = Subset(dataset, list(range(limit)))
        print(f"[data] Limited dataset to {limit} samples.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    print(f"Dataset size: {len(dataset)}")

    # 2. Build base encoder
    if USE_FACTORY:
        base_encoder = build_encoder(args.encoder_type).to(device)
    else:
        raise ImportError("build_encoder not found. Please implement your ResNet2+1D encoder.")
    print(f"Base encoder: {args.encoder_type}")

    # 3. BYOL model
    byol = BYOL(base_encoder, momentum=args.momentum).to(device)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(byol.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. Train
    pretrained_encoder = train_byol(
        model=byol,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.output_dir,
        save_every=max(1, args.epochs//10),
        amp=args.amp,
    )

    # 6. Save final encoder only
    final_path = os.path.join(args.output_dir, "pretrained_encoder_resnet2plus1d.pth")
    torch.save(pretrained_encoder.state_dict(), final_path)
    print(f"Pretrained encoder saved to {final_path}")
    print("BYOL pretraining completed.")


if __name__ == "__main__":
    main()