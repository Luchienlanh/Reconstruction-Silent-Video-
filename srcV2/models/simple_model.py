from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def norm3d(channels: int) -> nn.GroupNorm:
    groups = 16 if channels % 16 == 0 else 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)

class Conv2Plus1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        mid = max(out_ch, (in_ch * out_ch * 27) // max(1, in_ch * 9 + out_ch * 3))
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False),
            norm3d(mid),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class R2Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(Conv2Plus1D(in_ch, out_ch, stride), norm3d(out_ch), nn.SiLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2Plus1D(out_ch, out_ch), norm3d(out_ch))
        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=(1, stride, stride), bias=False),
                norm3d(out_ch),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        return self.act(self.conv2(self.conv1(x)) + residual)

class SimpleVisualTower(nn.Module):
    """Standard ResNet2+1D spatial encoder with configurable spatial pooling grid."""
    def __init__(self, dim: int = 512, spatial_tokens: int = 4):
        super().__init__()
        self.spatial_tokens = int(spatial_tokens)
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            norm3d(64),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
            norm3d(64),
            nn.SiLU(inplace=True),
        )
        self.layers = nn.Sequential(
            R2Block(64, 64, stride=1),
            R2Block(64, 128, stride=2),
            R2Block(128, 256, stride=2),
            R2Block(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d((None, self.spatial_tokens, self.spatial_tokens))
        self.proj = nn.Sequential(
            nn.Linear(512 * self.spatial_tokens * self.spatial_tokens, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, 1, T, H, W)
        x = self.pool(self.layers(self.stem(video.float()))) # (B, 512, T, S, S)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t, h * w * c) # Flatten spatial features
        return self.proj(x) # (B, T, dim)

class SimpleLandmarkTower(nn.Module):
    """Standard MLP for Landmark extraction with dynamics."""
    def __init__(self, num_points: int = 40, dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_points = int(num_points)
        self.input_dim = self.num_points * 6  # x, y, dx, dy, d2x, d2y
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        d = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)

    def _normalize(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = torch.nan_to_num(landmarks.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xy = landmarks[..., :2]
        center = xy.mean(dim=2, keepdim=True)
        xy_c = xy - center
        scale = xy_c.pow(2).sum(dim=-1).sqrt().amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
        if landmarks.shape[-1] >= 6:
            d1 = landmarks[..., 2:4] / scale
            d2 = landmarks[..., 4:6] / scale
        else:
            d1 = self._delta(xy_c / scale)
            d2 = self._delta(d1)
        return torch.cat([xy_c / scale, d1, d2], dim=-1)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        # landmarks: (B, T, N, 6)
        x = self._normalize(landmarks)
        x = x.flatten(start_dim=2) # (B, T, N * 6)
        return self.net(x) # (B, T, dim)

class SimpleLipToSpeechModel(nn.Module):
    """
    Simplified End-to-End model:
    ResNet2+1D + Landmark MLP -> Concat Fusion -> ConvTranspose upsampling -> GRU Decoder.
    Extremely stable, easy to converge, no SIREN or complex Fourier scaling needed.
    """
    def __init__(self, dim: int = 512, spatial_tokens: int = 4, num_points: int = 40, dropout: float = 0.0):
        super().__init__()
        self.visual = SimpleVisualTower(dim=dim, spatial_tokens=spatial_tokens)
        self.landmarks = SimpleLandmarkTower(num_points=num_points, dim=dim // 2, dropout=dropout)
        
        # Concat Fusion Layer (Visual (dim) + Landmark (dim // 2) -> Fused (dim))
        self.fusion = nn.Sequential(
            nn.Linear(dim + dim // 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Learned Upsampler from video rate (25 fps) to mel rate (62.5 fps) (ratio 2.5x)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1), # ~2x upsample
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        )
        
        # Recurrent Decoder (GRU)
        self.decoder_gru = nn.GRU(
            input_size=dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0
        )
        
        # Output project head (GRU output is 512-dim because of bidirectional)
        self.mel_head = nn.Linear(512, 80)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        video = batch["video"]
        landmarks = batch["landmarks"]
        target_mel = batch.get("mel")
        mel_mask = batch.get("mel_mask")

        # 1. Feature Extraction
        z_vis = self.visual(video) # (B, T, dim)
        z_lm = self.landmarks(landmarks) # (B, T, dim // 2)

        # 2. Concat Fusion
        z = torch.cat([z_vis, z_lm], dim=-1) # (B, T, dim + dim // 2)
        z = self.fusion(z) # (B, T, dim)

        # 3. Learned Upsampling
        x = z.transpose(1, 2) # (B, dim, T)
        x = self.upsample(x) # (B, dim, T_upsampled)
        
        # Target length matching (mel_mask has the exact padded target length)
        if mel_mask is not None:
            target_len = mel_mask.shape[1]
        elif target_mel is not None:
            target_len = target_mel.shape[1]
        else:
            target_len = int(round(z.shape[1] * 2.5))
            
        if x.shape[2] != target_len:
            x = F.interpolate(
                x,
                size=int(target_len),
                mode="linear",
                align_corners=False,
            )
        z_up = x.transpose(1, 2).contiguous() # (B, T_mel, dim)

        # 4. Recurrent decoding (GRU)
        g_out, _ = self.decoder_gru(z_up) # (B, T_mel, 512)
        
        # 5. Output Mel-spectrogram
        pred_mel = self.mel_head(g_out) # (B, T_mel, 80)
        
        # Apply mel mask if provided
        if mel_mask is not None:
            pred_mel = pred_mel * mel_mask.unsqueeze(-1).to(pred_mel.dtype)
            
        return pred_mel
