import os
import gc
import glob
import math
import random
import warnings
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from .snn import SpikingViTEncoder
from .non_snn import NonSpikingViTEncoder, ResNet18TemporalEncoder

def build_encoder(encoder_type="snn", **kwargs):
    encoder_type = encoder_type.lower()
    if encoder_type == "snn":
        return SpikingViTEncoder(**kwargs)
    if encoder_type in {"non_snn", "nonsnn", "cnn_transformer"}:
        return NonSpikingViTEncoder(**kwargs)
    if encoder_type == "resnet18_temporal":
        return ResNet18TemporalEncoder(**kwargs)
    raise ValueError(f"Unknown ENCODER_TYPE: {encoder_type}")

class LandmarkEncoder(nn.Module):
    """Encode lip/face landmark positions and frame-to-frame velocity at video-frame rate."""
    def __init__(self, num_points, hidden_dim=256, out_dim=512, dropout=0.1):
        super().__init__()
        self.num_points = num_points
        in_dim = num_points * 4  # x, y, dx, dy
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _normalize(self, landmarks):
        landmarks = torch.nan_to_num(landmarks.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xy = landmarks[..., :2]
        center = xy.mean(dim=2, keepdim=True)
        xy = xy - center
        scale = xy.pow(2).sum(dim=-1).sqrt().amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
        return xy / scale

    def forward(self, landmarks):
        # landmarks: (B, T_video, N, 2)
        xy = self._normalize(landmarks)
        delta = xy[:, 1:] - xy[:, :-1]
        delta = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1)
        x = torch.cat([xy, delta], dim=-1).flatten(start_dim=2)
        return self.net(x)

class VisualLandmarkEncoder(nn.Module):
    """Fuse the visual encoder with landmark dynamics before the mel decoder."""
    def __init__(self, visual_encoder, num_landmark_points, z_dim=512):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.landmark_encoder = LandmarkEncoder(num_landmark_points, out_dim=z_dim)
        self.fusion = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, video, landmarks=None):
        if landmarks is None:
            raise ValueError("VisualLandmarkEncoder requires landmarks. Set USE_LANDMARKS=False for visual-only training.")
        z_video = self.visual_encoder(video)
        z_landmark = self.landmark_encoder(landmarks)
        if z_landmark.shape[1] != z_video.shape[1]:
            z_landmark = F.interpolate(
                z_landmark.transpose(1, 2),
                size=z_video.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
        return self.fusion(torch.cat([z_video, z_landmark], dim=-1))

class LandmarkEncoderV2(nn.Module):
    """
    Encode lip/face landmarks with position + velocity + acceleration.
    Input: (B, T_video, N, 6) where last dim = [x, y, dx, dy, d2x, d2y]
    Output: (B, T_video, out_dim)
    """

    def __init__(self, num_points: int, hidden_dim: int = 256, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.num_points = num_points
        in_dim = num_points * 6  # x, y, dx, dy, d2x, d2y
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _normalize(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Normalize positions (channels 0-1) by centering and scaling.
        Velocity (2-3) and acceleration (4-5) are scaled by the same factor.
        """
        landmarks = torch.nan_to_num(landmarks.float(), nan=0.0, posinf=0.0, neginf=0.0)

        xy = landmarks[..., :2]
        center = xy.mean(dim=2, keepdim=True)   # (B, T, 1, 2)
        xy_centered = xy - center

        # Scale by max distance from center
        scale = (
            xy_centered.pow(2).sum(dim=-1).sqrt()  # (B, T, N)
            .amax(dim=2, keepdim=True)              # (B, T, 1)
            .unsqueeze(-1)                           # (B, T, 1, 1)
            .clamp_min(1e-4)
        )

        xy_norm = xy_centered / scale

        # Scale velocity and acceleration by the same spatial scale
        d1 = landmarks[..., 2:4] / scale
        d2 = landmarks[..., 4:6] / scale

        return torch.cat([xy_norm, d1, d2], dim=-1)  # (B, T, N, 6)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """landmarks: (B, T_video, N, 6) -> (B, T_video, out_dim)"""
        x = self._normalize(landmarks)
        x = x.flatten(start_dim=2)  # (B, T, N*6)
        return self.net(x)

class VisualLandmarkEncoderV2(nn.Module):
    """Fuse the visual encoder with 6-dim landmark dynamics before the mel decoder using Multi-Head Cross-Attention."""

    def __init__(self, visual_encoder: nn.Module, num_landmark_points: int, z_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.landmark_encoder = LandmarkEncoderV2(num_landmark_points, out_dim=z_dim)
        
        # Multi-Head Cross-Attention where Query is Video, Key and Value are Landmark features.
        self.cross_attn = nn.MultiheadAttention(embed_dim=z_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization and residual paths
        self.norm1 = nn.LayerNorm(z_dim)
        self.norm2 = nn.LayerNorm(z_dim)
        
        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(z_dim * 2, z_dim),
            nn.Dropout(dropout),
        )

    def forward(self, video: torch.Tensor, landmarks: torch.Tensor = None) -> torch.Tensor:
        if landmarks is None:
            raise ValueError("VisualLandmarkEncoderV2 requires landmarks (B, T, N, 6).")
        
        z_video = self.visual_encoder(video)       # (B, T_video, z_dim)
        z_landmark = self.landmark_encoder(landmarks)  # (B, T_landmark, z_dim)
        
        # Align temporal dimensions if needed
        if z_landmark.shape[1] != z_video.shape[1]:
            z_landmark = F.interpolate(
                z_landmark.transpose(1, 2),
                size=z_video.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
            
        # Cross-Attention: Query = z_video, Key = z_landmark, Value = z_landmark
        # attn_out: (B, T_video, z_dim)
        attn_out, _ = self.cross_attn(
            query=z_video,
            key=z_landmark,
            value=z_landmark,
        )
        
        # Residual Connection 1 + Norm 1 (Add & Norm)
        x = self.norm1(z_video + attn_out)
        
        # Feed-Forward + Residual Connection 2 + Norm 2
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        
        return out

