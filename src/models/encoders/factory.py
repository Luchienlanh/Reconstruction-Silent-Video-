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
from .non_snn import NonSpikingViTEncoder

def build_encoder(encoder_type="snn", **kwargs):
    encoder_type = encoder_type.lower()
    if encoder_type == "snn":
        return SpikingViTEncoder(**kwargs)
    if encoder_type in {"non_snn", "nonsnn", "cnn_transformer"}:
        return NonSpikingViTEncoder(**kwargs)
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
    """Fuse the visual encoder with landmark dynamics before the mel decoder using simple concatenation."""
    def __init__(self, visual_encoder, num_landmark_points, z_dim=512):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.landmark_encoder = LandmarkMotionEncoder(num_landmark_points, out_dim=z_dim)
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


class LandmarkTemporalBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        x = x + F.silu(y)
        return x + self.ffn(x)


class LandmarkMotionEncoder(nn.Module):
    """
    Landmark-first temporal encoder.

    It keeps landmark motion as the main signal: normalized position, velocity,
    acceleration, and simple mouth geometry are processed by TCN blocks followed
    by a small Transformer.
    """

    def __init__(
        self,
        num_points: int,
        hidden_dim: int = 512,
        out_dim: int = 512,
        n_heads: int = 8,
        n_transformer_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.geometry_dim = 14
        in_dim = num_points * 6 + self.geometry_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.ModuleList(
            [
                LandmarkTemporalBlock(hidden_dim, dilation=1, dropout=dropout),
                LandmarkTemporalBlock(hidden_dim, dilation=2, dropout=dropout),
                LandmarkTemporalBlock(hidden_dim, dilation=4, dropout=dropout),
                LandmarkTemporalBlock(hidden_dim, dilation=8, dropout=dropout),
            ]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        d = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)

    def _normalize(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = torch.nan_to_num(landmarks.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xy = landmarks[..., :2]
        center = xy.mean(dim=2, keepdim=True)
        xy_centered = xy - center
        scale = (
            xy_centered.pow(2).sum(dim=-1).sqrt()
            .amax(dim=2, keepdim=True)
            .unsqueeze(-1)
            .clamp_min(1e-4)
        )

        xy_norm = xy_centered / scale
        if landmarks.shape[-1] >= 6:
            d1 = landmarks[..., 2:4] / scale
            d2 = landmarks[..., 4:6] / scale
        else:
            d1 = self._delta(xy_norm)
            d2 = self._delta(d1)
        return torch.cat([xy_norm, d1, d2], dim=-1)

    def _geometry(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[..., :2]
        x_coord = xy[..., 0]
        y_coord = xy[..., 1]
        width = (x_coord.amax(dim=2) - x_coord.amin(dim=2)).unsqueeze(-1)
        height = (y_coord.amax(dim=2) - y_coord.amin(dim=2)).unsqueeze(-1)
        area = width * height
        aspect = height / width.clamp_min(1e-4)
        std_x = x_coord.std(dim=2, unbiased=False, keepdim=True)
        std_y = y_coord.std(dim=2, unbiased=False, keepdim=True)
        radial = xy.pow(2).sum(dim=-1).sqrt()
        radial_mean = radial.mean(dim=2, keepdim=True)
        radial_std = radial.std(dim=2, unbiased=False, keepdim=True)

        shape = torch.cat([width, height, area, aspect, std_x, std_y, radial_mean, radial_std], dim=-1)
        motion = torch.cat([self._delta(width), self._delta(height), self._delta(area)], dim=-1)
        accel = self._delta(motion)
        return torch.cat([shape, motion, accel], dim=-1)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        x = self._normalize(landmarks)
        geometry = self._geometry(x)
        x = torch.cat([x.flatten(start_dim=2), geometry], dim=-1)
        x = self.input_proj(x)
        for block in self.tcn:
            x = block(x)
        x = self.temporal(x)
        return self.output(x)


class VisualLandmarkEncoderV2(nn.Module):
    """Fuse the visual encoder with 6-dim landmark dynamics before the mel decoder using Multi-Head Cross-Attention."""

    def __init__(self, visual_encoder: nn.Module, num_landmark_points: int, z_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.landmark_encoder = LandmarkMotionEncoder(num_landmark_points, out_dim=z_dim, dropout=dropout)
        
        # Multi-Head Cross-Attention where landmark motion is the query and video is supporting context.
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
        
        z_video = self.visual_encoder(video)
        z_landmark = self.landmark_encoder(landmarks)
        
        # Align temporal dimensions if needed
        if z_video.shape[1] != z_landmark.shape[1]:
            z_video = F.interpolate(
                z_video.transpose(1, 2),
                size=z_landmark.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
            
        attn_out, _ = self.cross_attn(
            query=z_landmark,
            key=z_video,
            value=z_video,
        )
        
        x = self.norm1(z_landmark + attn_out)
        
        # Feed-Forward + Residual Connection 2 + Norm 2
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        
        return out

class VisualLandmarkEncoderGatedResidual(nn.Module):
    """Fuse visual and landmark features with video as the residual path."""

    def __init__(self, visual_encoder: nn.Module, num_landmark_points: int, z_dim: int = 512, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.landmark_encoder = LandmarkMotionEncoder(num_landmark_points, out_dim=z_dim, dropout=dropout)
        self.landmark_proj = nn.Linear(z_dim, z_dim)
        self.gate = nn.Sequential(
            nn.Linear(z_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )
        self.norm = nn.LayerNorm(z_dim)

        nn.init.zeros_(self.landmark_proj.bias)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, video: torch.Tensor, landmarks: torch.Tensor = None) -> torch.Tensor:
        if landmarks is None:
            raise ValueError("VisualLandmarkEncoderGatedResidual requires landmarks.")

        z_video = self.visual_encoder(video)
        z_landmark = self.landmark_encoder(landmarks)

        if z_landmark.shape[1] != z_video.shape[1]:
            z_landmark = F.interpolate(
                z_landmark.transpose(1, 2),
                size=z_video.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()

        gate = torch.sigmoid(self.gate(torch.cat([z_video, z_landmark], dim=-1)))
        return self.norm(z_video + gate * self.landmark_proj(z_landmark))


class VisualLandmarkEncoderLandmarkFirst(nn.Module):
    """Use landmark motion as the residual path and gated visual features as support."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        num_landmark_points: int,
        z_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.landmark_encoder = LandmarkMotionEncoder(
            num_landmark_points,
            hidden_dim=hidden_dim,
            out_dim=z_dim,
            dropout=dropout,
        )
        self.video_proj = nn.Linear(z_dim, z_dim)
        self.gate = nn.Sequential(
            nn.Linear(z_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, z_dim),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, z_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(z_dim * 2, z_dim),
        )
        self.norm = nn.LayerNorm(z_dim)

        nn.init.zeros_(self.video_proj.bias)
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(self, video: torch.Tensor, landmarks: torch.Tensor = None) -> torch.Tensor:
        if landmarks is None:
            raise ValueError("VisualLandmarkEncoderLandmarkFirst requires landmarks.")

        z_landmark = self.landmark_encoder(landmarks)
        z_video = self.visual_encoder(video)
        if z_video.shape[1] != z_landmark.shape[1]:
            z_video = F.interpolate(
                z_video.transpose(1, 2),
                size=z_landmark.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()

        gate = torch.sigmoid(self.gate(torch.cat([z_landmark, z_video], dim=-1)))
        z = self.norm(z_landmark + gate * self.video_proj(z_video))
        return self.norm(z + self.refine(z))

