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
from .non_snn import NonSpikingTemporalEncoder, NonSpikingVidResNet, NonSpikingViTEncoder

FLOW_ENCODER_TYPES = {"ephrat_flow_r2plus1d", "flow_r2plus1d", "two_tower_flow"}


def is_flow_encoder_type(encoder_type: str) -> bool:
    return str(encoder_type).lower() in FLOW_ENCODER_TYPES


def build_encoder(encoder_type="snn", **kwargs):
    encoder_type = encoder_type.lower()
    if encoder_type == "snn":
        return SpikingViTEncoder(**kwargs)
    if encoder_type in {"non_snn", "nonsnn", "cnn_transformer"}:
        return NonSpikingViTEncoder(**kwargs)
    if is_flow_encoder_type(encoder_type):
        num_landmark_points = kwargs.pop("num_landmark_points", kwargs.pop("num_points", None))
        if num_landmark_points is None:
            raise ValueError(f"{encoder_type} requires num_landmark_points")
        return EphratFlowR2Plus1DEncoder(num_landmark_points=num_landmark_points, **kwargs)
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


class EphratFlowR2Plus1DEncoder(nn.Module):
    """
    Ephrat-style two-tower encoder with R(2+1)D backbones.

    The RGB tower sees appearance, the flow tower sees explicit motion, and the
    landmark tower keeps mouth geometry/dynamics as a stable residual cue.
    Input video can be either:
      - video tensor: (B, C, T, H, W), then a lightweight torch motion proxy is used;
      - (video, flow) tuple, where flow is (B, 2, T, H, W).
    Output: (B, T, 512).
    """

    def __init__(
        self,
        num_landmark_points: int,
        z_dim: int = 512,
        tower_layers: Optional[list[int]] = None,
        temporal_blocks: int = 2,
        n_heads: int = 8,
        dropout: float = 0.05,
        T_max: int = 1000,
    ):
        super().__init__()
        layers = tower_layers or [1, 1, 1, 1]
        self.rgb_tower = NonSpikingVidResNet(layers=layers, in_channels=1, spatial_pool_size=1)
        self.flow_tower = NonSpikingVidResNet(layers=layers, in_channels=2, spatial_pool_size=1)
        self.landmark_encoder = LandmarkMotionEncoder(
            num_landmark_points,
            hidden_dim=z_dim,
            out_dim=z_dim,
            dropout=dropout,
        )
        self.visual_encoder = nn.ModuleDict({"rgb": self.rgb_tower, "flow": self.flow_tower})

        self.rgb_proj = nn.Linear(z_dim, z_dim)
        self.flow_proj = nn.Linear(z_dim, z_dim)
        self.landmark_proj = nn.Linear(z_dim, z_dim)
        self.gate = nn.Sequential(
            nn.Linear(z_dim * 3, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(z_dim, 2),
        )
        self.refine = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, z_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(z_dim * 2, z_dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, T_max, z_dim) * 0.02)
        self.temporal = NonSpikingTemporalEncoder(
            z_dim=z_dim,
            n_heads=n_heads,
            n_blocks=temporal_blocks,
            mlp_hidden_dim=z_dim * 4,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(z_dim)

        nn.init.constant_(self.gate[-1].bias, -1.0)

    @staticmethod
    def _align_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        if x.shape[1] == target_t:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=target_t,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).contiguous()

    @staticmethod
    def _estimate_motion_flow(video: torch.Tensor) -> torch.Tensor:
        gray = video.float().mean(dim=1, keepdim=True)
        dt = gray[:, :, 1:] - gray[:, :, :-1]
        dt = torch.cat([torch.zeros_like(dt[:, :, :1]), dt], dim=2)
        gx = gray[:, :, :, :, 1:] - gray[:, :, :, :, :-1]
        gx = F.pad(gx, (1, 0, 0, 0))
        gy = gray[:, :, :, 1:, :] - gray[:, :, :, :-1, :]
        gy = F.pad(gy, (0, 0, 1, 0))
        return torch.cat([dt * gx, dt * gy], dim=1)

    @staticmethod
    def _split_video_flow(video):
        if isinstance(video, (tuple, list)):
            if len(video) != 2:
                raise ValueError("Flow encoder expects video tuple/list as (frames, flow).")
            return video[0], video[1]
        return video, None

    def _prepare_flow(self, video: torch.Tensor, flow: Optional[torch.Tensor]) -> torch.Tensor:
        if flow is None:
            flow = self._estimate_motion_flow(video)
        flow = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if flow.dim() != 5:
            raise ValueError(f"Flow must be 5D, got {tuple(flow.shape)}")
        if flow.shape[1] != 2 and flow.shape[2] == 2:
            flow = flow.transpose(1, 2).contiguous()
        if flow.shape[1] != 2:
            raise ValueError(f"Flow must have 2 channels, got {tuple(flow.shape)}")

        target_size = (video.shape[2], video.shape[3], video.shape[4])
        if tuple(flow.shape[2:]) != target_size:
            flow = F.interpolate(flow, size=target_size, mode="trilinear", align_corners=False)
        return flow

    def forward(self, video, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if landmarks is None:
            raise ValueError("EphratFlowR2Plus1DEncoder requires landmarks.")

        video, flow = self._split_video_flow(video)
        video = torch.nan_to_num(video.float(), nan=0.0, posinf=0.0, neginf=0.0)
        flow = self._prepare_flow(video, flow)

        z_rgb = self.rgb_tower(video)
        z_flow = self.flow_tower(flow)
        z_lm = self.landmark_encoder(landmarks)

        target_t = z_lm.shape[1]
        z_rgb = self._align_time(z_rgb, target_t)
        z_flow = self._align_time(z_flow, target_t)

        gate = torch.sigmoid(self.gate(torch.cat([z_rgb, z_flow, z_lm], dim=-1)))
        z = self.rgb_proj(z_rgb)
        z = z + gate[..., 0:1] * self.flow_proj(z_flow)
        z = z + gate[..., 1:2] * self.landmark_proj(z_lm)
        z = self.norm(z + self.refine(z))

        if z.size(1) > self.pos_embedding.size(1):
            raise ValueError(f"T={z.size(1)} exceeds T_max={self.pos_embedding.size(1)}")
        z = z + self.pos_embedding[:, : z.size(1), :]
        return self.norm(self.temporal(z))


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

