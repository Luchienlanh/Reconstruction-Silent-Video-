"""Speech-inspired TCN + Transformer classifier for pose sequences."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..pose_features import NUM_LANDMARKS, LANDMARK_DIM


def _diff(x: torch.Tensor) -> torch.Tensor:
    first = torch.zeros_like(x[:, :1])
    return torch.cat([first, x[:, 1:] - x[:, :-1]], dim=1)


def pose_motion_features(x: torch.Tensor) -> torch.Tensor:
    """Create position + velocity + acceleration + geometry features.

    Args:
        x: [B, T, 132] normalized MediaPipe pose features.

    Returns:
        [B, T, 336] feature tensor.
    """
    bsz, steps, _ = x.shape
    pose = x.view(bsz, steps, NUM_LANDMARKS, LANDMARK_DIM)
    coords = pose[..., :3]
    visibility = pose[..., 3:4]
    d1 = _diff(coords)
    d2 = _diff(d1)

    flat = torch.cat([coords, d1, d2, visibility], dim=-1).flatten(start_dim=2)

    xy = coords[..., :2]
    min_xy = xy.amin(dim=2)
    max_xy = xy.amax(dim=2)
    width_height = max_xy - min_xy
    speed = d1.norm(dim=-1).mean(dim=2, keepdim=True)
    accel = d2.norm(dim=-1).mean(dim=2, keepdim=True)
    torso = (coords[:, :, 11, :2] + coords[:, :, 12, :2]) * 0.5 - (
        coords[:, :, 23, :2] + coords[:, :, 24, :2]
    ) * 0.5
    torso_angle = torch.atan2(torso[..., 1:2], torso[..., 0:1]) / torch.pi
    visibility_mean = visibility.mean(dim=2)
    geom = torch.cat([width_height, speed, accel, torso_angle, visibility_mean], dim=-1)
    return torch.cat([flat, geom], dim=-1)


MOTION_DIM = NUM_LANDMARKS * 10 + 6


class TemporalTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.05):
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


class SpeechTCNPoseClassifier(nn.Module):
    """Speech-style landmark-motion encoder adapted to full-body pose."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        transformer_layers: int = 1,
    ) -> None:
        super().__init__()
        del input_dim
        self.input = nn.Sequential(
            nn.Linear(MOTION_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.ModuleList(
            [
                TemporalTCNBlock(hidden_size, dilation=2 ** (i % 4), dropout=dropout)
                for i in range(num_layers)
            ]
        )
        if transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=max(1, min(8, hidden_size // 64)),
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        else:
            self.temporal = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(pose_motion_features(x))
        for block in self.tcn:
            x = block(x)
        x = self.temporal(x)
        pooled = torch.cat([x.mean(dim=1), x[:, -1]], dim=-1)
        return self.head(pooled)
