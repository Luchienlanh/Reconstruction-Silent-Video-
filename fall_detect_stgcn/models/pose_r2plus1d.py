"""Pose R(2+1)D classifier — factored 3D convolution over skeleton maps."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..pose_features import NUM_LANDMARKS, LANDMARK_DIM


def _diff(x: torch.Tensor) -> torch.Tensor:
    first = torch.zeros_like(x[:, :1])
    return torch.cat([first, x[:, 1:] - x[:, :-1]], dim=1)


def _groups(channels: int) -> int:
    for group in (16, 8, 4, 2):
        if channels % group == 0:
            return group
    return 1


class PoseR2Plus1DBlock(nn.Module):
    """R(2+1)D block over skeleton maps: [B, C, T, J, 1]."""

    def __init__(self, in_channels: int, out_channels: int, temporal_stride: int = 1, joint_stride: int = 1):
        super().__init__()
        mid_channels = max(out_channels, (in_channels * out_channels * 3 * 3) // max(in_channels * 3 + out_channels * 3, 1))
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels, mid_channels,
                kernel_size=(1, 3, 1), stride=(1, joint_stride, 1),
                padding=(0, 1, 0), bias=False,
            ),
            nn.GroupNorm(_groups(mid_channels), mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                mid_channels, out_channels,
                kernel_size=(3, 1, 1), stride=(temporal_stride, 1, 1),
                padding=(1, 0, 0), bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        if in_channels != out_channels or temporal_stride != 1 or joint_stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=(temporal_stride, joint_stride, 1),
                    bias=False,
                ),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(x) + self.skip(x))


class PoseR2Plus1DClassifier(nn.Module):
    """ResNet2+1D-style classifier without raw video.

    The input pose sequence is converted to a skeleton map:
    [B, T, 33 joints, 10 channels] where channels are position, velocity,
    acceleration, and visibility. The model applies spatial joint convolution
    and temporal convolution separately, mirroring R(2+1)D factorization.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        del input_dim
        width = hidden_size
        self.stem = nn.Sequential(
            nn.Conv3d(10, width, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(inplace=True),
            nn.Conv3d(width, width, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        channels = width
        for index in range(num_layers):
            out_channels = min(width * (2 ** (index // 2)), width * 4)
            joint_stride = 2 if index > 0 and index % 2 == 0 else 1
            blocks.append(PoseR2Plus1DBlock(channels, out_channels, joint_stride=joint_stride))
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    @staticmethod
    def _to_skeleton_map(x: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = x.shape
        pose = x.view(bsz, steps, NUM_LANDMARKS, LANDMARK_DIM)
        coords = pose[..., :3]
        visibility = pose[..., 3:4]
        d1 = _diff(coords)
        d2 = _diff(d1)
        skeleton = torch.cat([coords, d1, d2, visibility], dim=-1)
        return skeleton.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_skeleton_map(x)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
