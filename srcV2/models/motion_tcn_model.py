from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_model import R2Block, norm3d


class MotionVisualTower(nn.Module):
    def __init__(self, dim: int = 256, spatial_tokens: int = 2):
        super().__init__()
        self.spatial_tokens = int(spatial_tokens)
        self.stem = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
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

    @staticmethod
    def _frame_delta(video: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(video)
        if video.shape[2] > 1:
            delta[:, :, 1:] = video[:, :, 1:] - video[:, :, :-1]
        return delta

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = torch.cat([video.float(), self._frame_delta(video.float())], dim=1)
        x = self.pool(self.layers(self.stem(x)))
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t, h * w * c)
        return self.proj(x)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=dim,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.silu(y)
        y = self.pointwise(y).transpose(1, 2)
        x = x + y + self.ffn(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class LandmarkTCNTower(nn.Module):
    def __init__(self, num_points: int = 40, dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_points = int(num_points)
        self.input = nn.Sequential(
            nn.Linear(self.num_points * 6, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([TemporalConvBlock(dim, d, dropout) for d in (1, 2, 4, 8)])
        self.out = nn.LayerNorm(dim)

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
        xy_n = xy_c / scale
        if landmarks.shape[-1] >= 6:
            d1 = landmarks[..., 2:4] / scale
            d2 = landmarks[..., 4:6] / scale
        else:
            d1 = self._delta(xy_n)
            d2 = self._delta(d1)
        return torch.cat([xy_n, d1, d2], dim=-1)

    def forward(self, landmarks: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._normalize(landmarks).flatten(start_dim=2)
        x = self.input(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.out(x)


class MotionTCNLipToSpeechModel(nn.Module):
    """Motion-aware visual encoder + landmark TCN + dilated TCN mel decoder."""

    def __init__(
        self,
        dim: int = 256,
        spatial_tokens: int = 2,
        num_points: int = 40,
        dropout: float = 0.0,
        decoder_layers: int = 8,
    ):
        super().__init__()
        lm_dim = max(64, dim // 2)
        self.visual = MotionVisualTower(dim=dim, spatial_tokens=spatial_tokens)
        self.landmarks = LandmarkTCNTower(num_points=num_points, dim=lm_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(dim + lm_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.frame_refine = nn.ModuleList([TemporalConvBlock(dim, d, dropout) for d in (1, 2, 4, 8)])
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        dilations = [1, 2, 4, 8]
        self.decoder = nn.ModuleList(
            [TemporalConvBlock(dim, dilations[i % len(dilations)], dropout) for i in range(int(decoder_layers))]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.mel_head = nn.Linear(dim, 80)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        video = batch["video"]
        landmarks = batch["landmarks"]
        video_mask = batch.get("video_mask")
        mel = batch.get("mel")
        mel_mask = batch.get("mel_mask")

        z_vis = self.visual(video)
        z_lm = self.landmarks(landmarks, video_mask)
        z = self.fusion(torch.cat([z_vis, z_lm], dim=-1))
        for block in self.frame_refine:
            z = block(z, video_mask)

        x = self.upsample(z.transpose(1, 2))
        if mel_mask is not None:
            target_len = int(mel_mask.shape[1])
        elif mel is not None:
            target_len = int(mel.shape[1])
        else:
            target_len = int(round(z.shape[1] * 2.5))
        if x.shape[2] != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)

        h = x.transpose(1, 2).contiguous()
        for block in self.decoder:
            h = block(h, mel_mask)
        out = self.mel_head(self.out_norm(h))
        if mel_mask is not None:
            out = out * mel_mask.unsqueeze(-1).to(out.dtype)
        return out
