from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFourier(nn.Module):
    def __init__(self, dim: int, num_freqs: int = 32, max_freq: float = 16.0):
        super().__init__()
        self.register_buffer("freqs", torch.linspace(1.0, max_freq, num_freqs), persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(num_freqs * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        angles = times.unsqueeze(-1).float() * self.freqs.to(times.device).view(1, 1, -1) * (2.0 * math.pi)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(feats.to(dtype=times.dtype))


class TCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = 2 * dilation
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=5, padding=padding, dilation=dilation)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + F.gelu(y)
        x = x + self.ffn(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class LandmarkMotionEncoder(nn.Module):
    def __init__(
        self,
        num_points: int = 40,
        dim: int = 384,
        tcn_layers: int = 6,
        transformer_layers: int = 2,
        nhead: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.geometry_dim = 20
        self.input = nn.Sequential(
            nn.Linear(self.num_points * 6 + self.geometry_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.time = TimeFourier(dim)
        dilations = [1, 2, 4, 8]
        self.tcn = nn.ModuleList([TCNBlock(dim, dilations[i % len(dilations)], dropout) for i in range(tcn_layers)])
        if transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        else:
            self.temporal = None
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

    def _geometry(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[..., :2]
        d1 = x[..., 2:4]
        d2 = x[..., 4:6]
        xs = xy[..., 0]
        ys = xy[..., 1]
        width = (xs.amax(dim=2) - xs.amin(dim=2)).unsqueeze(-1)
        height = (ys.amax(dim=2) - ys.amin(dim=2)).unsqueeze(-1)
        area = width * height
        aspect = height / width.clamp_min(1e-4)
        std_x = xs.std(dim=2, unbiased=False, keepdim=True)
        std_y = ys.std(dim=2, unbiased=False, keepdim=True)
        radial = xy.pow(2).sum(dim=-1).sqrt()
        radial_mean = radial.mean(dim=2, keepdim=True)
        radial_std = radial.std(dim=2, unbiased=False, keepdim=True)
        speed = d1.pow(2).sum(dim=-1).sqrt()
        accel = d2.pow(2).sum(dim=-1).sqrt()
        speed_mean = speed.mean(dim=2, keepdim=True)
        speed_std = speed.std(dim=2, unbiased=False, keepdim=True)
        speed_max = speed.amax(dim=2, keepdim=True)
        accel_mean = accel.mean(dim=2, keepdim=True)
        accel_std = accel.std(dim=2, unbiased=False, keepdim=True)
        motion = torch.cat([self._delta(width), self._delta(height), self._delta(area)], dim=-1)
        motion2 = self._delta(motion)
        return torch.cat(
            [
                width,
                height,
                area,
                aspect,
                std_x,
                std_y,
                radial_mean,
                radial_std,
                speed_mean,
                speed_std,
                speed_max,
                accel_mean,
                accel_std,
                motion,
                motion2,
                self._delta(speed_mean),
            ],
            dim=-1,
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        video_times: torch.Tensor | None = None,
        landmark_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._normalize(landmarks)
        geom = self._geometry(x)
        x = torch.cat([x.flatten(start_dim=2), geom], dim=-1)
        x = self.input(x)
        if video_times is not None:
            x = x + self.time(video_times)
        for block in self.tcn:
            x = block(x, landmark_mask)
        if self.temporal is not None:
            key_padding_mask = None if landmark_mask is None else ~landmark_mask.bool()
            x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        x = self.out(x)
        if landmark_mask is not None:
            x = x * landmark_mask.unsqueeze(-1).to(x.dtype)
        return x


class LandmarkCTCModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_points: int = 40,
        dim: int = 384,
        tcn_layers: int = 6,
        transformer_layers: int = 2,
        nhead: int = 6,
        dropout: float = 0.1,
        blank_bias_init: float = -3.0,
    ):
        super().__init__()
        self.encoder = LandmarkMotionEncoder(
            num_points=num_points,
            dim=dim,
            tcn_layers=tcn_layers,
            transformer_layers=transformer_layers,
            nhead=nhead,
            dropout=dropout,
        )
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, int(vocab_size)),
        )
        with torch.no_grad():
            linear = self.ctc_head[-1]
            if isinstance(linear, nn.Linear) and linear.bias is not None and linear.bias.numel() > 0:
                linear.bias.zero_()
                linear.bias[0] = float(blank_bias_init)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            batch["landmarks"],
            video_times=batch.get("video_times"),
            landmark_mask=batch.get("landmark_mask"),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.ctc_head(self.encode(batch))

