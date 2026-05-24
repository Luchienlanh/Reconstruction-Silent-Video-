from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFourier(nn.Module):
    def __init__(self, dim: int, num_freqs: int = 64, max_freq: float = 64.0):
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


class R2Plus1DVisualTower(nn.Module):
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
        self.proj = nn.Sequential(nn.Linear(512, dim), nn.LayerNorm(dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, self.spatial_tokens * self.spatial_tokens, dim) * 0.02)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.layers(self.stem(video.float())))
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t, h * w, c)
        return self.proj(x) + self.spatial_embed[:, :, : h * w]


class LandmarkTemporalBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return x + F.silu(y) + self.ffn(x)


class LandmarkMotionTower(nn.Module):
    def __init__(self, num_points: int = 40, dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.num_points = int(num_points)
        self.geometry_dim = 14
        self.input = nn.Sequential(
            nn.Linear(self.num_points * 6 + self.geometry_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.ModuleList([LandmarkTemporalBlock(dim, d, dropout) for d in (1, 2, 4, 8)])
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
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
        if landmarks.shape[-1] >= 6:
            d1 = landmarks[..., 2:4] / scale
            d2 = landmarks[..., 4:6] / scale
        else:
            d1 = self._delta(xy_c / scale)
            d2 = self._delta(d1)
        return torch.cat([xy_c / scale, d1, d2], dim=-1)

    def _geometry(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[..., :2]
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
        base = torch.cat([width, height, area, aspect, std_x, std_y, radial_mean, radial_std], dim=-1)
        motion = torch.cat([self._delta(width), self._delta(height), self._delta(area)], dim=-1)
        accel = self._delta(motion)
        return torch.cat([base, motion, accel], dim=-1)

    def forward(self, landmarks: torch.Tensor, video_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._normalize(landmarks)
        x = torch.cat([x.flatten(start_dim=2), self._geometry(x)], dim=-1)
        x = self.input(x)
        for block in self.tcn:
            x = block(x)
        key_padding_mask = None if video_mask is None else ~video_mask.bool()
        return self.out(self.temporal(x, src_key_padding_mask=key_padding_mask))


class R2INRMemoryEncoder(nn.Module):
    def __init__(self, dim: int = 512, spatial_tokens: int = 4, num_points: int = 40, dropout: float = 0.0):
        super().__init__()
        self.visual = R2Plus1DVisualTower(dim=dim, spatial_tokens=spatial_tokens)
        self.landmarks = LandmarkMotionTower(num_points=num_points, dim=dim, dropout=dropout)
        self.time = TimeFourier(dim)
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video = batch["video"]
        landmarks = batch["landmarks"]
        video_times = batch["video_times"]
        video_mask = batch["video_mask"].bool()

        z_vis = self.visual(video)
        z_lm = self.landmarks(landmarks, video_mask=video_mask).unsqueeze(2)
        z = torch.cat([z_vis, z_lm], dim=2)
        b, t, k, d = z.shape
        time_emb = self.time(video_times).unsqueeze(2)
        z = self.norm(z + time_emb + self.fuse(z))
        memory = z.reshape(b, t * k, d)
        memory_times = video_times.unsqueeze(-1).expand(b, t, k).reshape(b, t * k)
        memory_mask = video_mask.unsqueeze(-1).expand(b, t, k).reshape(b, t * k)
        frame_memory = z.mean(dim=2)
        denom = memory_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        global_token = (memory * memory_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return {
            "memory": memory,
            "memory_times": memory_times,
            "memory_mask": memory_mask,
            "frame_memory": frame_memory,
            "frame_mask": video_mask,
            "global": global_token,
        }


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = float(omega)
        bound = math.sqrt(6.0 / in_dim) / self.omega
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class MelTemporalBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = 2 * dilation
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=5, padding=padding, dilation=dilation)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + F.silu(y)
        x = x + self.ffn(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class INRMelDecoder(nn.Module):
    def __init__(self, dim: int = 512, out_dim: int = 80, dropout: float = 0.0, output_bias_init: float = -4.0):
        super().__init__()
        self.time = TimeFourier(dim, num_freqs=64, max_freq=96.0)
        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=8, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.aligned_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.mel_refine = nn.ModuleList(
            [
                MelTemporalBlock(dim, dilation=1, dropout=dropout),
                MelTemporalBlock(dim, dilation=2, dropout=dropout),
                MelTemporalBlock(dim, dilation=4, dropout=dropout),
                MelTemporalBlock(dim, dilation=8, dropout=dropout),
            ]
        )
        self.coarse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )
        self.time_direct = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, out_dim),
        )
        self.time_conditioned = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, out_dim),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            SineLayer(dim, dim),
            nn.Linear(dim, out_dim),
        )
        self.time_direct_scale = nn.Parameter(torch.tensor(1.0))
        self.time_conditioned_scale = nn.Parameter(torch.tensor(0.25))
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        nn.init.constant_(self.coarse[-1].bias, output_bias_init)
        nn.init.zeros_(self.time_direct[-1].bias)
        nn.init.normal_(self.time_direct[-1].weight, 0.0, 1e-3)
        nn.init.zeros_(self.time_conditioned[-1].bias)
        nn.init.normal_(self.time_conditioned[-1].weight, 0.0, 1e-3)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.normal_(self.residual[-1].weight, 0.0, 1e-3)

    @staticmethod
    def _align_frame_memory(encoded: dict[str, torch.Tensor], target_len: int) -> torch.Tensor:
        frame_memory = encoded["frame_memory"]
        if frame_memory.shape[1] == target_len:
            return frame_memory
        return F.interpolate(
            frame_memory.transpose(1, 2),
            size=int(target_len),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).contiguous()

    def forward(self, encoded: dict[str, torch.Tensor], mel_times: torch.Tensor, mel_mask: torch.Tensor | None = None) -> torch.Tensor:
        q_time = self.time(mel_times)
        q = self.query(q_time + encoded["global"].unsqueeze(1))
        key_padding_mask = ~encoded["memory_mask"].bool()
        context, _ = self.attn(q, encoded["memory"], encoded["memory"], key_padding_mask=key_padding_mask, need_weights=False)
        aligned = self.aligned_proj(self._align_frame_memory(encoded, mel_times.shape[1]))
        h = self.cross_norm(q + context + aligned)
        for block in self.mel_refine:
            h = block(h, mel_mask)
        global_rep = encoded["global"].unsqueeze(1).expand(-1, h.shape[1], -1)
        features = torch.cat([h, global_rep], dim=-1)
        time_condition = torch.cat([q_time, global_rep], dim=-1)
        out = (
            self.coarse(features)
            + self.time_direct_scale * self.time_direct(q_time)
            + self.time_conditioned_scale * self.time_conditioned(time_condition)
            + self.residual_scale * self.residual(features)
        )
        if mel_mask is not None:
            out = out * mel_mask.unsqueeze(-1).to(out.dtype)
        return out


class R2INRModel(nn.Module):
    def __init__(self, dim: int = 512, spatial_tokens: int = 4, num_points: int = 40, dropout: float = 0.0):
        super().__init__()
        self.encoder = R2INRMemoryEncoder(dim=dim, spatial_tokens=spatial_tokens, num_points=num_points, dropout=dropout)
        self.decoder = INRMelDecoder(dim=dim, out_dim=80, dropout=dropout)
        self.mel_stats_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 160),
        )

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False):
        encoded = self.encoder(batch)
        mel = self.decoder(encoded, batch["mel_times"], batch.get("mel_mask"))
        if not return_aux:
            return mel
        return {
            "mel": mel,
            "mel_stats": self.mel_stats_head(encoded["global"]),
        }
