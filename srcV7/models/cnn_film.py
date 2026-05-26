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


def norm1d(channels: int) -> nn.GroupNorm:
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


class R2MemoryEncoder(nn.Module):
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


class FiLMConvBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.norm = norm1d(channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.film = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, channels * 2))
        self.ffn_norm = norm1d(channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.conv(self.norm(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma.transpose(1, 2)
        beta = beta.transpose(1, 2)
        y = F.silu(y * (1.0 + gamma) + beta)
        x = x + self.dropout(y)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if mask is not None:
            x = x * mask.unsqueeze(1).to(x.dtype)
        return x


class PlainConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.norm = norm1d(channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.ffn_norm = norm1d(channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(F.silu(self.conv(self.norm(x))))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if mask is not None:
            x = x * mask.unsqueeze(1).to(x.dtype)
        return x


class CNNFiLMMelDecoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        channels: int = 512,
        out_dim: int = 80,
        layers: int = 8,
        kernel_size: int = 5,
        dropout: float = 0.0,
        output_bias_init: float = -4.0,
        upsample_mode: str = "conv_transpose",
    ):
        super().__init__()
        self.upsample_mode = upsample_mode.lower()
        if self.upsample_mode == "conv_transpose":
            self.learned_upsample = nn.Sequential(
                nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            )
        self.global_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.input = nn.Conv1d(dim, channels, kernel_size=1)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                FiLMConvBlock(
                    channels=channels,
                    cond_dim=dim,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(layers)
            ]
        )
        self.head = nn.Sequential(
            norm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, out_dim, kernel_size=1),
        )
        nn.init.constant_(self.head[-1].bias, output_bias_init)
        nn.init.normal_(self.head[-1].weight, 0.0, 1e-3)

    def init_output_bias(self, mel_mean: torch.Tensor) -> None:
        last = self.head[-1]
        if last.bias is not None and last.bias.numel() == mel_mean.numel():
            with torch.no_grad():
                last.bias.copy_(mel_mean.to(device=last.bias.device, dtype=last.bias.dtype))

    def _align_frame_memory(self, encoded: dict[str, torch.Tensor], target_len: int) -> torch.Tensor:
        frame_memory = encoded["frame_memory"]
        if frame_memory.shape[1] == target_len:
            return frame_memory
        x = frame_memory.transpose(1, 2)
        if self.upsample_mode == "conv_transpose":
            x = self.learned_upsample(x)
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        return x.transpose(1, 2).contiguous()

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        mel_times: torch.Tensor,
        mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_len = int(mel_times.shape[1])
        local = self._align_frame_memory(encoded, target_len)
        global_rep = self.global_proj(encoded["global"]).unsqueeze(1)
        cond = self.cond_proj(local + global_rep)
        h = self.input(cond.transpose(1, 2))
        for block in self.blocks:
            h = block(h, cond, mel_mask)
        out = self.head(h).transpose(1, 2)
        if mel_mask is not None:
            out = out * mel_mask.unsqueeze(-1).to(out.dtype)
        return out


class CNNPlainMelDecoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        channels: int = 512,
        out_dim: int = 80,
        layers: int = 8,
        kernel_size: int = 5,
        dropout: float = 0.0,
        output_bias_init: float = -4.0,
        upsample_mode: str = "conv_transpose",
    ):
        super().__init__()
        self.upsample_mode = upsample_mode.lower()
        self.time = TimeFourier(dim, num_freqs=64, max_freq=96.0)
        if self.upsample_mode == "conv_transpose":
            self.learned_upsample = nn.Sequential(
                nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            )
        self.global_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.direct = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )
        self.input = nn.Conv1d(dim, channels, kernel_size=1)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                PlainConvBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(layers)
            ]
        )
        self.residual = nn.Sequential(
            norm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, out_dim, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.25))
        nn.init.constant_(self.direct[-1].bias, output_bias_init)
        nn.init.normal_(self.direct[-1].weight, 0.0, 1e-3)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.normal_(self.residual[-1].weight, 0.0, 1e-3)

    def init_output_bias(self, mel_mean: torch.Tensor) -> None:
        last = self.direct[-1]
        if last.bias is not None and last.bias.numel() == mel_mean.numel():
            with torch.no_grad():
                last.bias.copy_(mel_mean.to(device=last.bias.device, dtype=last.bias.dtype))
        res_last = self.residual[-1]
        if res_last.bias is not None:
            with torch.no_grad():
                res_last.bias.zero_()

    def _align_frame_memory(self, encoded: dict[str, torch.Tensor], target_len: int) -> torch.Tensor:
        frame_memory = encoded["frame_memory"]
        if frame_memory.shape[1] == target_len:
            return frame_memory
        x = frame_memory.transpose(1, 2)
        if self.upsample_mode == "conv_transpose":
            x = self.learned_upsample(x)
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        return x.transpose(1, 2).contiguous()

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        mel_times: torch.Tensor,
        mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_len = int(mel_times.shape[1])
        local = self._align_frame_memory(encoded, target_len)
        global_rep = self.global_proj(encoded["global"]).unsqueeze(1).expand(-1, target_len, -1)
        time = self.time(mel_times)
        cond = self.cond_proj(torch.cat([local, global_rep, time], dim=-1))
        coarse = self.direct(cond)
        h = self.input(cond.transpose(1, 2))
        for block in self.blocks:
            h = block(h, mel_mask)
        residual = self.residual(h).transpose(1, 2)
        out = coarse + self.residual_scale * residual
        if mel_mask is not None:
            out = out * mel_mask.unsqueeze(-1).to(out.dtype)
        return out


class R2CNNFiLMModel(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        spatial_tokens: int = 4,
        num_points: int = 40,
        dropout: float = 0.0,
        upsample_mode: str = "conv_transpose",
        decoder_channels: int | None = None,
        decoder_layers: int = 8,
        decoder_kernel_size: int = 5,
    ):
        super().__init__()
        channels = int(decoder_channels or dim)
        self.encoder = R2MemoryEncoder(dim=dim, spatial_tokens=spatial_tokens, num_points=num_points, dropout=dropout)
        self.decoder = CNNFiLMMelDecoder(
            dim=dim,
            channels=channels,
            out_dim=80,
            layers=decoder_layers,
            kernel_size=decoder_kernel_size,
            dropout=dropout,
            upsample_mode=upsample_mode,
        )
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


class R2CNNPlainModel(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        spatial_tokens: int = 4,
        num_points: int = 40,
        dropout: float = 0.0,
        upsample_mode: str = "conv_transpose",
        decoder_channels: int | None = None,
        decoder_layers: int = 8,
        decoder_kernel_size: int = 5,
    ):
        super().__init__()
        channels = int(decoder_channels or dim)
        self.encoder = R2MemoryEncoder(dim=dim, spatial_tokens=spatial_tokens, num_points=num_points, dropout=dropout)
        self.decoder = CNNPlainMelDecoder(
            dim=dim,
            channels=channels,
            out_dim=80,
            layers=decoder_layers,
            kernel_size=decoder_kernel_size,
            dropout=dropout,
            upsample_mode=upsample_mode,
        )
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
