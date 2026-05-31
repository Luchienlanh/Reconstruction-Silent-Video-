from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motion_tcn_model import MotionVisualTower, TemporalConvBlock


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    w = mask.to(x.device, x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)


def masked_std(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    mu = masked_mean(x, mask, dim).unsqueeze(dim)
    if mask is None:
        return (x - mu).pow(2).mean(dim=dim).sqrt()
    w = mask.to(x.device, x.dtype).unsqueeze(-1)
    var = ((x - mu).pow(2) * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)
    return var.clamp_min(1e-6).sqrt()


class ConformerLiteBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv_pw1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.conv_dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.conv_pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv_drop = nn.Dropout(dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        y = self.attn_norm(x)
        key_padding_mask = None if mask is None else ~mask.to(torch.bool)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.attn_drop(y)

        y = self.conv_norm(x).transpose(1, 2)
        y = F.glu(self.conv_pw1(y), dim=1)
        y = F.silu(self.conv_dw(y))
        y = self.conv_pw2(y).transpose(1, 2)
        x = x + self.conv_drop(y)
        x = x + 0.5 * self.ffn2(x)
        x = self.out_norm(x)
        if mask is not None:
            x = x * mask.to(x.device, x.dtype).unsqueeze(-1)
        return x


class LipGeometryTower(nn.Module):
    def __init__(self, dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(36, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([TemporalConvBlock(dim, d, dropout) for d in (1, 2, 4, 8)])
        self.out = nn.LayerNorm(dim)

    @staticmethod
    def _pair_dist(xy: torch.Tensor, left: int, right: int) -> torch.Tensor:
        return (xy[:, :, left] - xy[:, :, right]).pow(2).sum(dim=-1, keepdim=True).sqrt()

    @staticmethod
    def _signed_area(xy: torch.Tensor) -> torch.Tensor:
        x = xy[..., 0]
        y = xy[..., 1]
        return 0.5 * (x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1)).sum(
            dim=-1, keepdim=True
        ).abs()

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= 1:
            return torch.zeros_like(x)
        d = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)

    def _features(self, landmarks: torch.Tensor) -> torch.Tensor:
        xy = torch.nan_to_num(landmarks[..., :2].float(), nan=0.0, posinf=0.0, neginf=0.0)
        center = xy.mean(dim=2)
        xy_c = xy - center.unsqueeze(2)
        scale = xy_c.pow(2).sum(dim=-1).sqrt().amax(dim=2, keepdim=True).clamp_min(1e-4)

        outer = xy[:, :, :20]
        inner = xy[:, :, 20:40]
        outer_w = self._pair_dist(xy, 0, 10)
        outer_h = self._pair_dist(xy, 15, 5)
        inner_w = self._pair_dist(xy, 20, 30)
        inner_h = self._pair_dist(xy, 35, 25)
        open_ratio = inner_h / inner_w.clamp_min(1e-4)
        outer_area = self._signed_area(outer)
        inner_area = self._signed_area(inner)
        raw = torch.cat(
            [
                center,
                outer_w,
                outer_h,
                inner_w,
                inner_h,
                open_ratio,
                outer_area,
                inner_area,
                scale,
            ],
            dim=-1,
        )
        normed = torch.cat(
            [
                center - 0.5,
                outer_w / scale,
                outer_h / scale,
                inner_w / scale,
                inner_h / scale,
                open_ratio,
                outer_area / scale.pow(2),
                inner_area / scale.pow(2),
                scale,
            ],
            dim=-1,
        )
        d1 = self._delta(normed)
        d2 = self._delta(d1)
        return torch.cat([normed, d1, d2, raw[..., :6]], dim=-1)

    def forward(self, landmarks: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input(self._features(landmarks))
        for block in self.blocks:
            x = block(x, mask)
        return self.out(x)


class ContextSyncLipToSpeechModel(nn.Module):
    """SVTS/VCA-GAN inspired visual context model with landmark geometry conditioning."""

    def __init__(
        self,
        dim: int = 256,
        spatial_tokens: int = 2,
        num_points: int = 40,
        dropout: float = 0.0,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        heads: int = 4,
        num_units: int = 0,
    ):
        super().__init__()
        if num_points != 40:
            raise ValueError("ContextSyncLipToSpeechModel currently expects 40 lip landmarks.")
        geo_dim = max(64, dim // 2)
        self.visual = MotionVisualTower(dim=dim, spatial_tokens=spatial_tokens)
        self.geometry = LipGeometryTower(dim=geo_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(dim + geo_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.ModuleList(
            [ConformerLiteBlock(dim, heads=heads, kernel_size=7, dropout=dropout) for _ in range(int(encoder_layers))]
        )
        self.global_film = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.decoder = nn.ModuleList(
            [ConformerLiteBlock(dim, heads=heads, kernel_size=9, dropout=dropout) for _ in range(int(decoder_layers))]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.mel_head = nn.Linear(dim, 80)
        self.energy_head = nn.Linear(dim, 1)
        self.num_units = int(num_units)
        self.unit_head = nn.Linear(dim, self.num_units) if self.num_units > 0 else None

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        video = batch["video"]
        landmarks = batch["landmarks"]
        video_mask = batch.get("video_mask")
        mel = batch.get("mel")
        mel_mask = batch.get("mel_mask")

        z_vis = self.visual(video)
        z_geo = self.geometry(landmarks, video_mask)
        z = self.fusion(torch.cat([z_vis, z_geo], dim=-1))
        for block in self.encoder:
            z = block(z, video_mask)

        summary = torch.cat([masked_mean(z, video_mask, 1), masked_std(z, video_mask, 1)], dim=-1)
        scale, shift = self.global_film(summary).chunk(2, dim=-1)
        z = z * (1.0 + 0.5 * torch.tanh(scale).unsqueeze(1)) + 0.5 * shift.unsqueeze(1)
        if video_mask is not None:
            z = z * video_mask.to(z.device, z.dtype).unsqueeze(-1)

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
        h = self.out_norm(h)
        out = self.mel_head(h)
        if mel_mask is not None:
            out = out * mel_mask.to(out.device, out.dtype).unsqueeze(-1)
        if bool(batch.get("return_aux", False)):
            aux = {"mel": out}
            if self.unit_head is not None:
                aux["unit_logits"] = self.unit_head(h)
            return aux
        return out
