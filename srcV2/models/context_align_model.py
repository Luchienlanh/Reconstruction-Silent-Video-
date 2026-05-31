from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_sync_model import ConformerLiteBlock, LipGeometryTower, masked_mean, masked_std
from .motion_tcn_model import MotionVisualTower, TemporalConvBlock


def _fallback_times(mask: torch.Tensor | None, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    if mask is None:
        return base.view(1, length)
    return base.view(1, length).expand(mask.shape[0], -1)


def _normalize_times(times: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    times = torch.nan_to_num(times.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is None:
        t0 = times[:, :1]
        span = (times[:, -1:] - t0).abs().clamp_min(1e-4)
        return (times - t0) / span
    mask_f = mask.to(times.device, times.dtype)
    masked = times.masked_fill(~mask.to(torch.bool), 0.0)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = masked.sum(dim=1, keepdim=True) / denom
    centered = (times - mean).masked_fill(~mask.to(torch.bool), 0.0)
    scale = centered.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    return centered / scale


def _time_features(times: torch.Tensor) -> torch.Tensor:
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=times.device, dtype=times.dtype)
    phase = times.unsqueeze(-1) * freqs.view(1, 1, -1) * (2.0 * math.pi)
    return torch.cat([times.unsqueeze(-1), torch.sin(phase), torch.cos(phase)], dim=-1)


class GatedFusion(nn.Module):
    def __init__(self, visual_dim: int, geometry_dim: int, dim: int, dropout: float = 0.0, modality_dropout: float = 0.0):
        super().__init__()
        self.vis_proj = nn.Sequential(nn.Linear(visual_dim, dim), nn.LayerNorm(dim), nn.SiLU())
        self.geo_proj = nn.Sequential(nn.Linear(geometry_dim, dim), nn.LayerNorm(dim), nn.SiLU())
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout))
        self.modality_dropout = float(modality_dropout)

    def forward(self, z_vis: torch.Tensor, z_geo: torch.Tensor) -> torch.Tensor:
        vis = self.vis_proj(z_vis)
        geo = self.geo_proj(z_geo)
        if self.training and self.modality_dropout > 0:
            p = min(max(self.modality_dropout, 0.0), 0.45)
            keep_vis = (torch.rand(vis.shape[0], 1, 1, device=vis.device) > p).to(vis.dtype)
            keep_geo = (torch.rand(geo.shape[0], 1, 1, device=geo.device) > p).to(geo.dtype)
            vis = vis * keep_vis
            geo = geo * keep_geo
        gate = torch.sigmoid(self.gate(torch.cat([vis, geo], dim=-1)))
        return self.out(vis + gate * geo)


class LocalMelCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, window: float = 0.18):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.window = float(window)
        self.query = nn.Sequential(
            nn.Linear(9, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        memory: torch.Tensor,
        video_times: torch.Tensor | None,
        mel_times: torch.Tensor | None,
        video_mask: torch.Tensor | None,
        mel_mask: torch.Tensor | None,
        target_len: int,
    ) -> torch.Tensor:
        b, tv, _ = memory.shape
        device = memory.device
        dtype = memory.dtype
        if video_times is None:
            video_times = _fallback_times(video_mask, tv, device, dtype)
        if mel_times is None:
            mel_times = _fallback_times(mel_mask, target_len, device, dtype)
        video_times = video_times[:, :tv].to(device=device, dtype=dtype)
        mel_times = mel_times[:, :target_len].to(device=device, dtype=dtype)
        q_seed = self.query(_time_features(_normalize_times(mel_times, mel_mask)))

        q = self._shape(self.q_proj(q_seed))
        k = self._shape(self.k_proj(memory))
        v = self._shape(self.v_proj(memory))
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        if self.window > 0:
            dist = (mel_times.unsqueeze(-1) - video_times.unsqueeze(1)).abs()
            local = dist <= self.window
            if video_mask is not None:
                local = local & video_mask.to(torch.bool).unsqueeze(1)
            no_local = ~local.any(dim=-1, keepdim=True)
            if no_local.any():
                fallback = torch.ones_like(local, dtype=torch.bool)
                if video_mask is not None:
                    fallback = fallback & video_mask.to(torch.bool).unsqueeze(1)
                local = torch.where(no_local, fallback, local)
            logits = logits.masked_fill(~local.unsqueeze(1), -1e4)
        elif video_mask is not None:
            logits = logits.masked_fill(~video_mask.to(torch.bool).view(b, 1, 1, tv), -1e4)

        attn = torch.softmax(logits.float(), dim=-1).to(dtype)
        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, target_len, self.dim)
        h = self.norm(q_seed + self.out_proj(h))
        h = h + self.ffn(h)
        if mel_mask is not None:
            h = h * mel_mask.to(device, dtype).unsqueeze(-1)
        return h


class ContextAlignLipToSpeechModel(nn.Module):
    """ContextSync backbone with explicit mel-time alignment and detail decoder."""

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
        align_window: float = 0.18,
        modality_dropout: float = 0.05,
        detail_scale: float = 0.35,
    ):
        super().__init__()
        if num_points != 40:
            raise ValueError("ContextAlignLipToSpeechModel currently expects 40 lip landmarks.")
        geo_dim = max(64, dim // 2)
        self.visual = MotionVisualTower(dim=dim, spatial_tokens=spatial_tokens)
        self.geometry = LipGeometryTower(dim=geo_dim, dropout=dropout)
        self.fusion = GatedFusion(dim, geo_dim, dim, dropout=dropout, modality_dropout=modality_dropout)
        self.encoder = nn.ModuleList(
            [ConformerLiteBlock(dim, heads=heads, kernel_size=7, dropout=dropout) for _ in range(int(encoder_layers))]
        )
        self.global_film = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim * 2))
        self.upsample_prior = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.aligner = LocalMelCrossAttention(dim, heads=heads, dropout=dropout, window=align_window)
        self.align_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.ModuleList(
            [ConformerLiteBlock(dim, heads=heads, kernel_size=9, dropout=dropout) for _ in range(int(decoder_layers))]
        )
        self.detail_refine = nn.ModuleList([TemporalConvBlock(dim, d, dropout) for d in (1, 2, 4)])
        self.out_norm = nn.LayerNorm(dim)
        self.mel_base_head = nn.Linear(dim, 80)
        self.mel_detail_head = nn.Linear(dim, 80)
        self.energy_head = nn.Linear(dim, 1)
        self.detail_scale = float(detail_scale)
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
        z = self.fusion(z_vis, z_geo)
        for block in self.encoder:
            z = block(z, video_mask)

        summary = torch.cat([masked_mean(z, video_mask, 1), masked_std(z, video_mask, 1)], dim=-1)
        scale, shift = self.global_film(summary).chunk(2, dim=-1)
        z = z * (1.0 + 0.5 * torch.tanh(scale).unsqueeze(1)) + 0.5 * shift.unsqueeze(1)
        if video_mask is not None:
            z = z * video_mask.to(z.device, z.dtype).unsqueeze(-1)

        if mel_mask is not None:
            target_len = int(mel_mask.shape[1])
        elif mel is not None:
            target_len = int(mel.shape[1])
        else:
            target_len = int(round(z.shape[1] * 2.5))
        prior = self.upsample_prior(z.transpose(1, 2))
        if prior.shape[2] != target_len:
            prior = F.interpolate(prior, size=target_len, mode="linear", align_corners=False)
        prior = prior.transpose(1, 2).contiguous()
        h_align = self.aligner(z, batch.get("video_times"), batch.get("mel_times"), video_mask, mel_mask, target_len)
        h = self.align_fusion(torch.cat([prior, h_align], dim=-1))
        if mel_mask is not None:
            h = h * mel_mask.to(h.device, h.dtype).unsqueeze(-1)
        for block in self.decoder:
            h = block(h, mel_mask)
        detail = h
        for block in self.detail_refine:
            detail = block(detail, mel_mask)
        h = self.out_norm(h)
        detail = self.out_norm(detail)
        out = self.mel_base_head(h) + self.detail_scale * torch.tanh(self.mel_detail_head(detail))
        if mel_mask is not None:
            out = out * mel_mask.to(out.device, out.dtype).unsqueeze(-1)
        if bool(batch.get("return_aux", False)):
            aux = {"mel": out}
            if self.unit_head is not None:
                aux["unit_logits"] = self.unit_head(h)
            return aux
        return out
