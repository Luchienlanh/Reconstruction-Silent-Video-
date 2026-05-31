from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_sync_model import ConformerLiteBlock, LipGeometryTower, masked_mean, masked_std
from .motion_tcn_model import MotionVisualTower, TemporalConvBlock


class ContextDetailLipToSpeechModel(nn.Module):
    """ContextSync backbone plus a residual motion/detail mel head."""

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
        detail_scale: float = 0.35,
        detail_layers: int = 3,
    ):
        super().__init__()
        if num_points != 40:
            raise ValueError("ContextDetailLipToSpeechModel currently expects 40 lip landmarks.")
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
        self.motion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        dilations = [1, 2, 4, 8]
        self.detail_refine = nn.ModuleList(
            [TemporalConvBlock(dim, dilations[i % len(dilations)], dropout) for i in range(max(1, int(detail_layers)))]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.detail_norm = nn.LayerNorm(dim)
        self.mel_base_head = nn.Linear(dim, 80)
        self.mel_detail_head = nn.Linear(dim, 80)
        self.energy_head = nn.Linear(dim, 1)
        self.detail_scale = float(detail_scale)
        self.num_units = int(num_units)
        self.unit_head = nn.Linear(dim, self.num_units) if self.num_units > 0 else None

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= 1:
            return torch.zeros_like(x)
        d = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)

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
        base_h = self.out_norm(h)

        detail = self.motion_proj(torch.cat([h, self._delta(h)], dim=-1))
        for block in self.detail_refine:
            detail = block(detail, mel_mask)
        detail = self.detail_norm(detail)

        out = self.mel_base_head(base_h) + self.detail_scale * torch.tanh(self.mel_detail_head(detail))
        if mel_mask is not None:
            out = out * mel_mask.to(out.device, out.dtype).unsqueeze(-1)
        if bool(batch.get("return_aux", False)):
            aux = {"mel": out}
            if self.unit_head is not None:
                aux["unit_logits"] = self.unit_head(base_h)
            return aux
        return out


class ContextMotionDetailLipToSpeechModel(ContextDetailLipToSpeechModel):
    """ContextSync backbone with a visual-motion-only residual detail head."""

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
        detail_scale: float = 0.35,
        detail_layers: int = 3,
    ):
        super().__init__(
            dim=dim,
            spatial_tokens=spatial_tokens,
            num_points=num_points,
            dropout=dropout,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            heads=heads,
            num_units=num_units,
            detail_scale=detail_scale,
            detail_layers=detail_layers,
        )
        self.detail_upsample = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.detail_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(32, dim // 4)),
            nn.SiLU(),
            nn.Linear(max(32, dim // 4), 1),
        )
        nn.init.zeros_(self.mel_detail_head.weight)
        nn.init.zeros_(self.mel_detail_head.bias)
        nn.init.zeros_(self.detail_gate[-1].weight)
        nn.init.constant_(self.detail_gate[-1].bias, -1.0)

    @staticmethod
    def _upsample_to(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.shape[1] == target_len:
            return x
        y = F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return y.transpose(1, 2).contiguous()

    def _motion_amp(self, z_vis: torch.Tensor, target_len: int) -> torch.Tensor:
        amp = self._delta(z_vis).pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        amp = self._upsample_to(amp, target_len)
        denom = amp.detach().mean(dim=1, keepdim=True).clamp_min(0.05)
        return (amp / denom).clamp(0.0, 2.0)

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
            mask = video_mask.to(z.device, z.dtype).unsqueeze(-1)
            z = z * mask
            z_vis = z_vis * mask

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
        base_h = self.out_norm(h)
        base_mel = self.mel_base_head(base_h)

        motion = self.motion_proj(torch.cat([z_vis, self._delta(z_vis)], dim=-1))
        detail = self.detail_upsample(motion.transpose(1, 2))
        if detail.shape[2] != target_len:
            detail = F.interpolate(detail, size=target_len, mode="linear", align_corners=False)
        detail = detail.transpose(1, 2).contiguous()
        for block in self.detail_refine:
            detail = block(detail, mel_mask)
        detail = self.detail_norm(detail)

        motion_amp = self._motion_amp(z_vis, target_len)
        detail_gate = torch.sigmoid(self.detail_gate(detail))
        detail_mel = torch.tanh(self.mel_detail_head(detail)) * detail_gate * motion_amp
        out = base_mel + self.detail_scale * detail_mel
        if mel_mask is not None:
            out = out * mel_mask.to(out.device, out.dtype).unsqueeze(-1)
        if bool(batch.get("return_aux", False)):
            aux = {
                "mel": out,
                "base_mel": base_mel,
                "detail_mel": detail_mel,
                "detail_gate": detail_gate,
                "motion_amp": motion_amp,
            }
            if self.unit_head is not None:
                aux["unit_logits"] = self.unit_head(base_h)
            return aux
        return out
