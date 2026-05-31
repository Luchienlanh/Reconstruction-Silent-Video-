from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_sync_model import ConformerLiteBlock, LipGeometryTower
from .motion_tcn_model import MotionVisualTower


class ContentUnitLipToSpeechModel(nn.Module):
    """Content-first lip-to-speech model.

    The main path predicts discrete speech units from video. The mel decoder is
    conditioned on the unit posterior/teacher units so reconstruction loss cannot
    dominate the visual encoder as easily as in direct mel models.
    """

    def __init__(
        self,
        dim: int = 256,
        spatial_tokens: int = 2,
        num_points: int = 40,
        dropout: float = 0.05,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        heads: int = 4,
        num_units: int = 50,
        unit_temperature: float = 1.0,
        detach_unit_condition: bool = True,
        detach_content_hidden: bool = True,
        unit_teacher_prob: float = 0.0,
    ):
        super().__init__()
        if num_points != 40:
            raise ValueError("ContentUnitLipToSpeechModel currently expects 40 lip landmarks.")
        if num_units <= 0:
            raise ValueError("ContentUnitLipToSpeechModel requires num_units > 0.")
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
        self.unit_upsample = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.unit_refine = nn.ModuleList(
            [ConformerLiteBlock(dim, heads=heads, kernel_size=9, dropout=dropout) for _ in range(max(1, int(encoder_layers)))]
        )
        self.unit_norm = nn.LayerNorm(dim)
        self.unit_head = nn.Linear(dim, int(num_units))
        self.unit_embedding = nn.Embedding(int(num_units), dim)

        self.content_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.ModuleList(
            [ConformerLiteBlock(dim, heads=heads, kernel_size=9, dropout=dropout) for _ in range(int(decoder_layers))]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.mel_head = nn.Linear(dim, 80)
        self.energy_head = nn.Linear(dim, 1)

        self.num_units = int(num_units)
        self.unit_temperature = float(unit_temperature)
        self.detach_unit_condition = bool(detach_unit_condition)
        self.detach_content_hidden = bool(detach_content_hidden)
        self.unit_teacher_prob = float(unit_teacher_prob)

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= 1:
            return torch.zeros_like(x)
        d = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)

    @staticmethod
    def _target_len(batch: dict[str, torch.Tensor], fallback_len: int) -> int:
        if batch.get("mel_mask") is not None:
            return int(batch["mel_mask"].shape[1])
        if batch.get("mel") is not None:
            return int(batch["mel"].shape[1])
        return int(fallback_len)

    def _resize_time(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.shape[1] == target_len:
            return x
        y = F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return y.transpose(1, 2).contiguous()

    def _unit_targets(self, batch: dict[str, torch.Tensor], target_len: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        targets = batch.get("speech_units")
        if targets is None:
            return None
        targets = targets.long()
        if targets.shape[1] != target_len:
            x = targets.float().unsqueeze(1)
            targets = F.interpolate(x, size=target_len, mode="nearest").squeeze(1).long()
        mask = targets.ge(0)
        if batch.get("mel_mask") is not None and batch["mel_mask"].shape[1] == target_len:
            mask = mask & batch["mel_mask"].to(mask.device, dtype=torch.bool)
        return targets.clamp(0, self.num_units - 1), mask

    def _unit_condition(self, logits: torch.Tensor, h: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target_len = int(logits.shape[1])
        use_teacher = False
        if self.training and self.unit_teacher_prob > 0.0 and batch.get("speech_units") is not None:
            use_teacher = bool(torch.rand((), device=logits.device) < self.unit_teacher_prob)
        if use_teacher:
            target_info = self._unit_targets(batch, target_len)
            if target_info is not None:
                targets, mask = target_info
                emb = self.unit_embedding(targets)
                return emb * mask.to(emb.device, emb.dtype).unsqueeze(-1)

        probs = torch.softmax(logits.float() / max(0.1, self.unit_temperature), dim=-1).to(h.dtype)
        if self.detach_unit_condition:
            probs = probs.detach()
        return probs.matmul(self.unit_embedding.weight.to(device=probs.device, dtype=probs.dtype))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        video = batch["video"]
        landmarks = batch["landmarks"]
        video_mask = batch.get("video_mask")
        mel_mask = batch.get("mel_mask")

        z_vis = self.visual(video)
        z_geo = self.geometry(landmarks, video_mask)
        z = self.fusion(torch.cat([z_vis, z_geo], dim=-1))
        for block in self.encoder:
            z = block(z, video_mask)
        if video_mask is not None:
            z = z * video_mask.to(z.device, z.dtype).unsqueeze(-1)

        x = self.unit_upsample(z.transpose(1, 2)).transpose(1, 2).contiguous()
        target_len = self._target_len(batch, fallback_len=x.shape[1])
        h = self._resize_time(x, target_len)
        for block in self.unit_refine:
            h = block(h, mel_mask)
        h = self.unit_norm(h)
        unit_logits = self.unit_head(h)

        unit_content = self._unit_condition(unit_logits, h, batch)
        h_content = h.detach() if self.detach_content_hidden else h
        mel_h = self.content_fusion(torch.cat([unit_content, h_content, self._delta(h_content)], dim=-1))
        for block in self.decoder:
            mel_h = block(mel_h, mel_mask)
        mel_h = self.out_norm(mel_h)
        mel = self.mel_head(mel_h)
        if mel_mask is not None:
            mel = mel * mel_mask.to(mel.device, mel.dtype).unsqueeze(-1)
        if bool(batch.get("return_aux", False)):
            return {
                "mel": mel,
                "unit_logits": unit_logits,
                "unit_content": unit_content,
            }
        return mel
