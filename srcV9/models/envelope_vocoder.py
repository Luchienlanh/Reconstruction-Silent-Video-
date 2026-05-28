from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .landmark_ctc import LandmarkMotionEncoder


class EnvelopeTCNBlock(nn.Module):
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


class LandmarkEnvelopeVocoderModel(nn.Module):
    """Predict a smooth log-mel spectral envelope from lip landmarks.

    This is intentionally a classical-vocoder target, not a neural vocoder:
    the model predicts the envelope; a separate synthesis step supplies a
    carrier/excitation and applies the envelope.
    """

    def __init__(
        self,
        num_points: int = 40,
        dim: int = 384,
        n_mels: int = 80,
        tcn_layers: int = 6,
        transformer_layers: int = 2,
        nhead: int = 6,
        decoder_layers: int = 6,
        dropout: float = 0.1,
        output_bias_init: float = -4.0,
        residual_alpha_init: float = 0.25,
        enable_residual: bool = True,
    ):
        super().__init__()
        self.enable_residual = bool(enable_residual)
        self.encoder = LandmarkMotionEncoder(
            num_points=num_points,
            dim=dim,
            tcn_layers=tcn_layers,
            transformer_layers=transformer_layers,
            nhead=nhead,
            dropout=dropout,
        )
        self.decoder_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        dilations = [1, 2, 4, 8]
        self.decoder = nn.ModuleList(
            [EnvelopeTCNBlock(dim, dilation=dilations[i % len(dilations)], dropout=dropout) for i in range(decoder_layers)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.envelope_head = nn.Linear(dim, n_mels)
        self.residual_head = nn.Linear(dim, n_mels)
        self.energy_head = nn.Linear(dim, 1)
        init_alpha = max(1e-4, min(0.95, float(residual_alpha_init)))
        self.residual_alpha_logit = nn.Parameter(torch.logit(torch.tensor(init_alpha)))
        with torch.no_grad():
            self.envelope_head.bias.fill_(float(output_bias_init))
            self.residual_head.weight.zero_()
            self.residual_head.bias.zero_()
            self.energy_head.bias.fill_(float(output_bias_init))

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        with torch.no_grad():
            self.envelope_head.bias.copy_(mel_mean.to(device=self.envelope_head.bias.device, dtype=self.envelope_head.bias.dtype))
            self.energy_head.bias.fill_(float(mel_mean.mean().item()))

    def residual_alpha(self) -> torch.Tensor:
        if not self.enable_residual:
            return self.residual_alpha_logit.new_tensor(0.0)
        return torch.sigmoid(self.residual_alpha_logit).clamp(0.0, 1.0)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            batch["landmarks"],
            video_times=batch.get("video_times"),
            landmark_mask=batch.get("landmark_mask"),
        )

    @staticmethod
    def _upsample_to_mel(memory: torch.Tensor, mel_len: int) -> torch.Tensor:
        if memory.shape[1] == int(mel_len):
            return memory
        return F.interpolate(memory.transpose(1, 2), size=int(mel_len), mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, batch: dict[str, torch.Tensor], target_len: int | None = None) -> dict[str, torch.Tensor]:
        if target_len is None:
            target = batch.get("target_mel")
            if target is None:
                target = batch.get("mel")
            if target is not None:
                target_len = int(target.shape[1])
        if target_len is None:
            raise ValueError("target_len is required when batch has no mel target.")
        x = self._upsample_to_mel(self.encode(batch), int(target_len))
        x = self.decoder_in(x)
        mel_mask = batch.get("mel_mask")
        for block in self.decoder:
            x = block(x, mel_mask)
        x = self.out_norm(x)
        envelope = self.envelope_head(x)
        residual = self.residual_head(x)
        alpha = self.residual_alpha().to(dtype=envelope.dtype)
        mel = envelope + alpha * residual
        energy = self.energy_head(x)
        if mel_mask is not None:
            mel = mel * mel_mask.unsqueeze(-1).to(mel.dtype)
            envelope = envelope * mel_mask.unsqueeze(-1).to(envelope.dtype)
            residual = residual * mel_mask.unsqueeze(-1).to(residual.dtype)
            energy = energy * mel_mask.unsqueeze(-1).to(energy.dtype)
        return {"mel": mel, "envelope": envelope, "residual": residual, "residual_alpha": alpha, "energy": energy}
