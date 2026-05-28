from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .landmark_ctc import LandmarkMotionEncoder


class SourceFilterBlock(nn.Module):
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


class LandmarkSourceFilterVocoderModel(nn.Module):
    """Predict source-filter components for Griffin-Lim synthesis.

    In magnitude spectra, a classical source-filter model is roughly:
        speech_mag = source_mag * filter_envelope

    In log-mel space this becomes an additive decomposition:
        final_logmel = envelope_logmel + source_logmel

    The source branch is intentionally low-rank over frequency. This prevents
    it from becoming an unconstrained second mel decoder and keeps the model
    closer to source-filter vocoder behavior.
    """

    def __init__(
        self,
        num_points: int = 40,
        dim: int = 384,
        n_mels: int = 80,
        source_bands: int = 16,
        tcn_layers: int = 6,
        transformer_layers: int = 2,
        nhead: int = 6,
        decoder_layers: int = 6,
        dropout: float = 0.1,
        output_bias_init: float = -4.0,
        source_scale_init: float = 0.5,
    ):
        super().__init__()
        self.n_mels = int(n_mels)
        self.source_bands = int(source_bands)
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
            [SourceFilterBlock(dim, dilation=dilations[i % len(dilations)], dropout=dropout) for i in range(decoder_layers)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.envelope_head = nn.Linear(dim, self.n_mels)
        self.source_head = nn.Linear(dim, self.source_bands)
        self.source_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.energy_head = nn.Linear(dim, 1)
        init_scale = max(1e-4, min(0.95, float(source_scale_init)))
        self.source_scale_logit = nn.Parameter(torch.logit(torch.tensor(init_scale)))
        with torch.no_grad():
            self.envelope_head.bias.fill_(float(output_bias_init))
            self.source_head.weight.mul_(0.1)
            self.source_head.bias.zero_()
            self.source_gate[-1].bias.fill_(1.0)
            self.energy_head.bias.fill_(float(output_bias_init))

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        with torch.no_grad():
            self.envelope_head.bias.copy_(mel_mean.to(device=self.envelope_head.bias.device, dtype=self.envelope_head.bias.dtype))
            self.energy_head.bias.fill_(float(mel_mean.mean().item()))

    def source_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.source_scale_logit).clamp(0.0, 1.0)

    @staticmethod
    def _upsample_time(memory: torch.Tensor, target_len: int) -> torch.Tensor:
        if memory.shape[1] == int(target_len):
            return memory
        return F.interpolate(memory.transpose(1, 2), size=int(target_len), mode="linear", align_corners=False).transpose(1, 2)

    def _source_to_mel(self, source_bands: torch.Tensor) -> torch.Tensor:
        batch, frames, bands = source_bands.shape
        source = F.interpolate(
            source_bands.reshape(batch * frames, 1, bands),
            size=self.n_mels,
            mode="linear",
            align_corners=False,
        ).reshape(batch, frames, self.n_mels)
        return source - source.mean(dim=-1, keepdim=True)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            batch["landmarks"],
            video_times=batch.get("video_times"),
            landmark_mask=batch.get("landmark_mask"),
        )

    def forward(self, batch: dict[str, torch.Tensor], target_len: int | None = None) -> dict[str, torch.Tensor]:
        if target_len is None:
            target = batch.get("mel")
            if target is not None:
                target_len = int(target.shape[1])
        if target_len is None:
            raise ValueError("target_len is required when batch has no mel target.")
        x = self._upsample_time(self.encode(batch), int(target_len))
        x = self.decoder_in(x)
        mel_mask = batch.get("mel_mask")
        for block in self.decoder:
            x = block(x, mel_mask)
        x = self.out_norm(x)
        envelope = self.envelope_head(x)
        source_bands = self.source_head(x)
        source_mel = self._source_to_mel(source_bands)
        gate = torch.sigmoid(self.source_gate(x))
        scale = self.source_scale().to(dtype=envelope.dtype)
        source = scale * gate * source_mel
        mel = envelope + source
        energy = self.energy_head(x)
        if mel_mask is not None:
            mask = mel_mask.unsqueeze(-1).to(mel.dtype)
            envelope = envelope * mask
            source = source * mask
            source_bands = source_bands * mask
            gate = gate * mask
            mel = mel * mask
            energy = energy * mask
        return {
            "mel": mel,
            "envelope": envelope,
            "source": source,
            "source_bands": source_bands,
            "source_gate": gate,
            "source_scale": scale,
            "energy": energy,
        }
