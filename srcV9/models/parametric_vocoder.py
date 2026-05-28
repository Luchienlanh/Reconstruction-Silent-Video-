from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from srcV3.models.window_model import R2Plus1DVisualEncoder

from .landmark_ctc import LandmarkMotionEncoder
from .source_filter_vocoder import SourceFilterBlock


def _template_bank(n_mels: int) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.linspace(0.0, 1.0, int(n_mels))
    voiced = 1.2 * torch.exp(-((pos - 0.25) / 0.28).pow(2)) - 0.45 * pos
    voiced = voiced + 0.12 * torch.sin(2.0 * torch.pi * 7.0 * pos)
    unvoiced = 1.3 * pos - 0.45 + 0.35 * torch.exp(-((pos - 0.78) / 0.18).pow(2))
    voiced = (voiced - voiced.mean()) / voiced.std(unbiased=False).clamp_min(1e-6) * 0.35
    unvoiced = (unvoiced - unvoiced.mean()) / unvoiced.std(unbiased=False).clamp_min(1e-6) * 0.35
    return voiced.float(), unvoiced.float()


class VideoParametricVocoderModel(nn.Module):
    """Predict parametric source-filter controls instead of a full source spectrogram.

    Components:
      - filter/envelope shape from landmarks
      - frame energy from video
      - voicing probability from video
      - broad excitation/noise bands from video

    The final log-mel is assembled by a deterministic source-filter rule and
    can be inverted with inverse-mel + Griffin-Lim.
    """

    def __init__(
        self,
        num_points: int = 40,
        dim: int = 384,
        n_mels: int = 80,
        source_bands: int = 8,
        visual_width: int = 24,
        visual_layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        visual_temporal_layers: int = 1,
        landmark_tcn_layers: int = 6,
        landmark_transformer_layers: int = 2,
        nhead: int = 6,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        source_scale_init: float = 0.35,
    ):
        super().__init__()
        self.n_mels = int(n_mels)
        self.source_bands = int(source_bands)
        self.visual = R2Plus1DVisualEncoder(
            dim=dim,
            width=visual_width,
            layers=visual_layers,
            spatial_pool_size=2,
            temporal_layers=visual_temporal_layers,
            dropout=dropout,
        )
        self.landmarks = LandmarkMotionEncoder(
            num_points=num_points,
            dim=dim,
            tcn_layers=landmark_tcn_layers,
            transformer_layers=landmark_transformer_layers,
            nhead=nhead,
            dropout=dropout,
        )
        self.filter_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        self.source_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        dilations = [1, 2, 4, 8]
        self.filter_blocks = nn.ModuleList(
            [SourceFilterBlock(dim, dilation=dilations[i % len(dilations)], dropout=dropout) for i in range(max(1, decoder_layers // 2))]
        )
        self.source_blocks = nn.ModuleList(
            [SourceFilterBlock(dim, dilation=dilations[i % len(dilations)], dropout=dropout) for i in range(decoder_layers)]
        )
        self.filter_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.filter_head = nn.Linear(dim, self.n_mels)
        self.energy_head = nn.Linear(dim, 1)
        self.voicing_head = nn.Linear(dim, 1)
        self.bands_head = nn.Linear(dim, self.source_bands)
        init_scale = max(1e-4, min(0.95, float(source_scale_init)))
        self.source_scale_logit = nn.Parameter(torch.logit(torch.tensor(init_scale)))
        voiced, unvoiced = _template_bank(self.n_mels)
        self.register_buffer("voiced_template", voiced.view(1, 1, -1), persistent=False)
        self.register_buffer("unvoiced_template", unvoiced.view(1, 1, -1), persistent=False)
        with torch.no_grad():
            self.filter_head.weight.zero_()
            self.filter_head.bias.zero_()
            self.energy_head.weight.zero_()
            self.energy_head.bias.zero_()
            self.bands_head.weight.zero_()
            self.bands_head.bias.zero_()
            self.voicing_head.bias.zero_()

    def set_energy_bias(self, mel_mean: torch.Tensor) -> None:
        with torch.no_grad():
            self.energy_head.bias.fill_(float(mel_mean.float().mean().item()))

    def source_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.source_scale_logit).clamp(0.0, 1.0)

    @staticmethod
    def _upsample_time(memory: torch.Tensor, target_len: int) -> torch.Tensor:
        if memory.shape[1] == int(target_len):
            return memory
        return F.interpolate(memory.transpose(1, 2), size=int(target_len), mode="linear", align_corners=False).transpose(1, 2)

    def _bands_to_mel(self, bands: torch.Tensor) -> torch.Tensor:
        batch, frames, n_bands = bands.shape
        x = F.interpolate(
            bands.reshape(batch * frames, 1, n_bands),
            size=self.n_mels,
            mode="linear",
            align_corners=False,
        ).reshape(batch, frames, self.n_mels)
        return x - x.mean(dim=-1, keepdim=True)

    def forward(self, batch: dict[str, torch.Tensor], target_len: int | None = None) -> dict[str, torch.Tensor]:
        if target_len is None:
            target = batch.get("mel")
            if target is not None:
                target_len = int(target.shape[1])
        if target_len is None:
            raise ValueError("target_len is required when batch has no mel target.")
        video_mask = batch.get("landmark_mask")
        visual = self.visual(batch["video"], video_mask=video_mask)
        landmark = self.landmarks(
            batch["landmarks"],
            video_times=batch.get("video_times"),
            landmark_mask=batch.get("landmark_mask"),
        )
        visual = self._upsample_time(visual, int(target_len))
        landmark = self._upsample_time(landmark, int(target_len))
        mel_mask = batch.get("mel_mask")

        filt = self.filter_in(landmark)
        for block in self.filter_blocks:
            filt = block(filt, mel_mask)
        filt = self.filter_norm(filt)
        filter_shape = self.filter_head(filt)
        filter_shape = filter_shape - filter_shape.mean(dim=-1, keepdim=True)

        src = self.source_in(visual)
        for block in self.source_blocks:
            src = block(src, mel_mask)
        src = self.source_norm(src)
        energy = self.energy_head(src)
        voicing_logit = self.voicing_head(src)
        voicing = torch.sigmoid(voicing_logit)
        bands = self.bands_head(src)
        broad_source = self._bands_to_mel(bands)
        template = voicing * self.voiced_template.to(voicing.dtype) + (1.0 - voicing) * self.unvoiced_template.to(voicing.dtype)
        source = template + self.source_scale().to(broad_source.dtype) * broad_source
        mel = energy + filter_shape + source
        if mel_mask is not None:
            mask = mel_mask.unsqueeze(-1).to(mel.dtype)
            filter_shape = filter_shape * mask
            energy = energy * mask
            voicing = voicing * mask
            voicing_logit = voicing_logit * mask
            broad_source = broad_source * mask
            source = source * mask
            mel = mel * mask
        return {
            "mel": mel,
            "filter": filter_shape,
            "energy": energy,
            "voicing": voicing,
            "voicing_logit": voicing_logit,
            "broad_source": broad_source,
            "source": source,
            "source_scale": self.source_scale(),
        }
