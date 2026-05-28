from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from srcV3.models.window_model import R2Plus1DVisualEncoder

from .landmark_ctc import LandmarkMotionEncoder
from .source_filter_vocoder import SourceFilterBlock


class VideoSourceFilterVocoderModel(nn.Module):
    """Source-filter vocoder target with video frames for the source branch.

    Landmarks predict the smooth filter/envelope; visual mouth frames predict
    the source/excitation term. In log-mel space:

        final_logmel = envelope_from_landmarks + source_from_video
    """

    def __init__(
        self,
        num_points: int = 40,
        dim: int = 384,
        n_mels: int = 80,
        source_bands: int = 16,
        visual_width: int = 24,
        visual_layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        visual_temporal_layers: int = 1,
        landmark_tcn_layers: int = 6,
        landmark_transformer_layers: int = 2,
        nhead: int = 6,
        decoder_layers: int = 6,
        dropout: float = 0.1,
        output_bias_init: float = -4.0,
        source_scale_init: float = 0.5,
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
        self.env_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        self.src_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        dilations = [1, 2, 4, 8]
        self.env_decoder = nn.ModuleList(
            [SourceFilterBlock(dim, dilation=dilations[i % len(dilations)], dropout=dropout) for i in range(max(1, decoder_layers // 2))]
        )
        self.src_decoder = nn.ModuleList(
            [SourceFilterBlock(dim, dilation=dilations[i % len(dilations)], dropout=dropout) for i in range(decoder_layers)]
        )
        self.env_norm = nn.LayerNorm(dim)
        self.src_norm = nn.LayerNorm(dim)
        self.envelope_head = nn.Linear(dim, self.n_mels)
        self.source_head = nn.Linear(dim, self.source_bands)
        self.source_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        init_scale = max(1e-4, min(0.95, float(source_scale_init)))
        self.source_scale_logit = nn.Parameter(torch.logit(torch.tensor(init_scale)))
        with torch.no_grad():
            self.envelope_head.bias.fill_(float(output_bias_init))
            self.source_head.weight.mul_(0.1)
            self.source_head.bias.zero_()
            self.source_gate[-1].bias.fill_(1.0)

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        with torch.no_grad():
            self.envelope_head.bias.copy_(mel_mean.to(device=self.envelope_head.bias.device, dtype=self.envelope_head.bias.dtype))

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

        env = self.env_in(landmark)
        for block in self.env_decoder:
            env = block(env, mel_mask)
        env = self.env_norm(env)
        envelope = self.envelope_head(env)

        src = self.src_in(visual)
        for block in self.src_decoder:
            src = block(src, mel_mask)
        src = self.src_norm(src)
        source_bands = self.source_head(src)
        source_mel = self._source_to_mel(source_bands)
        gate = torch.ones_like(self.source_gate(src))
        scale = self.source_scale().to(dtype=envelope.dtype)
        source = scale * source_mel
        mel = envelope + source

        if mel_mask is not None:
            mask = mel_mask.unsqueeze(-1).to(mel.dtype)
            envelope = envelope * mask
            source = source * mask
            source_bands = source_bands * mask
            gate = gate * mask
            mel = mel * mask
        return {
            "mel": mel,
            "envelope": envelope,
            "source": source,
            "source_bands": source_bands,
            "source_gate": gate,
            "source_scale": scale,
        }
