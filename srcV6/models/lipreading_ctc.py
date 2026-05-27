from __future__ import annotations

import torch
import torch.nn as nn

from srcV3.models.window_model import Fusion, LandmarkMotionEncoder, R2Plus1DVisualEncoder


def _parse_layers(value) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.split(",") if x.strip()]
        return tuple(parts) if len(parts) == 4 else (1, 1, 1, 1)  # type: ignore[return-value]
    return tuple(value)


class LipreadingCTCModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_landmark_points: int = 40,
        fusion_type: str = "landmark_first",
        encoder_width: int = 32,
        resnet_layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        visual_temporal_layers: int = 1,
        landmark_temporal_layers: int = 1,
        dropout: float = 0.1,
        blank_bias_init: float = -2.0,
    ):
        super().__init__()
        self.visual = R2Plus1DVisualEncoder(
            dim=dim,
            width=encoder_width,
            layers=resnet_layers,
            temporal_layers=visual_temporal_layers,
            dropout=dropout,
        )
        self.landmarks = LandmarkMotionEncoder(
            num_points=num_landmark_points,
            dim=dim,
            temporal_layers=landmark_temporal_layers,
            dropout=dropout,
        )
        self.fusion = Fusion(dim=dim, fusion_type=fusion_type, dropout=dropout)
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, int(vocab_size)),
        )
        with torch.no_grad():
            linear = self.ctc_head[-1]
            if isinstance(linear, nn.Linear) and linear.bias is not None and linear.bias.numel() > 0:
                linear.bias.zero_()
                linear.bias[0] = float(blank_bias_init)
        self.vocab_size = int(vocab_size)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        video_mask = batch.get("video_mask")
        visual = self.visual(batch["video"], video_mask=video_mask)
        landmark_features = self.landmarks(batch["landmarks"], video_mask=video_mask)
        return self.fusion(visual, landmark_features)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.ctc_head(self.encode(batch))


def build_model_from_config(config: dict, vocab_size: int) -> LipreadingCTCModel:
    return LipreadingCTCModel(
        vocab_size=int(vocab_size),
        dim=int(config.get("dim", 512)),
        num_landmark_points=int(config.get("num_landmark_points", 40)),
        fusion_type=str(config.get("fusion_type", "landmark_first")),
        encoder_width=int(config.get("encoder_width", 32)),
        resnet_layers=_parse_layers(config.get("resnet_layers", (1, 1, 1, 1))),
        visual_temporal_layers=int(config.get("visual_temporal_layers", 1)),
        landmark_temporal_layers=int(config.get("landmark_temporal_layers", 1)),
        dropout=float(config.get("dropout", 0.1)),
        blank_bias_init=float(config.get("blank_bias_init", -2.0)),
    )
