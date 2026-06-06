from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=5, padding=2 * dilation, dilation=dilation)
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
            x = x * mask.to(x.device, x.dtype).unsqueeze(-1)
        return x


class LipTextCTCModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        dim: int = 512,
        tcn_layers: int = 4,
        transformer_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        upsample_factor: int = 2,
        blank_bias_init: float = -3.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.upsample_factor = max(1, int(upsample_factor))
        self.input = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        dilations = [1, 2, 4, 8]
        self.tcn = nn.ModuleList([TCNBlock(dim, dilations[i % len(dilations)], dropout) for i in range(int(tcn_layers))])
        if int(transformer_layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=max(1, int(nhead)),
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        else:
            self.temporal = None
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, self.vocab_size))
        with torch.no_grad():
            linear = self.head[-1]
            if isinstance(linear, nn.Linear) and linear.bias is not None and linear.bias.numel() > 0:
                linear.bias.zero_()
                linear.bias[0] = float(blank_bias_init)

    def forward(self, features: torch.Tensor, feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.upsample_factor > 1:
            size = int(features.shape[1] * self.upsample_factor)
            features = F.interpolate(features.transpose(1, 2), size=size, mode="linear", align_corners=False).transpose(1, 2)
            if feature_mask is not None:
                feature_mask = feature_mask.repeat_interleave(self.upsample_factor, dim=1)
        x = self.input(torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0))
        for block in self.tcn:
            x = block(x, feature_mask)
        if self.temporal is not None:
            key_padding_mask = None if feature_mask is None else ~feature_mask.to(x.device, dtype=torch.bool)
            x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        if feature_mask is not None:
            x = x * feature_mask.to(x.device, x.dtype).unsqueeze(-1)
        return self.head(x)

    def output_lengths(self, feature_lengths: torch.Tensor) -> torch.Tensor:
        return feature_lengths.long() * self.upsample_factor


def build_model_from_config(config: dict) -> LipTextCTCModel:
    return LipTextCTCModel(
        input_dim=int(config.get("input_dim", 768)),
        vocab_size=int(config.get("vocab_size", 40)),
        dim=int(config.get("dim", 512)),
        tcn_layers=int(config.get("tcn_layers", 4)),
        transformer_layers=int(config.get("transformer_layers", 4)),
        nhead=int(config.get("nhead", 8)),
        dropout=0.0,
        upsample_factor=int(config.get("upsample_factor", 2)),
        blank_bias_init=float(config.get("blank_bias_init", -3.0)),
    )
