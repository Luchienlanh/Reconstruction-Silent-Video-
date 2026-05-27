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
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class FeatureCTCModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        dim: int = 512,
        tcn_layers: int = 4,
        transformer_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        upsample_factor: int = 1,
        blank_bias_init: float = -2.0,
    ):
        super().__init__()
        self.upsample_factor = max(1, int(upsample_factor))
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        dilations = [1, 2, 4, 8]
        self.tcn = nn.ModuleList([TCNBlock(dim, dilations[i % len(dilations)], dropout) for i in range(tcn_layers)])
        if transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        else:
            self.temporal = None
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, int(vocab_size)))
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
        x = self.input(features)
        for block in self.tcn:
            x = block(x, feature_mask)
        if self.temporal is not None:
            key_padding_mask = None if feature_mask is None else ~feature_mask.bool()
            x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        return self.head(x)

    def output_lengths(self, feature_lengths: torch.Tensor) -> torch.Tensor:
        return feature_lengths.long() * self.upsample_factor

