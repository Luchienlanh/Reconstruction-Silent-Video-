from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectTCNMelDecoder(nn.Module):
    """Direct latent-to-mel temporal convolution decoder."""

    def __init__(
        self,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 80,
        num_layers: int = 6,
        dropout: float = 0.0,
        output_bias_init: float = -4.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(condition_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 4)
            padding = dilation * 2
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "conv": nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=5,
                            padding=padding,
                            dilation=dilation,
                        ),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.SiLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                    }
                )
            )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, output_bias_init)

    def forward(self, condition: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if target_len is not None and condition.shape[1] != target_len:
            condition = F.interpolate(
                condition.transpose(1, 2),
                size=int(target_len),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()

        x = self.input_proj(condition)
        for block in self.blocks:
            y = block["norm"](x).transpose(1, 2)
            y = block["conv"](y).transpose(1, 2)
            x = x + F.silu(y)
            x = x + block["ffn"](x)
        return self.output(self.output_norm(x))
