"""Spiking Neural Network classifier for pose sequences."""

from __future__ import annotations

import torch
from torch import nn

from ..pose_features import NUM_LANDMARKS, LANDMARK_DIM
from .speech_tcn import pose_motion_features, MOTION_DIM


class ConfALIFNode(nn.Module):
    """Adaptive LIF readout with surrogate gradient."""

    def __init__(self, tau_m: float = 2.0, tau_a: float = 1.5, beta: float = 0.1):
        super().__init__()
        from spikingjelly.activation_based import surrogate

        self.tau_m = tau_m
        self.tau_a = tau_a
        self.beta = beta
        self.threshold_base = 1.0
        self.surrogate = surrogate.ATan()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, channels = x.shape
        v_mem = x.new_zeros(bsz, channels)
        a_adapt = x.new_zeros(bsz, channels)
        spikes = []
        membranes = []
        for t in range(steps):
            current = x[:, t]
            a_adapt = a_adapt * (1.0 - 1.0 / self.tau_a)
            threshold = self.threshold_base + a_adapt
            v_mem = v_mem * (1.0 - 1.0 / self.tau_m) + current
            membranes.append(v_mem)
            spike = self.surrogate(v_mem - threshold)
            v_mem = v_mem * (1.0 - spike)
            a_adapt = a_adapt + self.beta * spike
            spikes.append(spike)
        return torch.stack(spikes, dim=1), torch.stack(membranes, dim=1)


class SpikingSequenceBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        from spikingjelly.activation_based import neuron, surrogate

        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.plif1 = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.plif2 = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in range(x.shape[1]):
            y = self.fc1(x[:, t])
            y = self.plif1(self.norm1(y))
            y = self.dropout(y)
            y = self.fc2(y)
            y = self.plif2(self.norm2(y))
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class SNNPoseClassifier(nn.Module):
    """Lightweight SNN classifier for pose motion sequences."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del input_dim
        self.input = nn.Sequential(
            nn.Linear(MOTION_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.blocks = nn.ModuleList([SpikingSequenceBlock(hidden_size, dropout=dropout) for _ in range(num_layers)])
        self.readout = ConfALIFNode()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from spikingjelly.activation_based import functional

        functional.reset_net(self.blocks)
        x = self.input(pose_motion_features(x))
        for block in self.blocks:
            x = x + block(x)
        _, membrane = self.readout(x)
        pooled = torch.cat([membrane.mean(dim=1), membrane[:, -1]], dim=-1)
        return self.head(pooled)
