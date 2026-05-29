"""Spiking Spatio-Temporal Graph Convolutional Network for fall detection.

Architecture overview:
    Input [B, T, 132] (normalized MediaPipe pose)
    → reshape to [B, T, 33, 4]
    → per-joint motion features [B, T, 33, 10]
    → Linear projection [B, T, 33, hidden]
    → Stack of SpikingGCNBlock × num_layers
        ├─ SpatialGraphConv on 33-joint skeleton graph
        ├─ TemporalConv (dilated 1D) on time axis
        └─ PLIF spiking neuron + residual
    → Joint pooling [B, T, hidden]
    → ALIF readout → membrane [B, T, hidden]
    → Temporal pooling (mean + last) [B, hidden×2]
    → Classification head → [B, num_classes]
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..pose_features import NUM_LANDMARKS, LANDMARK_DIM

# ---------------------------------------------------------------------------
# MediaPipe Pose skeleton graph (33 joints)
# ---------------------------------------------------------------------------

MEDIAPIPE_EDGES: list[tuple[int, int]] = [
    # Face (right side)
    (0, 1), (1, 2), (2, 3), (3, 7),
    # Face (left side)
    (0, 4), (4, 5), (5, 6), (6, 8),
    # Mouth
    (9, 10),
    # Shoulders
    (11, 12),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),
]

NUM_JOINTS = NUM_LANDMARKS  # 33


def build_adjacency(num_joints: int = NUM_JOINTS,
                    edges: list[tuple[int, int]] = MEDIAPIPE_EDGES,
                    self_loop: bool = True) -> torch.Tensor:
    """Build a symmetric normalized adjacency matrix.

    Returns:
        Float tensor of shape [num_joints, num_joints].
        Normalized as D^{-1/2} A D^{-1/2}.
    """
    A = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loop:
        A += torch.eye(num_joints, dtype=torch.float32)
    # Symmetric normalization
    D = A.sum(dim=1)
    D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diff(x: torch.Tensor) -> torch.Tensor:
    """Temporal finite difference with zero-padding at t=0."""
    first = torch.zeros_like(x[:, :1])
    return torch.cat([first, x[:, 1:] - x[:, :-1]], dim=1)


FEATURE_SETS = ("position", "pos_vel", "pos_vel_acc", "full")


def feature_dim_for_set(feature_set: str) -> int:
    if feature_set == "position":
        return 3
    if feature_set == "pos_vel":
        return 6
    if feature_set == "pos_vel_acc":
        return 9
    if feature_set == "full":
        return 10
    raise ValueError(f"Unknown feature_set: {feature_set!r}. Choose from {FEATURE_SETS}.")


def per_joint_motion_features(x: torch.Tensor, feature_set: str = "full") -> torch.Tensor:
    """Create per-joint motion features from normalized pose.

    Args:
        x: [B, T, 132] normalized MediaPipe pose.

    Returns:
        Tensor shaped [B, T, 33, C], depending on feature_set:
        position: 3 pos
        pos_vel: 3 pos + 3 vel
        pos_vel_acc: 3 pos + 3 vel + 3 acc
        full: 3 pos + 3 vel + 3 acc + 1 visibility
    """
    bsz, steps, _ = x.shape
    pose = x.view(bsz, steps, NUM_LANDMARKS, LANDMARK_DIM)
    coords = pose[..., :3]        # [B, T, 33, 3]
    visibility = pose[..., 3:4]   # [B, T, 33, 1]
    vel = _diff(coords)           # [B, T, 33, 3]
    acc = _diff(vel)              # [B, T, 33, 3]
    if feature_set == "position":
        return coords
    if feature_set == "pos_vel":
        return torch.cat([coords, vel], dim=-1)
    if feature_set == "pos_vel_acc":
        return torch.cat([coords, vel, acc], dim=-1)
    if feature_set == "full":
        return torch.cat([coords, vel, acc, visibility], dim=-1)
    raise ValueError(f"Unknown feature_set: {feature_set!r}. Choose from {FEATURE_SETS}.")


# ---------------------------------------------------------------------------
# Spatial Graph Convolution
# ---------------------------------------------------------------------------

class SpatialGraphConv(nn.Module):
    """Graph convolution on the spatial (joint) dimension.

    For each time step, performs: X' = sigma(A_hat @ X @ W)
    where A_hat = A_fixed + A_learnable (residual adaptive adjacency).
    """

    def __init__(self, in_channels: int, out_channels: int, num_joints: int = NUM_JOINTS):
        super().__init__()
        self.num_joints = num_joints
        # Register the fixed normalized adjacency
        self.register_buffer("A_fixed", build_adjacency(num_joints))
        # Learnable residual adjacency — initialized near zero
        self.A_residual = nn.Parameter(torch.zeros(num_joints, num_joints) * 0.01)
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, J, C_in]
        Returns:
            [B, T, J, C_out]
        """
        B, T, J, C = x.shape
        # Effective adjacency = fixed + learnable
        A = self.A_fixed + self.A_residual  # [J, J]
        # Graph message passing: [J, J] @ [B*T, J, C] -> [B*T, J, C]
        x_flat = x.reshape(B * T, J, C)
        x_flat = torch.matmul(A, x_flat)      # spatial aggregation
        x_flat = self.W(x_flat)               # feature transform → [B*T, J, C_out]
        # BatchNorm over channel dim: reshape to [B*T*J, C_out]
        C_out = x_flat.shape[-1]
        x_flat = self.bn(x_flat.reshape(-1, C_out)).reshape(B, T, J, C_out)
        return x_flat


# ---------------------------------------------------------------------------
# Temporal Convolution
# ---------------------------------------------------------------------------

class TemporalConvBlock(nn.Module):
    """Dilated 1D convolution along the time axis for each joint independently.

    Input: [B, T, J, C] → permute to [B*J, C, T] → Conv1d → back to [B, T, J, C].
    """

    def __init__(self, channels: int, kernel_size: int = 9, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, J, C]
        Returns:
            [B, T, J, C]
        """
        B, T, J, C = x.shape
        # Merge batch and joint dims, conv over time
        x_perm = x.permute(0, 2, 3, 1).reshape(B * J, C, T)  # [B*J, C, T]
        y = self.conv(x_perm)  # [B*J, C, T]
        # Trim to T if padding caused overshoot
        y = y[:, :, :T]
        y = self.bn(y)
        y = F.silu(y)
        y = self.dropout(y)
        y = y.reshape(B, J, C, T).permute(0, 3, 1, 2)  # [B, T, J, C]
        return y


# ---------------------------------------------------------------------------
# Spiking GCN Block = Spatial + Temporal + Spiking Neuron
# ---------------------------------------------------------------------------

class SpikingGCNBlock(nn.Module):
    """One spatio-temporal-spiking block.

    Pipeline: SpatialGraphConv → SiLU → TemporalConv → PLIF neuron → Dropout + Residual.
    """

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.spatial = SpatialGraphConv(channels, channels)
        self.temporal = TemporalConvBlock(channels, kernel_size=9, dilation=dilation,
                                          dropout=dropout)
        # PLIF spiking neuron for temporal dynamics
        from spikingjelly.activation_based import neuron, surrogate
        self.plif = neuron.ParametricLIFNode(
            init_tau=2.0,
            surrogate_function=surrogate.ATan(),
        )
        self.dropout = nn.Dropout(dropout)

        # Skip projection (identity if same channels)
        self.skip_norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, J, C]
        Returns:
            [B, T, J, C]
        """
        B, T, J, C = x.shape
        residual = x

        # Spatial graph convolution
        out = self.spatial(x)      # [B, T, J, C]
        out = F.silu(out)

        # Temporal convolution
        out = self.temporal(out)   # [B, T, J, C]

        # Spiking neuron — process timestep by timestep
        spike_out = []
        for t in range(T):
            s = self.plif(out[:, t].reshape(B * J, C))  # [B*J, C]
            spike_out.append(s.reshape(B, J, C))
        out = torch.stack(spike_out, dim=1)  # [B, T, J, C]

        out = self.dropout(out)

        # Residual connection
        out = out + residual
        return out


class STGCNBlock(nn.Module):
    """One non-spiking spatio-temporal graph block.

    Pipeline: SpatialGraphConv -> SiLU -> TemporalConv -> Dropout + Residual.
    This is the base ST-GCN counterpart for ablation against SpikingGCNBlock.
    """

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.spatial = SpatialGraphConv(channels, channels)
        self.temporal = TemporalConvBlock(
            channels,
            kernel_size=9,
            dilation=dilation,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.spatial(x)
        out = F.silu(out)
        out = self.temporal(out)
        out = self.dropout(out)
        return out + residual


# ---------------------------------------------------------------------------
# ALIF Readout (reusable spiking readout)
# ---------------------------------------------------------------------------

class ALIFReadout(nn.Module):
    """Adaptive Leaky Integrate-and-Fire readout with surrogate gradient."""

    def __init__(self, tau_m: float = 2.0, tau_a: float = 1.5, beta: float = 0.1):
        super().__init__()
        from spikingjelly.activation_based import surrogate

        self.tau_m = tau_m
        self.tau_a = tau_a
        self.beta = beta
        self.threshold_base = 1.0
        self.surrogate = surrogate.ATan()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C]
        Returns:
            spikes: [B, T, C]
            membranes: [B, T, C]
        """
        bsz, steps, channels = x.shape
        v_mem = x.new_zeros(bsz, channels)
        a_adapt = x.new_zeros(bsz, channels)
        spikes: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
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


# ---------------------------------------------------------------------------
# Main Model: SpikingSTGCN
# ---------------------------------------------------------------------------

class SpikingSTGCN(nn.Module):
    """Spiking Spatio-Temporal Graph Convolutional Network for fall detection.

    Combines spatial graph convolution (skeleton topology), temporal
    convolution (motion dynamics), and spiking neural network neurons
    (energy-efficient temporal coding) for activity classification.

    Args:
        input_dim: Input feature dimension (default 132 = 33×4).
        num_classes: Number of output classes (11 multiclass or 2 binary).
        hidden_size: Hidden channel dimension for GCN/temporal conv.
        num_layers: Number of SpikingGCN blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        feature_set: str = "full",
        **kwargs,
    ) -> None:
        super().__init__()
        del input_dim  # We compute features from raw pose internally
        del kwargs

        self.feature_set = feature_set
        per_joint_feat_dim = feature_dim_for_set(feature_set)

        # Stem: project per-joint features to hidden_size
        self.stem = nn.Sequential(
            nn.Linear(per_joint_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Stack of SpikingGCN blocks with increasing dilation
        self.blocks = nn.ModuleList([
            SpikingGCNBlock(
                channels=hidden_size,
                dilation=2 ** (i % 4),   # dilation: 1, 2, 4, 8, 1, 2, ...
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Joint pooling → [B, T, hidden] → ALIF readout
        self.joint_pool_norm = nn.LayerNorm(hidden_size)
        self.readout = ALIFReadout(tau_m=2.0, tau_a=1.5, beta=0.1)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 132] normalized MediaPipe pose features.

        Returns:
            [B, num_classes] logits.
        """
        from spikingjelly.activation_based import functional

        # Reset all spiking neuron states
        for block in self.blocks:
            functional.reset_net(block)

        # Per-joint motion features: [B, T, 33, 10]
        joint_feats = per_joint_motion_features(x, self.feature_set)

        # Project to hidden dim: [B, T, 33, hidden]
        h = self.stem(joint_feats)

        # Stack of spatio-temporal-spiking blocks
        for block in self.blocks:
            h = block(h)

        # Pool over joints: mean across J → [B, T, hidden]
        h_pooled = h.mean(dim=2)
        h_pooled = self.joint_pool_norm(h_pooled)

        # ALIF readout over time
        _, membrane = self.readout(h_pooled)  # [B, T, hidden]

        # Temporal pooling: mean + last
        pooled = torch.cat([membrane.mean(dim=1), membrane[:, -1]], dim=-1)  # [B, hidden*2]

        return self.head(pooled)


class STGCN(nn.Module):
    """Base Spatio-Temporal Graph Convolutional Network without SNN neurons.

    Uses the same MediaPipe 33-joint graph, per-joint motion features, stem,
    spatial graph convolution, and temporal convolution as SpikingSTGCN, but
    replaces PLIF/ALIF dynamics with standard activations and temporal pooling.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        feature_set: str = "full",
        **kwargs,
    ) -> None:
        super().__init__()
        del input_dim, kwargs

        self.feature_set = feature_set
        per_joint_feat_dim = feature_dim_for_set(feature_set)
        self.stem = nn.Sequential(
            nn.Linear(per_joint_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(
                    channels=hidden_size,
                    dilation=2 ** (i % 4),
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.joint_pool_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joint_feats = per_joint_motion_features(x, self.feature_set)
        h = self.stem(joint_feats)
        for block in self.blocks:
            h = block(h)

        h_pooled = h.mean(dim=2)
        h_pooled = self.joint_pool_norm(h_pooled)
        pooled = torch.cat([h_pooled.mean(dim=1), h_pooled[:, -1]], dim=-1)
        return self.head(pooled)
