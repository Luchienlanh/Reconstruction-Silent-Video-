import os
import gc
import glob
import math
import random
import warnings
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
from spikingjelly.activation_based import neuron, surrogate, functional

class SpikingDirectEncoder(nn.Sequential):
    def __init__(self, in_channels=1, out_channels=64) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv3d(in_channels, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(45)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

        self.temporal_conv = nn.Conv3d(45, out_channels, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, X):
        # X: (B, C, T, H, W)
        s1 = self.spatial_conv(X)
        b1 = self.bn1(s1)
        l1 = self.lif1(b1)

        s2 = self.temporal_conv(l1)
        b2 = self.bn2(s2)
        l2 = self.lif2(b2)
        return l2

class SpikingConv2DPlus1D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, stride: int=1, padding: int=1):
        super().__init__(
          nn.Conv3d(
              in_channels=in_channels,
              out_channels=mid_channels,
              kernel_size=(1, 3, 3),
              stride=(1, stride, stride),
              padding=(0, padding, padding),
              bias=False,
          ),
          nn.BatchNorm3d(mid_channels),
          neuron.LIFNode(surrogate_function=surrogate.ATan()),
          nn.Conv3d(
              in_channels=mid_channels,
              out_channels=out_channels,
              kernel_size=(3, 1, 1),
              stride=(1, 1, 1),
              padding=(1, 0, 0),
              bias=False,
          ),
        )
    @staticmethod
    def get_stride(stride: int):
        return stride, stride, stride

class SpikingBasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        mid_channels = (in_channels * out_channels * 3 * 3 * 3) // (in_channels * out_channels * 1 * 3 * 3 + in_channels * out_channels * 3 * 1 * 1)
        super().__init__()
        self.conv1 = nn.Sequential(
            conv_builder(in_channels, out_channels, mid_channels, stride),
            nn.BatchNorm3d(out_channels),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            conv_builder(out_channels, out_channels, mid_channels),
            nn.BatchNorm3d(out_channels),
        )
        self.lif = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.downsample = downsample
        self.stride = stride

    def forward(self, X: Tensor) -> Tensor:
        residual = X
        out = self.conv1(X)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(X)
        out += residual
        out = self.lif(out)
        return out

class VidResNet(nn.Module):
    def __init__(
        self,
        block = SpikingBasicBlock,
        conv_makers = [SpikingConv2DPlus1D] * 4,
        layers = [2, 2, 2, 2],
        zero_init_residual: bool = False,
    ):
        super().__init__()
        self.in_channels = 64
        self.stem = SpikingDirectEncoder(in_channels=1, out_channels=64)

        self.layer1 = self._make_layer(block, 64, conv_makers[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, conv_makers[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, conv_makers[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, conv_makers[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            # SpikingBasicBlock kh?ng c? bn3; conv2[-1] l? BatchNorm3d cu?i c?a residual branch.
            for m in self.modules():
                if isinstance(m, SpikingBasicBlock):
                    nn.init.constant_(m.conv2[-1].weight, 0)

    # ★ Helper: checkpoint an toàn cho SNN
    def _snn_checkpoint(self, layer, X):
        """Gradient checkpointing có reset neuron states."""
        functional.reset_net(layer)
        return layer(X)

    def forward(self, X: Tensor) -> Tensor:
        X = self.stem(X)
        # ★ Checkpoint từng block, không phải từng layer
        for block in self.layer1:
            X = self._snn_checkpoint(block, X)
        for block in self.layer2:
            X = self._snn_checkpoint(block, X)
        for block in self.layer3:
            X = self._snn_checkpoint(block, X)
        for block in self.layer4:
            X = self._snn_checkpoint(block, X)
        X = self.avgpool(X)
        X = X.squeeze(-1).squeeze(-1)
        X = X.permute(0, 2, 1)
        X = self.fc(X)
        return X

    def _make_layer(
        self,
        block: SpikingBasicBlock,
        out_channels: int,
        conv_builder: Callable[..., nn.Module],
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, conv_builder, stride, downsample))

        self.in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, conv_builder))

        return nn.Sequential(*layers)

class SpikingAttention(nn.Module):
    def __init__(self, z_dim=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.dim_head = z_dim // n_heads
        self.scale = self.dim_head ** -0.5

        self.q_linear = nn.Linear(z_dim, z_dim)
        self.k_linear = nn.Linear(z_dim, z_dim)
        self.v_linear = nn.Linear(z_dim, z_dim)

        self.q_plif = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
        self.k_plif = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
        self.v_plif = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())

        self.attn_plif = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
        self.proj = nn.Linear(z_dim, z_dim)

    def forward(self, X):
        B, T, C = X.shape

        # Q, K, V
        q = self.q_linear(X)                # (B, T, C)
        k = self.k_linear(X)
        v = self.v_linear(X)

        # Đưa qua PLIF (flatten theo thời gian)
        q = q.reshape(B * T, C)
        q = self.q_plif(q)
        q = q.reshape(B, T, C)

        k = k.reshape(B * T, C)
        k = self.k_plif(k)
        k = k.reshape(B, T, C)

        v = v.reshape(B * T, C)
        v = self.v_plif(v)
        v = v.reshape(B, T, C)

        # Multi-head reshape
        q = q.reshape(B, T, self.n_heads, self.dim_head).permute(0, 2, 1, 3)  # (B, n_heads, T, dim_head)
        k = k.reshape(B, T, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.dim_head).permute(0, 2, 1, 3)

        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale   # (B, n_heads, T, T)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_out = attn_scores @ v                             # (B, n_heads, T, dim_head)

        # Merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)

        # attn_plif
        attn_out = attn_out.reshape(B * T, C)
        attn_out = self.attn_plif(attn_out)
        attn_out = attn_out.reshape(B, T, C)

        # Projection
        out = self.proj(attn_out)          # (B, T, C)
        return out

class SpikingMLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.plif = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())
        
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, X):
        # X: (B, T, C)
        B, T, C = X.shape

        # fc1
        X = self.fc1(X)                     # (B, T, hidden_dim)
        # Chuẩn bị cho BN1 (yêu cầu (N, C, L) với L=1)
        X = X.reshape(B * T, -1)            # (B*T, hidden_dim)
        X = X.unsqueeze(-1)                 # (B*T, hidden_dim, 1)
        X = self.bn1(X)                     # BN trên kênh hidden_dim
        X = X.squeeze(-1)                   # (B*T, hidden_dim)

        # Neuron PLIF (xử lý từng vector độc lập)
        X = self.plif(X)                    # (B*T, hidden_dim)
        X = X.reshape(B, T, -1)             # (B, T, hidden_dim)

        # fc2
        X = self.fc2(X)                     # (B, T, out_dim)
        # BN2
        X = X.reshape(B * T, -1)            # (B*T, out_dim)
        X = X.unsqueeze(-1)                 # (B*T, out_dim, 1)
        X = self.bn2(X)
        X = X.squeeze(-1)                   # (B*T, out_dim)
        X = X.reshape(B, T, -1)             # (B, T, out_dim)

        return X

class SpikingVisionTransformer(nn.Module):
    def __init__(self, z_dim=512, n_heads=8, mlp_hidden_dim=2048):
        super().__init__()
        self.attn = SpikingAttention(z_dim=z_dim, n_heads=n_heads)
        self.attn_plif = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())

        self.mlp = SpikingMLP(in_dim=z_dim, hidden_dim=mlp_hidden_dim, out_dim=z_dim)
        self.mlp_plif = neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, X):
        # X: (B, T, C)
        # Attention
        attn_out = self.attn(X)                     # (B, T, C)
        residual = attn_out + X                      # residual connection

        # attn_plif
        residual_flat = residual.reshape(-1, residual.size(-1))  # (B*T, C)
        residual_spike = self.attn_plif(residual_flat)
        residual_spike = residual_spike.reshape(residual.shape)  # (B, T, C)

        # MLP
        mlp_out = self.mlp(residual_spike)           # (B, T, C)
        residual2 = mlp_out + residual_spike          # residual connection

        # mlp_plif
        residual2_flat = residual2.reshape(-1, residual2.size(-1))  # (B*T, C)
        out_spikes = self.mlp_plif(residual2_flat)
        out_spikes = out_spikes.reshape(residual2.shape)            # (B, T, C)

        return out_spikes

class ConfALIFNode(nn.Module):
    def __init__(self, tau_m=2.0, tau_a=1.5, beta=0.1):
        super().__init__()
        self.tau_m = tau_m
        self.tau_a = tau_a
        self.beta = beta
        self.surrogate = surrogate.ATan()
        self.threshold_base = 1.0

    def forward(self, X):
        B, T, C = X.shape
        v_mem = torch.zeros(B, C, device=X.device)
        a_adapt = torch.zeros(B, C, device=X.device)
        spikes_seq = []
        v_mem_seq = []
        for t in range(T):
            current_I = X[:, t, :]
            a_adapt = a_adapt * (1 - 1/self.tau_a)
            v_th = self.threshold_base + a_adapt
            v_mem = v_mem * (1 - 1/self.tau_m) + current_I
            v_mem_seq.append(v_mem)
            spike = self.surrogate(v_mem - v_th)
            v_mem = v_mem * (1 - spike)
            a_adapt = a_adapt + self.beta * spike
            spikes_seq.append(spike)
        return torch.stack(spikes_seq, dim=1), torch.stack(v_mem_seq, dim=1)

class ReadoutLayer(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, tau_a=1.5, tau_m=2.0, beta=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.alif = ConfALIFNode(tau_m=tau_m, tau_a=tau_a, beta=beta)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, X): # X: B T C 
        proj_X = self.proj(X)
        _, v_mem_seq = self.alif(proj_X)
        z_continuous = self.norm(v_mem_seq)
        return z_continuous      

class SpikingViTEncoder(nn.Module):
    def __init__(self, z_dim=512, n_heads=8, n_blocks=8, mlp_hidden_dim=2048, T_max=1000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, T_max, z_dim))
        self.backbone = VidResNet()
        self.transformers = nn.ModuleList([
            SpikingVisionTransformer(z_dim=z_dim, n_heads=n_heads, mlp_hidden_dim=mlp_hidden_dim) 
            for _ in range(n_blocks)
        ])
        self.readout = ReadoutLayer(in_dim=z_dim, out_dim=z_dim)
    
    def forward(self, X): # X: B C T H W
        features = self.backbone(X) # B T C
        features += self.pos_embedding[:, :features.size(1), :]
        for block in self.transformers:
            features = block(features) # B T C
        z_continuous = self.readout(features) # B T C
        return z_continuous

