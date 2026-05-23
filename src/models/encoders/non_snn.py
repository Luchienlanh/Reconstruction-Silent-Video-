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
from tqdm.auto import tqdm

class NonSpikingDirectEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45),
            nn.SiLU(inplace=True),
            nn.Conv3d(45, out_channels, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, X):
        return self.net(X)

class NonSpikingConv2DPlus1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, stride: int=1, padding: int=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
        )

    def forward(self, X):
        return self.net(X)

    @staticmethod
    def get_stride(stride: int):
        return stride, stride, stride

class NonSpikingBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, conv_builder: Callable[..., nn.Module],
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        mid_channels = (in_channels * out_channels * 3 * 3 * 3) // (
            in_channels * out_channels * 1 * 3 * 3 + in_channels * out_channels * 3 * 1 * 1
        )
        self.conv1 = nn.Sequential(
            conv_builder(in_channels, out_channels, mid_channels, stride),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(out_channels, out_channels, mid_channels),
            nn.BatchNorm3d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)
        self.downsample = downsample

    def forward(self, X: Tensor) -> Tensor:
        residual = X
        out = self.conv1(X)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(X)
        out = out + residual
        return self.activation(out)

class NonSpikingVidResNet(nn.Module):
    def __init__(self, block=NonSpikingBasicBlock, conv_makers=[NonSpikingConv2DPlus1D] * 4,
                 layers=[2, 2, 2, 2], zero_init_residual: bool=False, spatial_pool_size: int=1,
                 in_channels: int=1):
        super().__init__()
        self.in_channels = 64
        self.spatial_pool_size = spatial_pool_size
        self.stem = NonSpikingDirectEncoder(in_channels=in_channels, out_channels=64)
        self.layer1 = self._make_layer(block, 64, conv_makers[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, conv_makers[1], layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, conv_makers[2], layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, conv_makers[3], layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((None, spatial_pool_size, spatial_pool_size))
        self.fc = nn.Linear(512 * block.expansion * spatial_pool_size * spatial_pool_size, 512)

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
            for m in self.modules():
                if isinstance(m, NonSpikingBasicBlock):
                    nn.init.constant_(m.conv2[-1].weight, 0)

    def _make_layer(self, block, out_channels: int, conv_builder: Callable[..., nn.Module],
                    num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, conv_builder, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, conv_builder))
        return nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        X = self.stem(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.avgpool(X)
        X = X.permute(0, 2, 1, 3, 4).flatten(start_dim=2)  # (B, T, C*H_pool*W_pool)
        return self.fc(X)

class NonSpikingTemporalEncoder(nn.Module):
    def __init__(self, z_dim=512, n_heads=8, n_blocks=4, mlp_hidden_dim=2048, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=z_dim,
            nhead=n_heads,
            dim_feedforward=mlp_hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)

    def forward(self, X):
        return self.encoder(X)

class NonSpikingViTEncoder(nn.Module):
    def __init__(self, z_dim=512, n_heads=8, n_blocks=4, mlp_hidden_dim=2048, T_max=1000, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, T_max, z_dim) * 0.02)
        self.backbone = NonSpikingVidResNet()
        self.temporal = NonSpikingTemporalEncoder(
            z_dim=z_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, X):
        features = self.backbone(X)  # (B, T, 512)
        if features.size(1) > self.pos_embedding.size(1):
            raise ValueError(f"T={features.size(1)} v??t T_max={self.pos_embedding.size(1)}")
        features = features + self.pos_embedding[:, :features.size(1), :]
        features = self.temporal(features)
        return self.norm(features)
