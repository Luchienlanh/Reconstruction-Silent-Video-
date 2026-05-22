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

class FiLMFINER(nn.Module):
    def __init__(self, in_features, out_features, omega_zero=30.0, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_zero = omega_zero
        self.is_first = is_first
        with torch.no_grad():
            if self.is_first:
                bound = 1 / in_features
            else:
                bound = torch.sqrt(torch.tensor(6.0 / in_features)) / self.omega_zero
            self.linear.weight.uniform_(-bound, bound)
        
    def forward(self, X, gamma, beta):
        out_dtype = X.dtype
        X = self.linear(X)
        if not self.is_first:
            X = self.omega_zero * X
        modulated = gamma.float() * X.float() + beta.float()
        phase = self.omega_zero * (torch.abs(modulated) + 1.0) * modulated
        phase = torch.nan_to_num(phase, nan=0.0, posinf=1000.0, neginf=-1000.0).clamp(-1000.0, 1000.0)
        return torch.sin(phase).to(dtype=out_dtype)

class TFiLMFINERDecoder(nn.Module):
    def __init__(self, condition_dim=512, hidden_dim=256, out_dim=640, num_layers=4, omega_zero=30.0, use_conv=False, output_activation="tanh"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_activation = output_activation

        total_params = num_layers * 2 * hidden_dim
        if use_conv:
            self.param_net = nn.Conv1d(condition_dim, total_params, kernel_size=3, padding=1)
            nn.init.kaiming_uniform_(self.param_net.weight)
            self.param_net.weight.data *= 0.01
            with torch.no_grad():
                self.param_net.bias.data.uniform_(-1.0 / hidden_dim, 1.0 / hidden_dim)
        else:
            self.param_gru = nn.GRU(
                condition_dim, condition_dim,
                batch_first=True, bidirectional=True
            )
            self.param_proj = nn.Linear(condition_dim * 2, total_params)
            # Init: Kaiming * 0.01 theo SIREN paper
            nn.init.kaiming_uniform_(self.param_proj.weight)
            self.param_proj.weight.data *= 0.01
            self.param_proj.bias.data.uniform_(-1.0 / hidden_dim, 1.0 / hidden_dim)
            #     nn.Linear(condition_dim, condition_dim),
            #     nn.ReLU(),
            #     nn.Linear(condition_dim, total_params)
            # )

        self.finer_layers = nn.ModuleList()
        self.finer_layers.append(FiLMFINER(in_features=hidden_dim, out_features=hidden_dim, omega_zero=omega_zero, is_first=True))
        for _ in range(1, num_layers):
            self.finer_layers.append(FiLMFINER(in_features=hidden_dim, out_features=hidden_dim, omega_zero=omega_zero, is_first=False))
        self.final_layer = nn.Linear(hidden_dim , out_dim)
        self.output_scale = nn.Parameter(torch.ones(1))  # learnable scale
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

        self.input_constant = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)

        # Time positional encoding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, Condition):
        B, T, _ = Condition.shape
        if hasattr(self, 'param_net') and isinstance(self.param_net, nn.Conv1d):
            params = self.param_net(Condition.permute(0, 2, 1))
            params = params.permute(0, 2, 1)
        else:
            gru_out, _ = self.param_gru(Condition)  # (B, T, condition_dim*2)
            params = self.param_proj(gru_out.reshape(B * T, -1))  # (B*T, total_params)
            params = params.reshape(B, T, -1)
        
        gammas = []
        betas = []

        for i in range(self.num_layers):
            start = i * 2 * self.hidden_dim
            beta = params[:, :, start: start + self.hidden_dim]
            gamma = params[:, :, start + self.hidden_dim: start + 2 * self.hidden_dim]
            gammas.append(gamma)
            betas.append(beta)
        
        # Time positional encoding: mỗi frame nhận input khác nhau
        X_base = self.input_constant.expand(B * T, -1)
        # Tạo sinusoidal time positions
        t_pos = torch.linspace(0, 1, T, device=Condition.device)  # [0, 1]
        t_pos = t_pos.unsqueeze(0).expand(B, -1).reshape(B * T, 1)  # (B*T, 1)
        # Sinusoidal encoding: [sin(2^0 * pi * t), cos(2^0 * pi * t), ...]
        # Bounded sinusoidal frequencies. The previous 2**arange(hidden_dim//2)
        # overflowed for hidden_dim=256 and produced sin(inf)=nan.
        half_dim = self.hidden_dim // 2
        pe_dtype = torch.float32
        freqs = torch.linspace(1.0, 32.0, half_dim, device=Condition.device, dtype=pe_dtype)
        angles = t_pos.to(dtype=pe_dtype) * freqs.unsqueeze(0) * torch.pi
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B*T, hidden_dim)
        if pe.shape[-1] < self.hidden_dim:
            pe = torch.nn.functional.pad(pe, (0, self.hidden_dim - pe.shape[-1]))
        pe = pe.to(dtype=Condition.dtype)
        time_signal = self.time_embed(pe)  # (B*T, hidden_dim)
        X = X_base + time_signal  # Mỗi frame có input khác nhau
        gammas_flat = [g.reshape(B * T, -1) for g in gammas]
        betas_flat = [b.reshape(B * T, -1) for b in betas]

        for i, layer in enumerate(self.finer_layers):
            X = layer(X, gammas_flat[i], betas_flat[i])
        out = self.final_layer(X)
        if self.output_activation == "tanh":
            out = torch.tanh(out) * self.output_scale  # [-scale, +scale], waveform range
        elif self.output_activation not in (None, "none", "linear"):
            raise ValueError(f"Unsupported output_activation={self.output_activation}")
        out = out.reshape(B, T, -1)
        return out

