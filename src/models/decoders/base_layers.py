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
from .siren import FiLMSIREN
from .wire import FiLMWIRE
from .finer import TFiLMFINERDecoder
from .wrap import FiLMWrapFINSIREN, FiLMWrapFINWIRE
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class FiLMLayer(nn.Module):
    def __init__(self, z_dim=512, condition_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, 2 * z_dim)
        )
        nn.init.kaiming_uniform_(self.net[-1].weight)
        self.net[-1].weight.data *= 0.01
        self.net[-1].bias.data.uniform_(-1.0 / z_dim, 1.0 / z_dim)
    
    def forward(self, X, Condition):
        gamma, beta = self.net(Condition).chunk(2, dim=-1)

        shape = [1] * (X.ndim - 2) 
        gamma = gamma.view(gamma.size(0), -1, *shape)
        beta = beta.view(beta.size(0), -1, *shape)
        return X * gamma + beta

class TFiLM(nn.Module):
    def __init__(self, z_dim=512, condition_dim=512):
        super().__init__()
        self.film = FiLMLayer(z_dim=z_dim, condition_dim=condition_dim)

    def forward(self, X, Condition):
        B, T = X.shape[:2]
        X_flat = X.reshape(B * T, *X.shape[2:])
        Condition_flat = Condition.reshape(B * T, *Condition.shape[2:])
        out_flat = self.film(X_flat, Condition_flat)
        out = out_flat.reshape(B, T, *X.shape[2:])
        return out

class DualFiLMLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_siren=30.0, omega_wire=30.0, scale=5.0, is_first=False):
        super().__init__()
        self.siren = FiLMSIREN(in_features, out_features, omega_siren, is_first)
        self.wire = FiLMWIRE(in_features, out_features, omega_wire, scale, is_first)
        self.fusion = nn.Sequential(
            nn.Linear(out_features * 2, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, X, gamma_s, beta_s, gamma_w, beta_w):
        out_siren = self.siren(X, gamma_s, beta_s)
        out_wire = self.wire(X, gamma_w, beta_w)
        out = torch.cat([out_siren, out_wire], dim=-1)
        out = self.fusion(out)
        return out

class DualWrapLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_fisin=30.0, omega_fiwi=30.0, scale=5.0, is_first=False):
        super().__init__()
        self.fisin = FiLMWrapFINSIREN(in_features, out_features, omega_fisin, is_first)
        self.fiwi = FiLMWrapFINWIRE(in_features, out_features, omega_fiwi, scale, is_first)
        self.fusion = nn.Sequential(
            nn.Linear(out_features * 2, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
    
    def forward(self, X, gamma_fs, beta_fs, gamma_fw, beta_fw):
        out_fs = self.fisin(X, gamma_fs, beta_fs)
        out_fw = self.fiwi(X, gamma_fw, beta_fw)
        out = torch.cat([out_fs, out_fw], dim=-1)
        out = self.fusion(out)
        return out

