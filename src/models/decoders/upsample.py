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

class MelTemporalUpsampleDecoder(nn.Module):
    """Upsample video-frame latents to HiFi-GAN mel-frame rate before decoding."""
    def __init__(self, base_decoder, sample_rate=16000, fps=25, hop_length=256):
        super().__init__()
        self.base_decoder = base_decoder
        self.sample_rate = sample_rate
        self.fps = fps
        self.hop_length = hop_length
        self.ratio = (sample_rate / fps) / hop_length

    def infer_target_len(self, video_len):
        return max(1, int(round(float(video_len) * self.ratio)))

    def forward(self, condition, target_len=None):
        if target_len is None:
            target_len = self.infer_target_len(condition.shape[1])
        if condition.shape[1] != target_len:
            condition = F.interpolate(
                condition.transpose(1, 2),
                size=int(target_len),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
        return self.base_decoder(condition)

