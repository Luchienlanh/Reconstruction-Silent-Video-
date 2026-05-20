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

LANDMARK_KEYS = ("lip_landmarks", "mouth_landmarks", "landmarks", "face_landmarks")

def find_landmarks_in_data(data: dict, require: bool=False, path: str=None):
    for key in LANDMARK_KEYS:
        if key in data:
            lm = data[key]
            if not torch.is_tensor(lm):
                lm = torch.as_tensor(lm)
            lm = lm.float()
            if lm.dim() == 4 and lm.shape[0] == 1:
                lm = lm.squeeze(0)
            if lm.dim() != 3:
                raise ValueError(f"Landmarks must be (T,N,D), got {tuple(lm.shape)} from key={key}")
            if lm.shape[-1] < 2:
                raise ValueError(f"Landmarks need at least x,y coordinates, got {tuple(lm.shape)}")
            return lm[..., :2].contiguous(), key
    if require:
        hint = f" in {path}" if path else ""
        raise KeyError(
            f"USE_LANDMARKS=True but no landmark key{hint}. "
            f"Expected one of {LANDMARK_KEYS}. Regenerate .pt files with lip/frame landmarks first."
        )
    return None, None

def compute_landmark_derivatives(landmarks: torch.Tensor) -> torch.Tensor:
    """
    Compute position, velocity (1st derivative), and acceleration (2nd derivative).

    Args:
        landmarks: (T, N, 2) raw x/y landmark coordinates
    Returns:
        (T, N, 6) tensor with channels [x, y, dx, dy, d2x, d2y]
    """
    assert landmarks.dim() == 3 and landmarks.shape[-1] == 2, \
        f"Expected (T, N, 2), got {tuple(landmarks.shape)}"

    T = landmarks.shape[0]

    # First-order derivative (velocity): d1[t] = landmarks[t] - landmarks[t-1]
    d1 = torch.zeros_like(landmarks)
    if T > 1:
        d1[1:] = landmarks[1:] - landmarks[:-1]

    # Second-order derivative (acceleration): d2[t] = d1[t] - d1[t-1]
    d2 = torch.zeros_like(landmarks)
    if T > 2:
        d2[1:] = d1[1:] - d1[:-1]

    return torch.cat([landmarks, d1, d2], dim=-1)  # (T, N, 6)

def interpolate_missing_landmarks(landmarks: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """
    Detect frames where all landmark coords are near-zero (lost/missing)
    and linearly interpolate from nearest valid neighbors.

    Args:
        landmarks: (T, N, 2)
        threshold: if frame norm < threshold, treat as missing
    Returns:
        landmarks with missing frames interpolated (T, N, 2)
    """
    landmarks = landmarks.clone()
    T = landmarks.shape[0]
    if T < 2:
        return landmarks

    # Identify missing frames: per-frame L2 norm across all points
    frame_norms = landmarks.flatten(1).norm(dim=1)  # (T,)
    missing = frame_norms < threshold  # boolean mask

    if not missing.any():
        return landmarks

    valid_indices = torch.where(~missing)[0]
    if len(valid_indices) == 0:
        # All frames are missing - nothing we can do
        return landmarks

    for t in range(T):
        if not missing[t]:
            continue
        # Find nearest valid frame before and after
        before = valid_indices[valid_indices < t]
        after = valid_indices[valid_indices > t]

        if len(before) > 0 and len(after) > 0:
            t0 = int(before[-1].item())
            t1 = int(after[0].item())
            alpha = (t - t0) / max(t1 - t0, 1)
            landmarks[t] = (1 - alpha) * landmarks[t0] + alpha * landmarks[t1]
        elif len(before) > 0:
            landmarks[t] = landmarks[int(before[-1].item())]
        elif len(after) > 0:
            landmarks[t] = landmarks[int(after[0].item())]

    return landmarks

