"""Pose feature normalization shared by training and live inference."""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 33
LANDMARK_DIM = 4
INPUT_DIM = NUM_LANDMARKS * LANDMARK_DIM

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def normalize_pose_sequence(keypoints: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Normalize MediaPipe keypoints to root-relative, torso-scaled features.

    Args:
        keypoints: Array shaped [T, 33, 4] with x, y, z, visibility.
        mask: Optional [T] array where 1 means a pose was detected.

    Returns:
        Float32 array shaped [T, 132].
    """
    if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_LANDMARKS, LANDMARK_DIM):
        raise ValueError(f"Expected [T, 33, 4] keypoints, got {keypoints.shape}")

    data = keypoints.astype(np.float32, copy=True)
    coords = data[:, :, :3]
    visibility = data[:, :, 3:4]

    hip_center = (coords[:, LEFT_HIP : LEFT_HIP + 1] + coords[:, RIGHT_HIP : RIGHT_HIP + 1]) * 0.5
    shoulder_center = (
        coords[:, LEFT_SHOULDER : LEFT_SHOULDER + 1]
        + coords[:, RIGHT_SHOULDER : RIGHT_SHOULDER + 1]
    ) * 0.5

    torso_scale = np.linalg.norm((shoulder_center - hip_center)[:, 0, :2], axis=1)
    shoulder_width = np.linalg.norm(
        coords[:, LEFT_SHOULDER, :2] - coords[:, RIGHT_SHOULDER, :2], axis=1
    )
    hip_width = np.linalg.norm(coords[:, LEFT_HIP, :2] - coords[:, RIGHT_HIP, :2], axis=1)
    scale = np.maximum.reduce([torso_scale, shoulder_width, hip_width, np.full_like(torso_scale, 1e-3)])

    coords = (coords - hip_center) / scale[:, None, None]
    features = np.concatenate([coords, visibility], axis=2).reshape(len(data), INPUT_DIM)

    if mask is not None:
        present = mask.astype(bool)
        features[~present] = 0.0

    return features.astype(np.float32)


def resample_sequence(features: np.ndarray, seq_len: int) -> np.ndarray:
    """Uniformly resample or pad a temporal sequence to a fixed length."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if len(features) == 0:
        return np.zeros((seq_len, features.shape[-1] if features.ndim == 2 else INPUT_DIM), dtype=np.float32)
    if len(features) == seq_len:
        return features.astype(np.float32, copy=False)

    indices = np.linspace(0, len(features) - 1, seq_len)
    indices = np.round(indices).astype(np.int64)
    return features[indices].astype(np.float32, copy=False)
