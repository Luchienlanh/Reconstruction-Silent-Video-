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
from .preprocess import build_source_folder_map, load_fallback_face_frame
from .utils import find_landmarks_in_data, compute_landmark_derivatives, interpolate_missing_landmarks

class VNLipDataset(Dataset):
    def __init__(self, data_dir, max_frames=None, random_crop=True, return_path=False,
                 target_type="mel_hifigan", use_landmarks=True):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.random_crop = random_crop
        self.return_path = return_path
        self.target_type = target_type
        self.use_landmarks = use_landmarks
        self.landmark_num_points = None
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Khong tim thay data_dir: {self.data_dir}")
        self.files = sorted(f for f in os.listdir(self.data_dir) if f.endswith('.pt'))
        if len(self.files) == 0:
            raise RuntimeError(f"Khong co file .pt trong {self.data_dir}")
        if self.use_landmarks:
            sample_path = os.path.join(self.data_dir, self.files[0])
            sample = torch.load(sample_path, map_location='cpu', weights_only=False)
            lm, _ = find_landmarks_in_data(sample, require=True, path=sample_path)
            self.landmark_num_points = int(lm.shape[1])

    def __len__(self):
        return len(self.files)

    def _get_target(self, data, file_path):
        if self.target_type == "mel_hifigan":
            if "mel" not in data:
                raise KeyError(f"{file_path} has no 'mel' target")
            target = data['mel'].float()
            n_mels = int(data.get('n_mels', 80))
            if target.dim() != 2:
                raise ValueError(f"Mel target must be 2D, got {tuple(target.shape)} in {file_path}")
            if target.shape[0] == n_mels and target.shape[1] != n_mels:
                target = target.transpose(0, 1).contiguous()
            target_len = int(data.get('mel_len', target.shape[0]))
            return target[:target_len], target_len

        if self.target_type == "waveform":
            if "audio" not in data:
                raise KeyError(f"{file_path} has no 'audio' target")
            target = data['audio'].float()
            target_len = int(data.get('audio_len', target.shape[0]))
            return target[:target_len], target_len

        raise ValueError(f"Unsupported target_type={self.target_type}")

    def _crop(self, video, target, target_len, landmarks=None, data=None):
        video_len = int(data.get('video_len', video.shape[1])) if data else video.shape[1]
        video_len = min(video_len, video.shape[1])
        target_len = min(int(target_len), target.shape[0])
        video = video[:, :video_len]
        target = target[:target_len]
        if landmarks is not None:
            landmarks = landmarks[:video_len]

        if self.max_frames is None or video_len <= self.max_frames:
            return video, target, landmarks

        if self.random_crop:
            start = random.randint(0, video_len - self.max_frames)
        else:
            start = (video_len - self.max_frames) // 2
        end = start + self.max_frames

        ratio = target_len / max(video_len, 1)
        target_start = int(round(start * ratio))
        target_end = int(round(end * ratio))
        target_end = max(target_start + 1, min(target_end, target_len))

        video = video[:, start:end]
        target = target[target_start:target_end]
        if landmarks is not None:
            landmarks = landmarks[start:end]
        return video, target, landmarks

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        video = data['video'].float()
        target, target_len = self._get_target(data, file_path)

        if video.dim() == 3:
            video = video.unsqueeze(0)
        if video.dim() != 4 or target.dim() != 2:
            raise ValueError(f"Sai shape trong {file_path}: video={tuple(video.shape)}, target={tuple(target.shape)}")

        landmarks = None
        if self.use_landmarks:
            landmarks, _ = find_landmarks_in_data(data, require=True, path=file_path)

        video, target, landmarks = self._crop(video, target, target_len, landmarks=landmarks, data=data)

        if self.use_landmarks:
            if self.return_path:
                return video, landmarks, target, file_path
            return video, landmarks, target
        if self.return_path:
            return video, target, file_path
        return video, target

def collate_pad(batch):
    has_path = isinstance(batch[0][-1], str)
    has_landmarks = (len(batch[0]) >= 3 and torch.is_tensor(batch[0][1]) and batch[0][1].dim() == 3)

    if has_landmarks and has_path:
        videos, landmarks, targets, paths = zip(*batch)
    elif has_landmarks:
        videos, landmarks, targets = zip(*batch)
        paths = None
    elif has_path:
        videos, targets, paths = zip(*batch)
        landmarks = None
    else:
        videos, targets = zip(*batch)
        landmarks = None
        paths = None

    video_lengths = torch.tensor([v.shape[1] for v in videos], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    T_video_max = int(video_lengths.max().item())
    T_target_max = int(target_lengths.max().item())

    padded_videos = []
    padded_targets = []
    padded_landmarks = []

    for i, (v, t) in enumerate(zip(videos, targets)):
        v_len = v.shape[1]
        t_len = t.shape[0]
        if v_len < T_video_max:
            pad_v = torch.zeros(v.shape[0], T_video_max - v_len, v.shape[2], v.shape[3], dtype=v.dtype)
            v = torch.cat([v, pad_v], dim=1)
        if t_len < T_target_max:
            pad_t = torch.zeros(T_target_max - t_len, t.shape[1], dtype=t.dtype)
            t = torch.cat([t, pad_t], dim=0)
        padded_videos.append(v)
        padded_targets.append(t)

        if has_landmarks:
            lm = landmarks[i]
            if lm.shape[0] < T_video_max:
                pad_lm = torch.zeros(T_video_max - lm.shape[0], lm.shape[1], lm.shape[2], dtype=lm.dtype)
                lm = torch.cat([lm, pad_lm], dim=0)
            padded_landmarks.append(lm)

    video_batches = torch.stack(padded_videos, dim=0)      # (B, 1, T_video, 112, 112)
    target_batches = torch.stack(padded_targets, dim=0)    # mel: (B, T_mel, 80), waveform: (B, T_video, 640)

    if has_landmarks:
        landmark_batches = torch.stack(padded_landmarks, dim=0)  # (B, T_video, N, 2)
        if has_path:
            return video_batches, landmark_batches, target_batches, target_lengths, list(paths)
        return video_batches, landmark_batches, target_batches, target_lengths

    if has_path:
        return video_batches, target_batches, target_lengths, list(paths)
    return video_batches, target_batches, target_lengths

def pick_pt_file(data_dir, pt_path=None):
    if pt_path is not None:
        return pt_path
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
    if not files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")
    return os.path.join(data_dir, files[0])

def infer_landmark_num_points(data_dir, pt_path=None):
    pt_path = pick_pt_file(data_dir, pt_path)
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    landmarks, key = find_landmarks_in_data(data, require=True, path=pt_path)
    print(f"Using landmarks from key='{key}' with {landmarks.shape[1]} points: {os.path.basename(pt_path)}")
    return int(landmarks.shape[1])

def build_overfit_2pt_dataset():
    data_dir = globals().get("DATA_DIR", globals().get("MEL_DATA_DIR", "Processed_Data_Mel_HiFiGAN"))

    if OVERFIT_2PT_FILES is None:
        base_dataset = VNLipDataset(
            data_dir,
            max_frames=OVERFIT_2PT_MAX_FRAMES,
            random_crop=True,
            target_type="mel_hifigan",
            use_landmarks=True,
            return_path=True,
        )
        if len(base_dataset) < 2:
            raise RuntimeError(f"Need at least 2 .pt files in {data_dir}, got {len(base_dataset)}")
        return Subset(base_dataset, [0, 1])

    class TwoFileDataset(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = files

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            path = self.files[idx]
            data = torch.load(path, map_location="cpu", weights_only=False)

            video = data["video"].float()
            if video.dim() == 3:
                video = video.unsqueeze(0)

            mel = data["mel"].float()
            n_mels = int(data.get("n_mels", 80))
            if mel.dim() == 2 and mel.shape[0] == n_mels and mel.shape[1] != n_mels:
                mel = mel.transpose(0, 1).contiguous()

            landmarks, _ = find_landmarks_in_data(data, require=True, path=path)

            video_len = int(data.get("video_len", video.shape[1]))
            mel_len = int(data.get("mel_len", mel.shape[0]))

            video = video[:, :video_len]
            landmarks = landmarks[:video_len]
            mel = mel[:mel_len]

            if OVERFIT_2PT_MAX_FRAMES is not None and video.shape[1] > OVERFIT_2PT_MAX_FRAMES:
                start = 0
                end = start + OVERFIT_2PT_MAX_FRAMES

                ratio = mel.shape[0] / max(video.shape[1], 1)
                mel_start = int(round(start * ratio))
                mel_end = int(round(end * ratio))
                mel_end = max(mel_start + 1, min(mel_end, mel.shape[0]))

                video = video[:, start:end]
                landmarks = landmarks[start:end]
                mel = mel[mel_start:mel_end]

            return video, landmarks, mel, path

    return TwoFileDataset(OVERFIT_2PT_FILES)

class VNLipDatasetV2(Dataset):
    """
    Enhanced dataset that:
    - Computes 1st/2nd-order landmark derivatives (velocity, acceleration)
    - Detects lost/corrupted lip crop frames
    - Falls back to full-face frames from source video when lip crop is lost
    - Returns landmarks as (T, N, 6) instead of (T, N, 2)
    """

    def __init__(
        self,
        data_dir: str,
        max_frames: Optional[int] = None,
        random_crop: bool = True,
        return_path: bool = False,
        target_type: str = "mel_hifigan",
        use_landmarks: bool = True,
        dataset_output_dir: str = "Dataset_Output",
        enable_fallback: bool = True,
        force_full_frame: bool = False,
        lost_frame_threshold: float = 0.01,
        use_offline_features: bool = False,
        use_optical_flow: bool = False,
        flow_cache_dir: Optional[str] = None,
        flow_method: str = "farneback",
        flow_scale: float = 10.0,
    ):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.random_crop = random_crop
        self.return_path = return_path
        self.target_type = target_type
        self.use_landmarks = use_landmarks
        self.enable_fallback = enable_fallback
        self.force_full_frame = force_full_frame
        self.lost_frame_threshold = lost_frame_threshold
        self.use_offline_features = use_offline_features
        self.use_optical_flow = use_optical_flow
        self.flow_method = flow_method
        self.flow_scale = max(float(flow_scale), 1e-6)
        self.flow_cache_dir = flow_cache_dir
        self.landmark_num_points = None
        self._flow_warned = False

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Khong tim thay data_dir: {self.data_dir}")
        if self.use_optical_flow and self.flow_cache_dir:
            os.makedirs(self.flow_cache_dir, exist_ok=True)

        self.files = sorted(f for f in os.listdir(self.data_dir) if f.endswith(".pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"Khong co file .pt trong {self.data_dir}")

        # Pre-scan first file for landmark shape
        if self.use_landmarks:
            sample_path = os.path.join(self.data_dir, self.files[0])
            sample = torch.load(sample_path, map_location="cpu", weights_only=False)
            lm, _ = find_landmarks_in_data(sample, require=True, path=sample_path)
            self.landmark_num_points = int(lm.shape[1])

        # Build source folder map for fallback
        self.source_folder_map = {}
        if (self.enable_fallback or self.force_full_frame) and os.path.isdir(dataset_output_dir):
            self.source_folder_map = build_source_folder_map(dataset_output_dir)

    def __len__(self) -> int:
        return len(self.files)

    def _get_target(self, data: dict, file_path: str):
        if self.target_type == "mel_hifigan":
            if "mel" not in data:
                raise KeyError(f"{file_path} has no 'mel' target")
            target = data["mel"].float()
            n_mels = int(data.get("n_mels", 80))
            if target.dim() != 2:
                raise ValueError(f"Mel target must be 2D, got {tuple(target.shape)} in {file_path}")
            if target.shape[0] == n_mels and target.shape[1] != n_mels:
                target = target.transpose(0, 1).contiguous()
            target_len = int(data.get("mel_len", target.shape[0]))
            return target[:target_len], target_len

        if self.target_type == "waveform":
            if "audio" not in data:
                raise KeyError(f"{file_path} has no 'audio' target")
            target = data["audio"].float()
            target_len = int(data.get("audio_len", target.shape[0]))
            return target[:target_len], target_len

        raise ValueError(f"Unsupported target_type={self.target_type}")

    def _detect_lost_frames(self, video: torch.Tensor) -> torch.Tensor:
        """
        Detect lost/corrupted lip crop frames by checking if frame is near-black.
        video: (C, T, H, W)
        Returns: boolean mask (T,) where True = lost
        """
        # Per-frame mean pixel intensity
        frame_means = video.mean(dim=(0, 2, 3))  # (T,)
        return frame_means < self.lost_frame_threshold

    def _apply_fallback(
        self,
        video: torch.Tensor,
        lost_mask: torch.Tensor,
        file_name: str,
    ) -> torch.Tensor:
        """Replace lost frames with full-face frames from source video."""
        if not lost_mask.any() or not self.source_folder_map:
            return video

        safe_name = os.path.splitext(file_name)[0]
        folder = self.source_folder_map.get(safe_name)
        if folder is None:
            return video

        video_candidates = glob.glob(os.path.join(folder, "video.*"))
        if not video_candidates:
            return video

        source_video_path = video_candidates[0]

        for t in torch.where(lost_mask)[0].tolist():
            face_frame = load_fallback_face_frame(source_video_path, t)
            if face_frame is not None:
                video[0, t] = face_frame  # Replace in channel 0 (grayscale)

        return video

    def _apply_full_frame_input(self, video: torch.Tensor, file_name: str) -> torch.Tensor:
        """Replace every lip-crop frame with the corresponding full source frame resized to 112x112."""
        if not self.source_folder_map:
            return video

        safe_name = os.path.splitext(file_name)[0]
        folder = self.source_folder_map.get(safe_name)
        if folder is None:
            return video

        video_candidates = glob.glob(os.path.join(folder, "video.*"))
        if not video_candidates:
            return video

        source_video_path = video_candidates[0]
        full_frames = video.clone()
        for t in range(video.shape[1]):
            face_frame = load_fallback_face_frame(source_video_path, t)
            if face_frame is not None:
                full_frames[0, t] = face_frame
        return full_frames

    def _flow_cache_path(self, file_name: str) -> Optional[str]:
        if not self.flow_cache_dir:
            return None
        stem = os.path.splitext(os.path.basename(file_name))[0]
        return os.path.join(self.flow_cache_dir, f"{stem}.flow.pt")

    def _video_to_uint8(self, video: torch.Tensor) -> np.ndarray:
        # video: (C, T, H, W). Flow uses the first/mean grayscale channel.
        x = video.detach().float()
        if x.shape[0] > 1:
            x = x.mean(dim=0)
        else:
            x = x[0]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = float(x.amin())
        x_max = float(x.amax())
        if x_max <= 1.5 and x_min >= -0.5:
            x = x.clamp(0.0, 1.0) * 255.0
        else:
            x = x.clamp(0.0, 255.0)
        return x.byte().cpu().numpy()

    def _pseudo_flow(self, frames: np.ndarray) -> torch.Tensor:
        frames_f = frames.astype(np.float32) / 255.0
        t, h, w = frames_f.shape
        flow = np.zeros((2, t, h, w), dtype=np.float32)
        for i in range(1, t):
            dt = frames_f[i] - frames_f[i - 1]
            gy, gx = np.gradient(frames_f[i])
            flow[0, i] = dt * gx
            flow[1, i] = dt * gy
        return torch.from_numpy(flow)

    def _compute_optical_flow(self, video: torch.Tensor) -> torch.Tensor:
        frames = self._video_to_uint8(video)
        t, h, w = frames.shape
        if t <= 1:
            return torch.zeros(2, t, h, w, dtype=torch.float32)

        if self.flow_method == "pseudo":
            return self._pseudo_flow(frames)

        try:
            import cv2
        except Exception:
            if not self._flow_warned:
                warnings.warn("cv2 is not available; using pseudo optical flow fallback.")
                self._flow_warned = True
            return self._pseudo_flow(frames)

        flow = np.zeros((2, t, h, w), dtype=np.float32)
        for i in range(1, t):
            f = cv2.calcOpticalFlowFarneback(
                frames[i - 1],
                frames[i],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            flow[0, i] = f[..., 0] / self.flow_scale
            flow[1, i] = f[..., 1] / self.flow_scale
        flow = np.clip(flow, -5.0, 5.0)
        return torch.from_numpy(flow)

    def _load_or_compute_flow(self, video: torch.Tensor, file_name: str) -> torch.Tensor:
        cache_path = self._flow_cache_path(file_name)
        if cache_path and os.path.isfile(cache_path):
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(cached, dict):
                cached = cached.get("flow", cached.get("optical_flow"))
            if torch.is_tensor(cached) and cached.dim() == 4 and cached.shape[0] == 2:
                if (
                    cached.shape[1] == video.shape[1]
                    and cached.shape[2] == video.shape[2]
                    and cached.shape[3] == video.shape[3]
                ):
                    return cached.float()

        flow = self._compute_optical_flow(video).float()
        if cache_path:
            torch.save({"flow": flow}, cache_path)
        return flow

    def _crop(self, video, target, target_len, landmarks=None, data=None, flow=None):
        is_features = (video.dim() == 2)  # (T, D) shape
        raw_video_len = video.shape[0] if is_features else video.shape[1]
        
        video_len = int(data.get("video_len", raw_video_len)) if data else raw_video_len
        video_len = min(video_len, raw_video_len)
        target_len = min(int(target_len), target.shape[0])
        
        if is_features:
            video = video[:video_len]
        else:
            video = video[:, :video_len]
        if flow is not None:
            flow = flow[:, :video_len]
            
        target = target[:target_len]
        if landmarks is not None:
            landmarks = landmarks[:video_len]

        if self.max_frames is None or video_len <= self.max_frames:
            return video, target, landmarks, flow

        if self.random_crop:
            start = random.randint(0, video_len - self.max_frames)
        else:
            start = (video_len - self.max_frames) // 2
        end = start + self.max_frames

        ratio = target_len / max(video_len, 1)
        target_start = int(round(start * ratio))
        target_end = int(round(end * ratio))
        target_end = max(target_start + 1, min(target_end, target_len))

        if is_features:
            video = video[start:end]
        else:
            video = video[:, start:end]
        if flow is not None:
            flow = flow[:, start:end]
            
        target = target[target_start:target_end]
        if landmarks is not None:
            landmarks = landmarks[start:end]
        return video, target, landmarks, flow

    def __getitem__(self, idx: int):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        target, target_len = self._get_target(data, file_path)

        # --- Check if using pre-extracted SOTA visual features ---
        use_offline = self.use_offline_features and "visual_feature" in data
        
        if use_offline:
            video = data["visual_feature"].float()  # (T, 512)
            if video.dim() != 2:
                raise ValueError(
                    f"Sai shape visual_feature trong {file_path}: expected 2D (T, 512), got {tuple(video.shape)}"
                )
        else:
            video = data["video"].float()
            if video.dim() == 3:
                video = video.unsqueeze(0)
            if video.dim() != 4 or target.dim() != 2:
                raise ValueError(
                    f"Sai shape trong {file_path}: video={tuple(video.shape)}, target={tuple(target.shape)}"
                )

            # --- Optional full-frame visual input ---
            if self.force_full_frame:
                video = self._apply_full_frame_input(video, file_name)

            # --- Fallback for lost frames ---
            if self.enable_fallback and not self.force_full_frame:
                lost_mask = self._detect_lost_frames(video)
                if lost_mask.any():
                    video = self._apply_fallback(video, lost_mask, file_name)

        flow = None
        if self.use_optical_flow and not use_offline:
            flow = self._load_or_compute_flow(video, file_name)

        # --- Landmarks ---
        landmarks = None
        if self.use_landmarks:
            lm_raw, _ = find_landmarks_in_data(data, require=True, path=file_path)
            # lm_raw: (T, N, 2)
            # Interpolate missing landmark frames
            lm_raw = interpolate_missing_landmarks(lm_raw)
            # Compute derivatives: (T, N, 2) -> (T, N, 6)
            landmarks = compute_landmark_derivatives(lm_raw)

        video, target, landmarks, flow = self._crop(video, target, target_len, landmarks=landmarks, data=data, flow=flow)

        if self.use_landmarks:
            if self.return_path:
                if flow is not None:
                    return video, flow, landmarks, target, file_path
                return video, landmarks, target, file_path
            if flow is not None:
                return video, flow, landmarks, target
            return video, landmarks, target
        if self.return_path:
            return video, target, file_path
        return video, target


def collate_pad_v2(batch):
    """
    Collate function that pads variable-length sequences.
    Handles landmarks with 6 channels (x, y, dx, dy, d2x, d2y).
    Supports visual feature sequences (T, 512) and raw videos (C, T, H, W).
    """
    has_path = isinstance(batch[0][-1], str)
    sample_payload = batch[0][:-1] if has_path else batch[0]
    has_flow = (
        len(sample_payload) >= 4
        and torch.is_tensor(sample_payload[1])
        and sample_payload[1].dim() == 4
        and torch.is_tensor(sample_payload[2])
        and sample_payload[2].dim() == 3
    )
    has_landmarks = (
        (has_flow and len(sample_payload) >= 4)
        or (
            len(sample_payload) >= 3
            and torch.is_tensor(sample_payload[1])
            and sample_payload[1].dim() == 3
        )
    )

    flows = None
    if has_flow and has_path:
        videos, flows, landmarks, targets, paths = zip(*batch)
    elif has_flow:
        videos, flows, landmarks, targets = zip(*batch)
        paths = None
    elif has_landmarks and has_path:
        videos, landmarks, targets, paths = zip(*batch)
    elif has_landmarks:
        videos, landmarks, targets = zip(*batch)
        paths = None
    elif has_path:
        videos, targets, paths = zip(*batch)
        landmarks = None
    else:
        videos, targets = zip(*batch)
        landmarks = None
        paths = None

    # Detect if we are using 2D visual features (T, 512) or 4D raw videos (C, T, H, W)
    video_lengths = torch.tensor([v.shape[0] if v.dim() == 2 else v.shape[1] for v in videos], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    T_video_max = int(video_lengths.max().item())
    T_target_max = int(target_lengths.max().item())

    padded_videos = []
    padded_flows = []
    padded_targets = []
    padded_landmarks = []

    for i, (v, t) in enumerate(zip(videos, targets)):
        is_features = (v.dim() == 2)
        v_len = v.shape[0] if is_features else v.shape[1]
        t_len = t.shape[0]
        
        if v_len < T_video_max:
            if is_features:
                # pad features: (T, 512) -> (T_max, 512)
                pad_v = torch.zeros(T_video_max - v_len, v.shape[1], dtype=v.dtype)
                v = torch.cat([v, pad_v], dim=0)
            else:
                # pad raw video: (C, T, H, W) -> (C, T_max, H, W)
                pad_v = torch.zeros(v.shape[0], T_video_max - v_len, v.shape[2], v.shape[3], dtype=v.dtype)
                v = torch.cat([v, pad_v], dim=1)
                
        if t_len < T_target_max:
            pad_t = torch.zeros(T_target_max - t_len, t.shape[1], dtype=t.dtype)
            t = torch.cat([t, pad_t], dim=0)
            
        padded_videos.append(v)
        padded_targets.append(t)

        if has_flow:
            fl = flows[i]
            if fl.shape[1] < T_video_max:
                pad_fl = torch.zeros(fl.shape[0], T_video_max - fl.shape[1], fl.shape[2], fl.shape[3], dtype=fl.dtype)
                fl = torch.cat([fl, pad_fl], dim=1)
            padded_flows.append(fl)

        if has_landmarks:
            lm = landmarks[i]
            if lm.shape[0] < T_video_max:
                # lm: (T, N, D) where D=6
                pad_lm = torch.zeros(
                    T_video_max - lm.shape[0], lm.shape[1], lm.shape[2],
                    dtype=lm.dtype,
                )
                lm = torch.cat([lm, pad_lm], dim=0)
            padded_landmarks.append(lm)

    video_batches = torch.stack(padded_videos, dim=0)
    flow_batches = torch.stack(padded_flows, dim=0) if has_flow else None
    target_batches = torch.stack(padded_targets, dim=0)

    if has_landmarks:
        landmark_batches = torch.stack(padded_landmarks, dim=0)  # (B, T, N, 6)
        if has_flow:
            if has_path:
                return video_batches, flow_batches, landmark_batches, target_batches, target_lengths, list(paths)
            return video_batches, flow_batches, landmark_batches, target_batches, target_lengths
        if has_path:
            return video_batches, landmark_batches, target_batches, target_lengths, list(paths)
        return video_batches, landmark_batches, target_batches, target_lengths

    if has_path:
        return video_batches, target_batches, target_lengths, list(paths)
    return video_batches, target_batches, target_lengths
