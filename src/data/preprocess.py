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
import cv2
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    pass

def affine_trans(landmarks, frame, target_size=(224, 224)):
  h, w, _ = frame.shape
  left = np.array([landmarks[33][0] * w, landmarks[33][1] * h])
  right = np.array([landmarks[263][0] * w, landmarks[263][1] * h])

  dY = right[1] - left[1]
  dX = right[0] - left[0]

  angle = np.degrees(np.arctan2(dY, dX))

  theta = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)

  distance = np.sqrt((left[0] - right[0])**2 + (left[1] - right[1])**2)
  ratio = target_size[0] * 0.3
  scale = ratio / distance

  Matrix = cv2.getRotationMatrix2D(theta, angle, scale)
  Matrix[0, 2] += (target_size[0] * 0.5) - theta[0]
  Matrix[1, 2] += (target_size[1] * 0.35) - theta[1]

  transform = cv2.warpAffine(frame, Matrix, target_size, flags=cv2.INTER_CUBIC)

  return transform

def lips(transform, landmark_path, size=112):
      data = np.load(landmark_path, allow_pickle=True)
      landmarks = data['landmarks']
      midX = (landmarks[61][0] + landmarks[291][0]) / 2
      midY = (landmarks[0][1] + landmarks[17][1]) / 2

      half = size // 2
      
      y1, y2 = max(0, int(midY - half)), int(midY + half)
      x1, x2 = max(0, int(midX - half)), int(midX + half)
      
      lip_roi = transform[y1:y2, x1:x2]

      # Nếu cắt bị thiếu do chạm biên, resize lại cho đủ 112x112
      if lip_roi.shape[0] != size or lip_roi.shape[1] != size:
            lip_roi = cv2.resize(lip_roi, (size, size))
      
      # if len(lip_roi.shape) == 3:
      #       lip_roi = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2GRAY)

      return lip_roi

def safe_name_from_folder(folder_path):
    parts = folder_path.split(os.sep)
    return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]

def build_source_folder_map(dataset_path):
    video_paths = glob.glob(os.path.join(dataset_path, '**', 'video.mp4'), recursive=True)
    folder_map = {}
    for video_path in video_paths:
        folder = os.path.dirname(video_path)
        folder_map[safe_name_from_folder(folder)] = folder
    return folder_map

def convert_video_fps(input_path, output_path, target_fps=25):
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', input_path,
        '-r', str(target_fps),
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-preset', 'fast',
        output_path,
    ], check=True, capture_output=True)

def safe_remove(path, retries=8, delay=0.25):
    if not path or not os.path.exists(path):
        return True
    for attempt in range(retries):
        try:
            os.remove(path)
            return True
        except PermissionError:
            gc.collect()
            time.sleep(delay * (attempt + 1))
    print(f"  -> Warning: could not remove temp file: {path}")
    return False

def extract_lip_landmarks(lm_list):
    return np.asarray(
        [[lm_list[idx][0], lm_list[idx][1]] for idx in LIP_LANDMARK_IDXS],
        dtype=np.float32,
    )

def resample_landmarks_to_len(landmarks_tensor, target_len):
    # landmarks_tensor: (T, N, 2). Linear interpolate over time if ffmpeg/decord gives a slightly different frame count.
    if landmarks_tensor.shape[0] == target_len:
        return landmarks_tensor
    x = landmarks_tensor.permute(1, 2, 0).reshape(1, -1, landmarks_tensor.shape[0])
    x = torch.nn.functional.interpolate(x, size=int(target_len), mode='linear', align_corners=False)
    return x.reshape(landmarks_tensor.shape[1], landmarks_tensor.shape[2], target_len).permute(2, 0, 1).contiguous()

def extract_landmarks_from_video(video_path, target_video_len=None):
    base_options = python.BaseOptions(model_asset_path=MIGRATE_LM_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    detector = None
    vr = None
    landmark_frames = []
    last_lip_landmarks = None
    try:
        detector = vision.FaceLandmarker.create_from_options(options)
        vr = VideoReader(video_path, ctx=cpu(0))
        for i in range(len(vr)):
            frame = vr[i].asnumpy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(i * (1000 / MIGRATE_LM_FPS))
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                lip_lm = extract_lip_landmarks(lm_list)
                landmark_frames.append(lip_lm)
                last_lip_landmarks = lip_lm
            else:
                if last_lip_landmarks is not None:
                    landmark_frames.append(last_lip_landmarks.copy())
                else:
                    landmark_frames.append(np.zeros((len(LIP_LANDMARK_IDXS), 2), dtype=np.float32))
    finally:
        if detector is not None:
            detector.close()
        if vr is not None:
            del vr
        gc.collect()

    if not landmark_frames:
        raise RuntimeError(f'No frames processed from {video_path}')

    landmarks_tensor = torch.from_numpy(np.stack(landmark_frames)).float()  # (T_video, 40, 2)
    if target_video_len is not None:
        landmarks_tensor = resample_landmarks_to_len(landmarks_tensor, int(target_video_len))
    return landmarks_tensor

def load_fallback_face_frame(
    video_path: str,
    frame_idx: int,
    target_size: int = 112,
) -> Optional[torch.Tensor]:
    """
    Read a single frame from the source video and return a grayscale 112x112 tensor.
    Returns None if reading fails.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target_size, target_size))
        return torch.from_numpy(gray).float() / 255.0
    except Exception:
        return None

