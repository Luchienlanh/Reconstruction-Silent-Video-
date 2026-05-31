from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from srcV2.utils.audio import log_mel_from_audio
from srcV2.utils.common import safe_name, seed_everything


DEFAULT_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


LIP_LANDMARK_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]


def resolve_face_landmarker_model(model_path: str | Path, auto_download: bool = True) -> Path:
    path = Path(model_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        candidates.extend(
            [
                Path.cwd() / path,
                repo_root / path,
                repo_root.parent / path,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if not auto_download:
        raise FileNotFoundError(f"FaceLandmarker task model not found: {model_path}")

    download_path = candidates[1] if len(candidates) > 1 else path
    download_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[landmarks] downloading FaceLandmarker model -> {download_path}")
    urllib.request.urlretrieve(DEFAULT_FACE_LANDMARKER_URL, download_path)
    return download_path.resolve()


def scan_raw_pairs(raw_dir: str | Path) -> list[tuple[Path, Path]]:
    root = Path(raw_dir)
    pairs = []
    for video_path in sorted(root.rglob("video.mp4")):
        audio_path = video_path.with_name("audio.wav")
        if audio_path.is_file():
            pairs.append((video_path, audio_path))
    return pairs


def center_square(gray: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    h, w = gray.shape[:2]
    side = float(min(h, w))
    x1 = (w - side) * 0.5
    y1 = (h - side) * 0.5
    crop = gray[int(round(y1)) : int(round(y1 + side)), int(round(x1)) : int(round(x1 + side))]
    return crop, (x1, y1, side)


def crop_with_box(gray: np.ndarray, box: tuple[float, float, float], frame_size: int) -> np.ndarray:
    x1, y1, side = box
    h, w = gray.shape[:2]
    x2 = x1 + side
    y2 = y1 + side

    src_x1 = max(0, int(math.floor(x1)))
    src_y1 = max(0, int(math.floor(y1)))
    src_x2 = min(w, int(math.ceil(x2)))
    src_y2 = min(h, int(math.ceil(y2)))
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        crop, _ = center_square(gray)
        return cv2.resize(crop, (frame_size, frame_size), interpolation=cv2.INTER_AREA)

    side_i = max(1, int(math.ceil(side)))
    canvas = np.zeros((side_i, side_i), dtype=gray.dtype)
    dst_x1 = max(0, src_x1 - int(math.floor(x1)))
    dst_y1 = max(0, src_y1 - int(math.floor(y1)))
    patch = gray[src_y1:src_y2, src_x1:src_x2]
    dst_x2 = min(side_i, dst_x1 + patch.shape[1])
    dst_y2 = min(side_i, dst_y1 + patch.shape[0])
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        crop, _ = center_square(gray)
        return cv2.resize(crop, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = patch[: dst_y2 - dst_y1, : dst_x2 - dst_x1]
    return cv2.resize(canvas, (frame_size, frame_size), interpolation=cv2.INTER_AREA)


def box_from_landmarks(lip_xy: np.ndarray, width: int, height: int, margin: float) -> tuple[float, float, float]:
    px = lip_xy[:, 0] * width
    py = lip_xy[:, 1] * height
    x_min, x_max = float(px.min()), float(px.max())
    y_min, y_max = float(py.min()), float(py.max())
    bw = max(2.0, x_max - x_min)
    bh = max(2.0, y_max - y_min)
    side = max(bw, bh) * margin
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    return cx - side * 0.5, cy - side * 0.5, side


def landmarks_to_crop_coords(lip_xy: np.ndarray, box: tuple[float, float, float], width: int, height: int) -> np.ndarray:
    x1, y1, side = box
    px = lip_xy[:, 0] * width
    py = lip_xy[:, 1] * height
    out = np.stack([(px - x1) / max(side, 1e-6), (py - y1) / max(side, 1e-6)], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def interpolate_missing(xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if bool(valid.all()):
        return xy
    if not bool(valid.any()):
        return xy
    out = xy.clone()
    valid_idx = torch.where(valid)[0]
    first = int(valid_idx[0].item())
    last = int(valid_idx[-1].item())
    out[:first] = out[first]
    out[last + 1 :] = out[last]
    valid_idx = torch.where(valid)[0]
    for left, right in zip(valid_idx[:-1].tolist(), valid_idx[1:].tolist()):
        if right - left <= 1:
            continue
        steps = right - left
        for i in range(1, steps):
            alpha = i / steps
            out[left + i] = (1.0 - alpha) * out[left] + alpha * out[right]
    return out


def add_derivatives(xy: torch.Tensor) -> torch.Tensor:
    d1 = torch.zeros_like(xy)
    d2 = torch.zeros_like(xy)
    if xy.shape[0] > 1:
        d1[1:] = xy[1:] - xy[:-1]
    if xy.shape[0] > 2:
        d2[1:] = d1[1:] - d1[:-1]
    return torch.cat([xy, d1, d2], dim=-1)


class LipLandmarkExtractor:
    def __init__(
        self,
        enabled: bool = True,
        model_path: str | Path = "face_landmarker_v2_with_blendshapes.task",
        auto_download: bool = True,
    ):
        self.enabled = enabled
        self.backend = "none"
        self.face_mesh = None
        self.mp = None
        if not enabled:
            return
        try:
            import mediapipe as mp

            self.mp = mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.backend = "solutions"
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                return

            self.backend = "tasks"
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = resolve_face_landmarker_model(model_path, auto_download=auto_download)
            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            print(f"[landmarks] mediapipe unavailable, using center crops only: {exc}")
            self.backend = "none"
            self.face_mesh = None

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if self.face_mesh is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            result = self.face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            lms = result.multi_face_landmarks[0].landmark
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            result = self.face_mesh.detect(mp_image)
            if not result.face_landmarks:
                return None
            lms = result.face_landmarks[0]
        return np.asarray([[lms[i].x, lms[i].y] for i in LIP_LANDMARK_IDXS], dtype=np.float32)


def decode_video_with_mouth_crops(
    video_path: Path,
    frame_size: int,
    margin: float,
    extractor: LipLandmarkExtractor,
    force_fps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if force_fps > 0:
        fps = float(force_fps)
    if fps <= 0:
        fps = 25.0

    crops: list[torch.Tensor] = []
    landmarks: list[torch.Tensor] = []
    valid: list[bool] = []
    boxes: list[torch.Tensor] = []
    last_box: tuple[float, float, float] | None = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        lip_xy = extractor.detect(frame)
        is_valid = lip_xy is not None and np.isfinite(lip_xy).all()
        if is_valid:
            box = box_from_landmarks(lip_xy, w, h, margin)
            last_box = box
            lm_crop = landmarks_to_crop_coords(lip_xy, box, w, h)
        else:
            if last_box is None:
                _crop, box = center_square(gray)
            else:
                box = last_box
            lm_crop = np.zeros((len(LIP_LANDMARK_IDXS), 2), dtype=np.float32)

        crop = crop_with_box(gray, box, frame_size)
        crops.append(torch.from_numpy(crop).float().div(255.0))
        landmarks.append(torch.from_numpy(lm_crop).float())
        valid.append(bool(is_valid))
        boxes.append(torch.tensor(box, dtype=torch.float32))

    cap.release()
    if not crops:
        raise RuntimeError(f"No frames decoded: {video_path}")

    video = torch.stack(crops, dim=0).unsqueeze(0)
    xy = torch.stack(landmarks, dim=0)
    valid_t = torch.tensor(valid, dtype=torch.bool)
    xy = interpolate_missing(xy, valid_t)
    lm6 = add_derivatives(xy)
    crop_boxes = torch.stack(boxes, dim=0)
    video_times = (torch.arange(video.shape[1], dtype=torch.float32) + 0.5) / float(fps)
    return video, lm6, valid_t, crop_boxes, video_times, fps


def process_pair(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    args,
    extractor: LipLandmarkExtractor,
) -> dict:
    video, landmarks, valid_mask, crop_boxes, video_times, fps = decode_video_with_mouth_crops(
        video_path,
        frame_size=args.frame_size,
        margin=args.margin,
        extractor=extractor,
        force_fps=args.force_fps,
    )
    mel = log_mel_from_audio(
        audio_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
    )
    mel_times = (torch.arange(mel.shape[0], dtype=torch.float32) + 0.5) * (args.hop_length / args.sample_rate)
    payload = {
        "format": "r2inr_v1",
        "video": video.contiguous(),
        "landmarks": landmarks.contiguous(),
        "mel": mel.contiguous(),
        "video_len": int(video.shape[1]),
        "mel_len": int(mel.shape[0]),
        "fps": float(fps),
        "sample_rate": int(args.sample_rate),
        "hop_length": int(args.hop_length),
        "video_times": video_times.contiguous(),
        "mel_times": mel_times.contiguous(),
        "mouth_valid_mask": valid_mask.contiguous(),
        "crop_boxes": crop_boxes.contiguous(),
        "source_video": str(video_path),
        "source_audio": str(audio_path),
    }
    torch.save(payload, output_path)
    return {
        "file": str(output_path),
        "video_len": int(video.shape[1]),
        "mel_len": int(mel.shape[0]),
        "valid_ratio": float(valid_mask.float().mean().item()),
    }


def run(args) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = scan_raw_pairs(args.raw_dir)
    if args.limit is not None:
        pairs = pairs[: max(1, min(int(args.limit), len(pairs)))]
    if not pairs:
        raise RuntimeError(f"No video.mp4/audio.wav pairs found under {args.raw_dir}")

    extractor = LipLandmarkExtractor(
        enabled=not args.no_mediapipe,
        model_path=args.face_landmarker_model,
        auto_download=not args.no_download_face_landmarker,
    )
    manifest = []
    failures = []
    try:
        for video_path, audio_path in tqdm(pairs, desc="build-cache"):
            out_name = safe_name(video_path) + ".pt"
            output_path = output_dir / out_name
            if output_path.exists() and not args.overwrite:
                manifest.append({"file": str(output_path), "skipped": True})
                continue
            try:
                manifest.append(process_pair(video_path, audio_path, output_path, args, extractor))
            except Exception as exc:
                failures.append({"video": str(video_path), "audio": str(audio_path), "error": repr(exc)})
                print(f"[fail] {video_path}: {exc}")
    finally:
        extractor.close()

    summary = {
        "raw_dir": str(args.raw_dir),
        "output_dir": str(output_dir),
        "total_pairs": len(pairs),
        "ok": len([x for x in manifest if not x.get("skipped")]),
        "skipped": len([x for x in manifest if x.get("skipped")]),
        "failed": len(failures),
        "success_rate": (len(pairs) - len(failures)) / max(1, len(pairs)),
        "config": vars(args),
        "items": manifest,
        "failures": failures,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] ok_or_skipped={len(pairs) - len(failures)}/{len(pairs)} success_rate={summary['success_rate']:.3f}")
    print(f"[out] {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build srcV2 R2INR cache from Dataset_Output raw folders.")
    parser.add_argument("--raw-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default="Processed_Data_R2INR")
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--margin", type=float, default=1.8)
    parser.add_argument("--force-fps", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mediapipe", action="store_true", help="Use center crop fallback for every frame.")
    parser.add_argument("--face-landmarker-model", default="face_landmarker_v2_with_blendshapes.task")
    parser.add_argument(
        "--no-download-face-landmarker",
        action="store_true",
        help="Do not auto-download the default MediaPipe FaceLandmarker task model when it is missing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
