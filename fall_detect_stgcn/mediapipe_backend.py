"""Compatibility layer for MediaPipe Pose solutions and tasks APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

POSE_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]


def download_pose_model(path: Path, variant: str) -> None:
    if variant not in POSE_MODEL_URLS:
        raise ValueError(f"Unknown pose model variant: {variant}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    url = POSE_MODEL_URLS[variant]
    print(f"Downloading MediaPipe pose model: {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp_path.replace(path)


def has_solutions_pose() -> bool:
    try:
        import mediapipe as mp
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
    except Exception:
        return False


class SolutionsPoseEstimator:
    def __init__(
        self,
        static_image_mode: bool,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        **_: Any,
    ) -> None:
        import mediapipe as mp
        self.pose_module = mp.solutions.pose
        self.pose = self.pose_module.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self) -> "SolutionsPoseEstimator":
        return self

    def __exit__(self, *_: object) -> None:
        self.pose.close()

    def detect(self, rgb: np.ndarray, timestamp_ms: int = 0) -> tuple[np.ndarray, bool]:
        del timestamp_ms
        result = self.pose.process(rgb)
        keypoints = np.zeros((33, 4), dtype=np.float32)
        if not result.pose_landmarks:
            return keypoints, False
        for index, landmark in enumerate(result.pose_landmarks.landmark[:33]):
            keypoints[index] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        return keypoints, True


class TasksPoseEstimator:
    def __init__(
        self,
        pose_model: str | Path,
        pose_model_variant: str,
        download_model: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        **_: Any,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = Path(pose_model)
        if not model_path.exists():
            if not download_model:
                raise FileNotFoundError(
                    f"MediaPipe pose model not found: {model_path}. "
                    "Pass --download-pose-model or provide --pose-model."
                )
            download_pose_model(model_path, pose_model_variant)

        self.mp = mp
        self.vision = vision
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def __enter__(self) -> "TasksPoseEstimator":
        return self

    def __exit__(self, *_: object) -> None:
        self.landmarker.close()

    def detect(self, rgb: np.ndarray, timestamp_ms: int = 0) -> tuple[np.ndarray, bool]:
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        keypoints = np.zeros((33, 4), dtype=np.float32)
        if not result.pose_landmarks:
            return keypoints, False
        for index, landmark in enumerate(result.pose_landmarks[0][:33]):
            visibility = getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))
            keypoints[index] = [landmark.x, landmark.y, landmark.z, visibility]
        return keypoints, True


def pose_estimator(**kwargs: Any) -> SolutionsPoseEstimator | TasksPoseEstimator:
    if has_solutions_pose():
        return SolutionsPoseEstimator(**kwargs)
    return TasksPoseEstimator(**kwargs)


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, found: bool, threshold: float = 0.3) -> None:
    if not found:
        return
    height, width = frame.shape[:2]
    points: list[tuple[int, int] | None] = []
    for x, y, _, visibility in keypoints:
        if visibility < threshold:
            points.append(None)
        else:
            points.append((int(x * width), int(y * height)))

    for start, end in POSE_CONNECTIONS:
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], (60, 220, 60), 2)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 3, (30, 144, 255), -1)
