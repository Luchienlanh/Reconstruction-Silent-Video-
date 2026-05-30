# -*- coding: utf-8 -*-

"""
Pose and Gait Tracking Module.
Trích xuất khớp xương cơ thể bằng MediaPipe Pose (Solutions API hoặc Tasks API).
Tính toán các đặc trưng dáng đi: độ nghiêng thân, chiều dài bước, vận tốc bước,
đối xứng dáng đi, và tỷ lệ hông-vai.
"""

from __future__ import annotations

import collections
import logging
import math
import time

import cv2
import numpy as np

logger = logging.getLogger("PoseGaitTracker")

# ---- Thử import MediaPipe ----
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "Thư viện 'mediapipe' chưa được cài đặt. "
        "PoseGaitTracker sẽ chạy ở chế độ giả lập."
    )

# Ánh xạ chỉ số landmark MediaPipe Pose (33 điểm) sang tên khớp
LANDMARK_MAP: dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# Các khớp chính dùng cho phân tích dáng đi (bỏ qua mặt để giảm nhiễu)
GAIT_JOINTS: list[str] = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
    "nose",
]


class PoseGaitTracker:
    """Trích xuất skeleton và tính toán gait features theo thời gian thực."""

    def __init__(
        self,
        model_path: str = "",
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.pose = None
        self._sim_t0 = time.time()

        # Lịch sử vị trí mắt cá chân để tính vận tốc bước
        self._ankle_history: collections.deque[dict[str, tuple[int, int]]] = (
            collections.deque(maxlen=15)
        )

        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()

    # ------------------------------------------------------------------
    # Khởi tạo MediaPipe Pose (Solutions API)
    # ------------------------------------------------------------------
    def _init_mediapipe(self) -> None:
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            logger.info("MediaPipe Pose initialized (complexity=%d).", self.model_complexity)
        except Exception as e:
            logger.error("Lỗi khởi tạo MediaPipe Pose: %s", e)
            self.pose = None

    # ------------------------------------------------------------------
    # API chính: track_pose
    # ------------------------------------------------------------------
    def track_pose(
        self, frame_rgb: np.ndarray
    ) -> tuple[dict[str, tuple[int, int]], bool]:
        """
        Trả về:
            keypoints – dict {tên_khớp: (x_pixel, y_pixel)}
            pose_found – True nếu phát hiện được người
        """
        h, w = frame_rgb.shape[:2]

        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return self._fallback_simulation(h, w)

        try:
            results = self.pose.process(frame_rgb)
            if not results.pose_landmarks:
                return {}, False

            keypoints: dict[str, tuple[int, int]] = {}
            for idx, name in LANDMARK_MAP.items():
                lm = results.pose_landmarks.landmark[idx]
                vis = getattr(lm, "visibility", 1.0)
                if vis < 0.35:
                    continue
                keypoints[name] = (int(lm.x * w), int(lm.y * h))

            found = len(keypoints) >= 6  # ít nhất 6 khớp mới coi là hợp lệ

            # Cập nhật lịch sử ankle
            if found:
                ankles: dict[str, tuple[int, int]] = {}
                if "left_ankle" in keypoints:
                    ankles["left_ankle"] = keypoints["left_ankle"]
                if "right_ankle" in keypoints:
                    ankles["right_ankle"] = keypoints["right_ankle"]
                if ankles:
                    self._ankle_history.append(ankles)

            return keypoints, found

        except Exception as e:
            logger.error("track_pose error: %s", e)
            return {}, False

    # ------------------------------------------------------------------
    # API chính: calculate_gait_features
    # ------------------------------------------------------------------
    def calculate_gait_features(
        self, keypoints: dict[str, tuple[int, int]]
    ) -> dict[str, float]:
        """
        Trả về dict chứa các chỉ số dáng đi:
            body_tilt      – Góc nghiêng thân (độ). 0 = thẳng đứng.
            step_length    – Khoảng cách ngang 2 mắt cá chân (pixel).
            step_velocity  – Tốc độ di chuyển trung bình của mắt cá chân (px/frame).
            gait_symmetry  – Tỷ lệ đối xứng trái-phải (0..1). 1 = hoàn toàn đối xứng.
            hip_shoulder_ratio – Tỷ lệ chiều rộng hông / vai.
        """
        features: dict[str, float] = {
            "body_tilt": 0.0,
            "step_length": 0.0,
            "step_velocity": 0.0,
            "gait_symmetry": 1.0,
            "hip_shoulder_ratio": 0.0,
        }

        # ---- Body Tilt ----
        ls = keypoints.get("left_shoulder")
        rs = keypoints.get("right_shoulder")
        lh = keypoints.get("left_hip")
        rh = keypoints.get("right_hip")

        if ls and rs and lh and rh:
            mid_sx = (ls[0] + rs[0]) / 2.0
            mid_sy = (ls[1] + rs[1]) / 2.0
            mid_hx = (lh[0] + rh[0]) / 2.0
            mid_hy = (lh[1] + rh[1]) / 2.0

            dx = mid_sx - mid_hx
            dy = mid_sy - mid_hy  # y hướng xuống

            features["body_tilt"] = math.degrees(math.atan2(dx, -dy))

            # ---- Hip/Shoulder Ratio ----
            shoulder_w = math.dist(ls, rs)
            hip_w = math.dist(lh, rh)
            if shoulder_w > 1e-3:
                features["hip_shoulder_ratio"] = hip_w / shoulder_w

        # ---- Step Length ----
        la = keypoints.get("left_ankle")
        ra = keypoints.get("right_ankle")
        if la and ra:
            features["step_length"] = float(abs(la[0] - ra[0]))

        # ---- Gait Symmetry ----
        lk = keypoints.get("left_knee")
        rk = keypoints.get("right_knee")
        if la and ra and lk and rk and lh and rh:
            left_leg = math.dist(lh, lk) + math.dist(lk, la)
            right_leg = math.dist(rh, rk) + math.dist(rk, ra)
            total = left_leg + right_leg
            if total > 1e-3:
                features["gait_symmetry"] = 1.0 - abs(left_leg - right_leg) / total

        # ---- Step Velocity ----
        if len(self._ankle_history) >= 3:
            recent = list(self._ankle_history)[-5:]
            deltas: list[float] = []
            for i in range(1, len(recent)):
                for side in ("left_ankle", "right_ankle"):
                    if side in recent[i] and side in recent[i - 1]:
                        d = math.dist(recent[i][side], recent[i - 1][side])
                        deltas.append(d)
            if deltas:
                features["step_velocity"] = float(np.mean(deltas))

        return features

    # ------------------------------------------------------------------
    # Giải phóng tài nguyên
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def __enter__(self) -> PoseGaitTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Fallback mô phỏng
    # ------------------------------------------------------------------
    def _fallback_simulation(
        self, h: int, w: int
    ) -> tuple[dict[str, tuple[int, int]], bool]:
        """Trả về skeleton giả lập dao động theo thời gian để test pipeline."""
        t = time.time() - self._sim_t0
        phase = math.sin(t * 2.0)
        sway = int(phase * 15)

        cx = w // 2 + sway
        cy_hip = int(h * 0.55)

        kp: dict[str, tuple[int, int]] = {
            "nose": (cx, cy_hip - 130),
            "left_shoulder": (cx - 35, cy_hip - 90),
            "right_shoulder": (cx + 35, cy_hip - 90),
            "left_elbow": (cx - 50, cy_hip - 45),
            "right_elbow": (cx + 50, cy_hip - 45),
            "left_wrist": (cx - 55, cy_hip - 5),
            "right_wrist": (cx + 55, cy_hip - 5),
            "left_hip": (cx - 22, cy_hip),
            "right_hip": (cx + 22, cy_hip),
            "left_knee": (cx - 25 + int(phase * 8), cy_hip + 50),
            "right_knee": (cx + 25 - int(phase * 8), cy_hip + 50),
            "left_ankle": (cx - 28 + int(phase * 18), cy_hip + 100),
            "right_ankle": (cx + 28 - int(phase * 18), cy_hip + 100),
            "left_heel": (cx - 30 + int(phase * 18), cy_hip + 105),
            "right_heel": (cx + 30 - int(phase * 18), cy_hip + 105),
            "left_foot_index": (cx - 22 + int(phase * 20), cy_hip + 108),
            "right_foot_index": (cx + 22 - int(phase * 20), cy_hip + 108),
        }

        ankles: dict[str, tuple[int, int]] = {
            "left_ankle": kp["left_ankle"],
            "right_ankle": kp["right_ankle"],
        }
        self._ankle_history.append(ankles)

        return kp, True
