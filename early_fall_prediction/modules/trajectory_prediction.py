# -*- coding: utf-8 -*-

"""
Foot Trajectory Prediction Module.
Dự đoán vị trí tiếp đất tiếp theo của bàn chân dựa trên lịch sử bước đi.
Hỗ trợ hai phương pháp:
  1. Ngoại suy tuyến tính (Linear Extrapolation) – nhanh, phù hợp realtime.
  2. Hồi quy đa thức bậc 2 (Quadratic Regression) – bắt được chuyển hướng nhẹ.
"""

from __future__ import annotations

import collections
import math

import numpy as np


class TrajectoryPredictor:
    """Dự đoán quỹ đạo bước chân trong tương lai gần."""

    def __init__(
        self,
        history_len: int = 30,
        prediction_horizon: float = 1.0,
        fps: float = 30.0,
        method: str = "quadratic",
    ) -> None:
        """
        Args:
            history_len: Số frame lưu lịch sử tọa độ chân.
            prediction_horizon: Dự đoán trước bao nhiêu giây.
            fps: Tốc độ khung hình của nguồn video.
            method: 'linear' hoặc 'quadratic'.
        """
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        self.fps = fps
        self.method = method
        self.horizon_frames = max(1, int(prediction_horizon * fps))

        # Bộ đệm nội bộ lưu tọa độ chân trái và phải riêng biệt
        self._left_history: collections.deque[tuple[int, int]] = collections.deque(maxlen=history_len)
        self._right_history: collections.deque[tuple[int, int]] = collections.deque(maxlen=history_len)

    # ------------------------------------------------------------------
    # Cập nhật vị trí chân mỗi frame
    # ------------------------------------------------------------------
    def update(self, keypoints: dict[str, tuple[int, int]]) -> None:
        """Ghi nhận vị trí mắt cá chân hiện tại vào lịch sử."""
        if "left_ankle" in keypoints:
            self._left_history.append(keypoints["left_ankle"])
        if "right_ankle" in keypoints:
            self._right_history.append(keypoints["right_ankle"])

    # ------------------------------------------------------------------
    # Dự đoán bước chân tiếp theo
    # ------------------------------------------------------------------
    def predict_next_footprint(
        self, foot_history: list[tuple[int, int]] | None = None
    ) -> tuple[int, int] | None:
        """
        Dự đoán tọa độ pixel (x, y) nơi bàn chân sẽ chạm đất.

        Args:
            foot_history: Nếu được cung cấp, dùng trực tiếp danh sách này.
                          Nếu None, tự động chọn chân có nhiều dữ liệu nhất.
        Returns:
            (x, y) hoặc None nếu không đủ dữ liệu.
        """
        if foot_history is None:
            foot_history = self._select_best_history()

        if foot_history is None or len(foot_history) < 3:
            return None

        if self.method == "quadratic" and len(foot_history) >= 5:
            return self._predict_quadratic(foot_history)
        return self._predict_linear(foot_history)

    # ------------------------------------------------------------------
    # Dự đoán cả hai chân
    # ------------------------------------------------------------------
    def predict_both_feet(self) -> dict[str, tuple[int, int] | None]:
        """Trả về dự đoán cho cả chân trái và chân phải."""
        result: dict[str, tuple[int, int] | None] = {
            "left": None,
            "right": None,
        }
        if len(self._left_history) >= 3:
            result["left"] = self._predict_from_deque(self._left_history)
        if len(self._right_history) >= 3:
            result["right"] = self._predict_from_deque(self._right_history)
        return result

    # ------------------------------------------------------------------
    # Ước lượng vận tốc di chuyển hiện tại (pixel / frame)
    # ------------------------------------------------------------------
    def estimate_velocity(self) -> float:
        """Trả về tốc độ di chuyển trung bình gần nhất (pixel/frame)."""
        history = self._select_best_history()
        if history is None or len(history) < 3:
            return 0.0

        pts = np.array(history[-6:], dtype=np.float64)
        deltas = np.diff(pts, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        return float(np.mean(speeds))

    # ------------------------------------------------------------------
    # Ước lượng hướng di chuyển (radian, 0 = phải, π/2 = xuống)
    # ------------------------------------------------------------------
    def estimate_heading(self) -> float | None:
        """Trả về góc hướng di chuyển (radian) hoặc None."""
        history = self._select_best_history()
        if history is None or len(history) < 3:
            return None

        pts = np.array(history[-6:], dtype=np.float64)
        deltas = np.diff(pts, axis=0)
        avg_delta = np.mean(deltas, axis=0)

        if np.linalg.norm(avg_delta) < 0.5:
            return None

        return float(math.atan2(avg_delta[1], avg_delta[0]))

    # ------------------------------------------------------------------
    # Ngoại suy tuyến tính
    # ------------------------------------------------------------------
    def _predict_linear(self, history: list[tuple[int, int]]) -> tuple[int, int]:
        pts = np.array(history, dtype=np.float64)
        window = pts[-min(8, len(pts)):]
        deltas = np.diff(window, axis=0)
        avg_v = np.mean(deltas, axis=0)

        current = pts[-1]
        future = current + avg_v * self.horizon_frames

        return (int(round(future[0])), int(round(future[1])))

    # ------------------------------------------------------------------
    # Hồi quy đa thức bậc 2
    # ------------------------------------------------------------------
    def _predict_quadratic(self, history: list[tuple[int, int]]) -> tuple[int, int]:
        pts = np.array(history, dtype=np.float64)
        window = pts[-min(15, len(pts)):]
        n = len(window)
        t = np.arange(n, dtype=np.float64)
        t_future = float(n - 1) + self.horizon_frames

        try:
            # Fit đa thức bậc 2 cho x(t) và y(t) riêng biệt
            coef_x = np.polyfit(t, window[:, 0], deg=2)
            coef_y = np.polyfit(t, window[:, 1], deg=2)

            pred_x = np.polyval(coef_x, t_future)
            pred_y = np.polyval(coef_y, t_future)

            return (int(round(pred_x)), int(round(pred_y)))

        except (np.linalg.LinAlgError, ValueError):
            # Nếu polyfit thất bại, quay về linear
            return self._predict_linear(history)

    # ------------------------------------------------------------------
    # Helpers nội bộ
    # ------------------------------------------------------------------
    def _predict_from_deque(
        self, dq: collections.deque[tuple[int, int]]
    ) -> tuple[int, int] | None:
        history = list(dq)
        if len(history) < 3:
            return None
        if self.method == "quadratic" and len(history) >= 5:
            return self._predict_quadratic(history)
        return self._predict_linear(history)

    def _select_best_history(self) -> list[tuple[int, int]] | None:
        """Chọn chân có nhiều dữ liệu lịch sử hơn."""
        left = list(self._left_history)
        right = list(self._right_history)

        if len(left) >= len(right) and len(left) >= 3:
            return left
        if len(right) >= 3:
            return right
        if len(left) >= 3:
            return left
        return None
