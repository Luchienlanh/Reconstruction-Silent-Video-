# -*- coding: utf-8 -*-

"""
Visualization Module.
Vẽ trực quan tất cả các lớp thông tin phân tích lên khung hình video:
  - Hazard Mask overlay (tô đỏ bán trong suốt).
  - Skeleton keypoints & connections.
  - Quỹ đạo bước chân hiện tại và dự đoán.
  - Depth Map picture-in-picture.
  - Status Panel + Risk Bar + FPS + Time-to-Collision.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---- Liên kết khung xương để vẽ ----
SKELETON_CONNECTIONS: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
]

# Màu sắc chủ đề
_COL_SAFE = (0, 220, 80)
_COL_WARN = (0, 180, 255)
_COL_DANGER = (0, 0, 255)
_COL_BONE = (60, 230, 60)
_COL_JOINT = (255, 180, 30)
_COL_FACE = (255, 255, 100)
_COL_TRAJ = (0, 255, 255)
_COL_TEXT = (230, 230, 230)
_COL_DIM = (140, 140, 140)


class Visualizer:
    """Vẽ đồ họa trực quan lên frame OpenCV."""

    def __init__(
        self,
        show_hazard: bool = True,
        show_skeleton: bool = True,
        show_trajectory: bool = True,
        show_depth: bool = False,
        overlay_alpha: float = 0.35,
    ) -> None:
        self.show_hazard = show_hazard
        self.show_skeleton = show_skeleton
        self.show_trajectory = show_trajectory
        self.show_depth = show_depth
        self.overlay_alpha = overlay_alpha

        # Lưu lịch sử chân để vẽ vệt quỹ đạo
        self._trail_left: list[tuple[int, int]] = []
        self._trail_right: list[tuple[int, int]] = []
        self._max_trail = 40

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        risk_score: float,
        hazard_mask: np.ndarray | None,
        depth_map: np.ndarray | None,
        keypoints: dict[str, tuple[int, int]] | None,
        future_foot: tuple[int, int] | None,
        risk_cfg: dict,
        ttc: float | None = None,
    ) -> None:
        """Vẽ tất cả các lớp lên frame (in-place)."""

        # 1. Hazard overlay
        if self.show_hazard:
            self._draw_hazard(frame, hazard_mask)

        # 2. Depth PiP
        if self.show_depth:
            self._draw_depth_pip(frame, depth_map)

        # 3. Skeleton
        if self.show_skeleton:
            self._draw_skeleton(frame, keypoints)

        # 4. Trajectory trail + future foot
        if self.show_trajectory:
            self._draw_trajectory(frame, keypoints, future_foot)

        # 5. Status panel
        self._draw_status_panel(frame, risk_score, risk_cfg, ttc)

    # ------------------------------------------------------------------
    # Hazard Mask
    # ------------------------------------------------------------------
    def _draw_hazard(self, frame: np.ndarray, mask: np.ndarray | None) -> None:
        if mask is None or not mask.any():
            return

        overlay = frame.copy()
        # Tô đỏ vùng vật cản
        overlay[mask > 0] = _COL_DANGER

        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)

        # Vẽ viền contour quanh vật cản
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 180), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Skeleton
    # ------------------------------------------------------------------
    def _draw_skeleton(
        self, frame: np.ndarray, kp: dict[str, tuple[int, int]] | None
    ) -> None:
        if not kp:
            return

        # Bones
        for a, b in SKELETON_CONNECTIONS:
            if a in kp and b in kp:
                cv2.line(frame, kp[a], kp[b], _COL_BONE, 2, cv2.LINE_AA)

        # Joints
        for name, pt in kp.items():
            is_face = any(f in name for f in ("nose", "eye", "ear", "mouth"))
            color = _COL_FACE if is_face else _COL_JOINT
            radius = 2 if is_face else 4
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Trajectory trail + future footprint
    # ------------------------------------------------------------------
    def _draw_trajectory(
        self,
        frame: np.ndarray,
        kp: dict[str, tuple[int, int]] | None,
        future_foot: tuple[int, int] | None,
    ) -> None:
        if kp:
            if "left_ankle" in kp:
                self._trail_left.append(kp["left_ankle"])
                if len(self._trail_left) > self._max_trail:
                    self._trail_left = self._trail_left[-self._max_trail:]
            if "right_ankle" in kp:
                self._trail_right.append(kp["right_ankle"])
                if len(self._trail_right) > self._max_trail:
                    self._trail_right = self._trail_right[-self._max_trail:]

        # Vẽ vệt quỹ đạo dần mờ (fade)
        self._draw_fading_trail(frame, self._trail_left, (180, 255, 180))
        self._draw_fading_trail(frame, self._trail_right, (180, 220, 255))

        # Bước chân dự đoán
        if future_foot is not None:
            # Mũi tên từ vị trí chân hiện tại
            current = None
            if kp:
                current = kp.get("left_ankle") or kp.get("right_ankle")
            if current is not None:
                cv2.arrowedLine(frame, current, future_foot, _COL_TRAJ, 2, cv2.LINE_AA, tipLength=0.18)

            # Vòng tròn hồng tâm
            cv2.circle(frame, future_foot, 6, _COL_TRAJ, -1, cv2.LINE_AA)
            cv2.circle(frame, future_foot, 12, _COL_TRAJ, 2, cv2.LINE_AA)
            cv2.circle(frame, future_foot, 18, _COL_TRAJ, 1, cv2.LINE_AA)
            cv2.putText(
                frame, "PREDICTED",
                (future_foot[0] + 22, future_foot[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, _COL_TRAJ, 1, cv2.LINE_AA,
            )

    @staticmethod
    def _draw_fading_trail(
        frame: np.ndarray,
        trail: list[tuple[int, int]],
        color: tuple[int, int, int],
    ) -> None:
        n = len(trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = i / n  # 0 → 1 (mờ → rõ)
            thickness = max(1, int(alpha * 3))
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, trail[i - 1], trail[i], c, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Depth Map PiP
    # ------------------------------------------------------------------
    def _draw_depth_pip(self, frame: np.ndarray, depth_map: np.ndarray | None) -> None:
        if depth_map is None:
            return

        h, w = frame.shape[:2]
        pip_h, pip_w = int(h * 0.20), int(w * 0.20)

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-5:
            norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(depth_map, dtype=np.uint8)

        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        pip = cv2.resize(colored, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

        margin = 12
        x0 = w - pip_w - margin
        y0 = margin

        frame[y0:y0 + pip_h, x0:x0 + pip_w] = pip
        cv2.rectangle(frame, (x0, y0), (x0 + pip_w, y0 + pip_h), _COL_DIM, 1)
        cv2.putText(
            frame, "DEPTH", (x0 + 4, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Status Panel + Risk Bar
    # ------------------------------------------------------------------
    def _draw_status_panel(
        self,
        frame: np.ndarray,
        risk_score: float,
        risk_cfg: dict,
        ttc: float | None = None,
    ) -> None:
        h, w = frame.shape[:2]
        warn_t = risk_cfg.get("warning_threshold", 0.70)
        danger_t = risk_cfg.get("danger_threshold", 0.85)

        # Trạng thái
        if risk_score >= danger_t:
            status = "DANGER - TRIP HAZARD"
            col = _COL_DANGER
        elif risk_score >= warn_t:
            status = "WARNING - OBSTACLE AHEAD"
            col = _COL_WARN
        else:
            status = "SAFE"
            col = _COL_SAFE

        # Panel nền
        pw = min(580, w - 100)
        ph = 108
        panel = frame.copy()
        cv2.rectangle(panel, (12, 12), (12 + pw, 12 + ph), (10, 10, 10), -1)
        cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (12, 12), (12 + pw, 12 + ph), (70, 70, 70), 1)

        # Dòng 1: trạng thái
        cv2.putText(frame, status, (26, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, 2, cv2.LINE_AA)

        # Dòng 2: risk score
        cv2.putText(
            frame, f"Risk Score: {risk_score * 100:.1f}%",
            (26, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _COL_TEXT, 1, cv2.LINE_AA,
        )

        # Dòng 3: TTC nếu có
        if ttc is not None and ttc < 100.0:
            ttc_text = f"Time to collision: {ttc:.1f}s"
            ttc_col = _COL_DANGER if ttc < 1.5 else _COL_WARN if ttc < 3.0 else _COL_TEXT
            cv2.putText(
                frame, ttc_text,
                (26, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ttc_col, 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame, "Press Q / ESC to exit",
                (26, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _COL_DIM, 1, cv2.LINE_AA,
            )

        # ---- Vertical Risk Bar (cạnh phải) ----
        bar_w, bar_h = 22, h - 80
        bar_x = w - bar_w - 18
        bar_y = 40

        bar_bg = frame.copy()
        cv2.rectangle(bar_bg, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 20, 20), -1)
        cv2.addWeighted(bar_bg, 0.6, frame, 0.4, 0, frame)

        filled = int(bar_h * min(risk_score, 1.0))
        if filled > 0:
            fill_col = _COL_SAFE if risk_score < warn_t else _COL_WARN if risk_score < danger_t else _COL_DANGER
            cv2.rectangle(
                frame,
                (bar_x, bar_y + bar_h - filled),
                (bar_x + bar_w, bar_y + bar_h),
                fill_col, -1,
            )

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

        # Vạch chia 50% và 75%
        for pct in (0.50, 0.75):
            ty = int(bar_y + bar_h * (1.0 - pct))
            cv2.line(frame, (bar_x, ty), (bar_x + bar_w, ty), (90, 90, 90), 1)
            cv2.putText(
                frame, f"{int(pct * 100)}",
                (bar_x - 28, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, _COL_DIM, 1, cv2.LINE_AA,
            )
