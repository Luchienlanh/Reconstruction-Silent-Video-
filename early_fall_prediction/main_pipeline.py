#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Pipeline – Early Fall Risk Prediction System.

Đây là file điều phối trung tâm ("trái tim") của hệ thống.
Nhiệm vụ:
  1. Đọc cấu hình từ config.yaml.
  2. Khởi tạo tất cả các module con (Terrain, Depth, Pose, Trajectory,
     SNN Temporal, KAN Risk Fusion).
  3. Mở luồng camera / video và chạy vòng lặp xử lý theo từng frame.
  4. Gom kết quả từ các module, tính Risk Score, và kích hoạt cảnh báo
     sớm khi điểm nguy cơ vượt ngưỡng.
  5. Gọi module visualization để hiển thị kết quả lên màn hình.

Cách chạy:
  python main_pipeline.py
  python main_pipeline.py --config custom_config.yaml
  python main_pipeline.py --source video.mp4
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Project root setup – đảm bảo import được các module con
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy imports cho các module con
# Mỗi module sẽ được import khi file tương ứng đã được code xong.
# Trước đó, pipeline vẫn chạy được bằng các hàm stub bên dưới.
# ---------------------------------------------------------------------------
try:
    from modules.terrain_perception import TerrainPerception
except ImportError:
    TerrainPerception = None  # type: ignore[assignment,misc]

try:
    from modules.depth_estimation import DepthEstimation
except ImportError:
    DepthEstimation = None  # type: ignore[assignment,misc]

try:
    from modules.pose_and_gait import PoseGaitTracker
except ImportError:
    PoseGaitTracker = None  # type: ignore[assignment,misc]

try:
    from modules.trajectory_prediction import TrajectoryPredictor
except ImportError:
    TrajectoryPredictor = None  # type: ignore[assignment,misc]

try:
    from modules.temporal_snn import TemporalSNN
except ImportError:
    TemporalSNN = None  # type: ignore[assignment,misc]

try:
    from modules.risk_fusion_kan import RiskFusionKAN
except ImportError:
    RiskFusionKAN = None  # type: ignore[assignment,misc]

try:
    from utils.visualization import Visualizer
except ImportError:
    Visualizer = None  # type: ignore[assignment,misc]

try:
    from utils.geometry_math import (
        foot_inside_hazard,
        distance_foot_to_nearest_hazard,
    )
except ImportError:
    foot_inside_hazard = None
    distance_foot_to_nearest_hazard = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EarlyFallPrediction")


# ===================================================================
#  1. ĐỌC CẤU HÌNH
# ===================================================================
def load_config(config_path: str | Path) -> dict:
    """Đọc file config.yaml và trả về dictionary."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    logger.info("Config loaded from: %s", config_path)
    return config


# ===================================================================
#  2. KHỞI TẠO CÁC MODULE
# ===================================================================
def init_modules(config: dict) -> dict:
    """
    Khởi tạo từng module con dựa trên config.
    Nếu module chưa được code (import = None), ghi log cảnh báo
    và gán giá trị None → pipeline sẽ bỏ qua module đó.
    """
    modules: dict = {}
    model_paths = config.get("model_paths", {})

    # --- Terrain Perception (YOLOv8-seg) ---
    if TerrainPerception is not None:
        modules["terrain"] = TerrainPerception(
            model_path=model_paths.get("terrain_seg", ""),
            hazard_classes=config.get("terrain", {}).get("hazard_classes", []),
            confidence=config.get("terrain", {}).get("confidence_threshold", 0.45),
        )
        logger.info("✔ Terrain Perception module loaded.")
    else:
        modules["terrain"] = None
        logger.warning("✘ Terrain Perception module NOT available (chưa code).")

    # --- Depth Estimation ---
    if DepthEstimation is not None:
        modules["depth"] = DepthEstimation(
            model_path=model_paths.get("depth", ""),
            encoder=config.get("depth", {}).get("encoder", "vits"),
        )
        logger.info("✔ Depth Estimation module loaded.")
    else:
        modules["depth"] = None
        logger.warning("✘ Depth Estimation module NOT available (chưa code).")

    # --- Pose & Gait ---
    if PoseGaitTracker is not None:
        pose_cfg = config.get("pose", {})
        modules["pose"] = PoseGaitTracker(
            model_path=model_paths.get("pose", ""),
            model_complexity=pose_cfg.get("model_complexity", 1),
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
        )
        logger.info("✔ Pose & Gait module loaded.")
    else:
        modules["pose"] = None
        logger.warning("✘ Pose & Gait module NOT available (chưa code).")

    # --- Trajectory Prediction ---
    if TrajectoryPredictor is not None:
        traj_cfg = config.get("trajectory", {})
        modules["trajectory"] = TrajectoryPredictor(
            history_len=traj_cfg.get("history_length", 30),
            prediction_horizon=traj_cfg.get("prediction_horizon_sec", 1.0),
        )
        logger.info("✔ Trajectory Prediction module loaded.")
    else:
        modules["trajectory"] = None
        logger.warning("✘ Trajectory Prediction module NOT available (chưa code).")

    # --- Temporal SNN ---
    if TemporalSNN is not None:
        modules["snn"] = TemporalSNN(
            model_path=model_paths.get("snn_temporal", ""),
        )
        logger.info("✔ Temporal SNN module loaded.")
    else:
        modules["snn"] = None
        logger.warning("✘ Temporal SNN module NOT available (chưa code).")

    # --- Risk Fusion KAN ---
    if RiskFusionKAN is not None:
        modules["kan"] = RiskFusionKAN(
            model_path=model_paths.get("kan_fusion", ""),
        )
        logger.info("✔ Risk Fusion KAN module loaded.")
    else:
        modules["kan"] = None
        logger.warning("✘ Risk Fusion KAN module NOT available (chưa code).")

    # --- Visualizer ---
    if Visualizer is not None:
        vis_cfg = config.get("visualization", {})
        modules["visualizer"] = Visualizer(
            show_hazard=vis_cfg.get("show_hazard_overlay", True),
            show_skeleton=vis_cfg.get("show_skeleton", True),
            show_trajectory=vis_cfg.get("show_trajectory", True),
            show_depth=vis_cfg.get("show_depth_map", False),
            overlay_alpha=vis_cfg.get("overlay_alpha", 0.35),
        )
        logger.info("✔ Visualizer module loaded.")
    else:
        modules["visualizer"] = None
        logger.warning("✘ Visualizer module NOT available → dùng fallback vẽ cơ bản.")

    return modules


# ===================================================================
#  3. MỞ NGUỒN VIDEO (Camera hoặc File)
# ===================================================================
def open_video_source(config: dict, source_override: str | None = None) -> cv2.VideoCapture:
    """Mở camera hoặc file video dựa trên config hoặc CLI argument."""
    cam_cfg = config.get("camera", {})

    if source_override is not None:
        # File video hoặc camera index từ command line
        source = int(source_override) if source_override.isdigit() else source_override
    else:
        source = cam_cfg.get("device_index", 0)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        sys.exit(2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video source opened: %s → %dx%d", source, actual_w, actual_h)

    return cap


# ===================================================================
#  4. TÍNH RISK SCORE (RULE-BASED FALLBACK)
# ===================================================================
def compute_risk_score_fallback(
    hazard_mask: np.ndarray | None,
    foot_position: tuple[int, int] | None,
    future_foot: tuple[int, int] | None,
    gait_features: dict | None,
    collision_distance_px: int,
) -> float:
    """
    Rule-based risk score khi chưa có SNN/KAN.
    Được thay thế bởi modules/risk_fusion_kan.py ở Phase 4.

    Logic:
      - Nếu không có hazard_mask hoặc foot_position → risk = 0.0
      - Tính khoảng cách từ chân (hiện tại & dự đoán) đến vùng hazard
      - Kết hợp với gait instability (độ nghiêng thân) để cho ra risk 0..1
    """
    if hazard_mask is None or foot_position is None:
        return 0.0

    risk = 0.0

    # --- Chân hiện tại gần vật cản? ---
    if distance_foot_to_nearest_hazard is not None:
        dist_current = distance_foot_to_nearest_hazard(hazard_mask, foot_position)
    else:
        # Stub: check pixel tại vị trí chân
        fx, fy = foot_position
        h, w = hazard_mask.shape[:2]
        fx, fy = max(0, min(fx, w - 1)), max(0, min(fy, h - 1))
        dist_current = float("inf") if hazard_mask[fy, fx] == 0 else 0.0

    if dist_current < collision_distance_px:
        risk += 0.5 * max(0.0, 1.0 - dist_current / collision_distance_px)

    # --- Quỹ đạo tương lai cắt vào vùng hazard? ---
    if future_foot is not None:
        if foot_inside_hazard is not None:
            if foot_inside_hazard(hazard_mask, future_foot):
                risk += 0.35
        else:
            ffx, ffy = future_foot
            h, w = hazard_mask.shape[:2]
            ffx, ffy = max(0, min(ffx, w - 1)), max(0, min(ffy, h - 1))
            if hazard_mask[ffy, ffx] > 0:
                risk += 0.35

    # --- Dáng đi bất ổn? ---
    if gait_features is not None:
        body_tilt = abs(gait_features.get("body_tilt", 0.0))
        if body_tilt > 15.0:  # Nghiêng > 15 độ
            risk += 0.15

    return min(risk, 1.0)


# ===================================================================
#  5. VẼ FALLBACK (khi chưa có utils/visualization.py)
# ===================================================================
def draw_fallback_overlay(
    frame: np.ndarray,
    risk_score: float,
    hazard_mask: np.ndarray | None,
    keypoints: dict | None,
    future_foot: tuple[int, int] | None,
    risk_cfg: dict,
    alpha: float = 0.35,
) -> None:
    """Vẽ trực tiếp lên frame khi Visualizer chưa sẵn sàng."""
    h, w = frame.shape[:2]

    # --- Tô đỏ vùng hazard ---
    if hazard_mask is not None and hazard_mask.any():
        overlay = frame.copy()
        overlay[hazard_mask > 0] = (0, 0, 220)  # Đỏ
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # --- Vẽ quỹ đạo bước chân dự đoán ---
    if future_foot is not None:
        cv2.circle(frame, future_foot, 10, (0, 255, 255), 2)  # Vàng
        cv2.putText(
            frame, "PREDICTED", (future_foot[0] + 14, future_foot[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
        )

    # --- Panel trạng thái ---
    warning_thresh = risk_cfg.get("warning_threshold", 0.70)
    danger_thresh = risk_cfg.get("danger_threshold", 0.85)

    if risk_score >= danger_thresh:
        status_text = "DANGER - OBSTACLE AHEAD!"
        status_color = (0, 0, 255)       # Đỏ
    elif risk_score >= warning_thresh:
        status_text = "WARNING - Watch your step"
        status_color = (0, 165, 255)     # Cam
    else:
        status_text = "SAFE"
        status_color = (0, 200, 0)       # Xanh lá

    panel_w = min(520, w - 20)
    cv2.rectangle(frame, (10, 10), (panel_w, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (panel_w, 90), (80, 80, 80), 1)

    cv2.putText(
        frame, status_text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2,
    )
    cv2.putText(
        frame, f"Risk: {risk_score:.2f}", (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )

    # --- Thanh risk bar ---
    bar_x, bar_y, bar_w, bar_h = w - 50, 30, 20, h - 60
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

    filled_h = int(bar_h * risk_score)
    if filled_h > 0:
        bar_color = (
            (0, 200, 0) if risk_score < warning_thresh
            else (0, 165, 255) if risk_score < danger_thresh
            else (0, 0, 255)
        )
        cv2.rectangle(
            frame,
            (bar_x, bar_y + bar_h - filled_h),
            (bar_x + bar_w, bar_y + bar_h),
            bar_color, -1,
        )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    # --- FPS ---
    # (FPS sẽ được tính và vẽ trong vòng lặp chính)


# ===================================================================
#  6. VÒNG LẶP XỬ LÝ CHÍNH
# ===================================================================
def run_pipeline(config: dict, modules: dict, cap: cv2.VideoCapture) -> int:
    """
    Vòng lặp chính: đọc frame → chạy module → tính risk → hiển thị.
    Trả về exit code.
    """
    terrain_mod = modules.get("terrain")
    depth_mod = modules.get("depth")
    pose_mod = modules.get("pose")
    traj_mod = modules.get("trajectory")
    snn_mod = modules.get("snn")
    kan_mod = modules.get("kan")
    vis_mod = modules.get("visualizer")

    # Cấu hình xử lý
    terrain_cfg = config.get("terrain", {})
    risk_cfg = config.get("risk", {})
    vis_cfg = config.get("visualization", {})

    detect_every = terrain_cfg.get("detect_every_n_frames", 5)
    depth_every = config.get("depth", {}).get("run_every_n_frames", 5)
    collision_dist = risk_cfg.get("collision_distance_px", 120)
    alert_cooldown = risk_cfg.get("alert_cooldown_sec", 3.0)

    # Bộ đệm lịch sử
    traj_history_len = config.get("trajectory", {}).get("history_length", 30)
    foot_history: collections.deque[tuple[int, int]] = collections.deque(maxlen=traj_history_len)

    # Trạng thái giữa các frame
    hazard_mask: np.ndarray | None = None
    depth_map: np.ndarray | None = None
    keypoints: dict | None = None
    gait_features: dict | None = None
    foot_position: tuple[int, int] | None = None
    future_foot: tuple[int, int] | None = None
    risk_score: float = 0.0
    last_alert_time: float = 0.0

    frame_index: int = 0
    fps_counter: collections.deque[float] = collections.deque(maxlen=30)

    window_title = "Early Fall Risk Prediction"
    logger.info("Pipeline started. Press 'q' or ESC to quit.")

    while True:
        t_start = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            logger.info("End of video stream.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------------------------------
        # BƯỚC 1: Terrain Perception (mỗi N frames)
        # -------------------------------------------------------
        if terrain_mod is not None and frame_index % detect_every == 0:
            hazard_mask = terrain_mod.detect_hazards(rgb)

        # -------------------------------------------------------
        # BƯỚC 2: Depth Estimation (mỗi N frames)
        # -------------------------------------------------------
        if depth_mod is not None and frame_index % depth_every == 0:
            depth_map = depth_mod.estimate_depth(rgb)

        # -------------------------------------------------------
        # BƯỚC 3: Pose & Gait Analysis (mỗi frame)
        # -------------------------------------------------------
        if pose_mod is not None:
            keypoints, pose_found = pose_mod.track_pose(rgb)
            if pose_found and keypoints:
                gait_features = pose_mod.calculate_gait_features(keypoints)
                # Lấy vị trí bàn chân (ankle/foot keypoint)
                foot_position = keypoints.get("left_ankle") or keypoints.get("right_ankle")
                if foot_position is not None:
                    foot_history.append(foot_position)

        # -------------------------------------------------------
        # BƯỚC 4: Trajectory Prediction
        # -------------------------------------------------------
        if traj_mod is not None and len(foot_history) >= 5:
            future_foot = traj_mod.predict_next_footprint(list(foot_history))

        # -------------------------------------------------------
        # BƯỚC 5: Temporal SNN (phân tích chuỗi thời gian)
        # -------------------------------------------------------
        temporal_feature = None
        if snn_mod is not None and keypoints is not None:
            temporal_feature = snn_mod.forward(keypoints, hazard_mask)

        # -------------------------------------------------------
        # BƯỚC 6: Risk Fusion → Risk Score
        # -------------------------------------------------------
        if kan_mod is not None:
            # Phase 4: Dùng KAN model thực sự
            risk_score = kan_mod.predict_risk(
                temporal_feature=temporal_feature,
                hazard_mask=hazard_mask,
                depth_map=depth_map,
                foot_position=foot_position,
                future_foot=future_foot,
                gait_features=gait_features,
            )
        else:
            # Phase 1-3: Dùng rule-based fallback
            risk_score = compute_risk_score_fallback(
                hazard_mask=hazard_mask,
                foot_position=foot_position,
                future_foot=future_foot,
                gait_features=gait_features,
                collision_distance_px=collision_dist,
            )

        # -------------------------------------------------------
        # BƯỚC 7: Phát cảnh báo âm thanh (nếu risk cao)
        # -------------------------------------------------------
        now = time.time()
        danger_thresh = risk_cfg.get("danger_threshold", 0.85)
        if risk_score >= danger_thresh and (now - last_alert_time) > alert_cooldown:
            logger.warning("⚠ DANGER ALERT! Risk = %.2f", risk_score)
            last_alert_time = now
            # TODO: Phát âm thanh cảnh báo (beep / gTTS)
            #       Ví dụ: os.system("paplay /usr/share/sounds/...")

        # -------------------------------------------------------
        # BƯỚC 8: Visualization
        # -------------------------------------------------------
        if vis_mod is not None:
            vis_mod.draw(
                frame=frame,
                risk_score=risk_score,
                hazard_mask=hazard_mask,
                depth_map=depth_map,
                keypoints=keypoints,
                future_foot=future_foot,
                risk_cfg=risk_cfg,
            )
        else:
            draw_fallback_overlay(
                frame=frame,
                risk_score=risk_score,
                hazard_mask=hazard_mask,
                keypoints=keypoints,
                future_foot=future_foot,
                risk_cfg=risk_cfg,
                alpha=vis_cfg.get("overlay_alpha", 0.35),
            )

        # --- FPS ---
        t_elapsed = time.perf_counter() - t_start
        fps_counter.append(t_elapsed)
        avg_fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0.0
        cv2.putText(
            frame, f"FPS: {avg_fps:.1f}", (w - 140, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # --- Hiển thị ---
        cv2.imshow(window_title, frame)
        frame_index += 1

        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), 27}:  # q hoặc ESC
            logger.info("User requested quit.")
            break

    return 0


# ===================================================================
#  7. CLI ARGUMENTS
# ===================================================================
def parse_args() -> argparse.Namespace:
    """Xử lý tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Early Fall Risk Prediction – Main Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Override video source: camera index (0,1,...) or path to video file.",
    )
    return parser.parse_args()


# ===================================================================
#  8. ENTRY POINT
# ===================================================================
def main() -> int:
    """Hàm chính – được gọi khi chạy `python main_pipeline.py`."""
    # Đảm bảo console hỗ trợ UTF-8
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # 1. Đọc config
    config = load_config(args.config)

    # 2. Khởi tạo modules
    modules = init_modules(config)

    # 3. Mở video source
    cap = open_video_source(config, source_override=args.source)

    # 4. Chạy pipeline
    try:
        exit_code = run_pipeline(config, modules, cap)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
        exit_code = 0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline shutdown. Resources released.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
