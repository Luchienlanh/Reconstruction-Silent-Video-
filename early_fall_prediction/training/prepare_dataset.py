# -*- coding: utf-8 -*-

"""
Dataset Preparation – Chuẩn bị dữ liệu huấn luyện.

Chức năng:
  1. Đọc video (synthetic hoặc real), chạy MediaPipe Pose để trích xuất skeleton.
  2. Chạy YOLOv8 để tạo hazard mask cho mỗi frame.
  3. Tính các đặc trưng hình học (khoảng cách chân ↔ vật cản, body tilt, ...).
  4. Gom thành chuỗi thời gian và lưu ra file .npz để train SNN / KAN.

Cách dùng:
  python training/prepare_dataset.py --video-dir data/videos/ --output data/dataset.npz
  python training/prepare_dataset.py --video-dir data/videos/ --output data/dataset.npz --label-file data/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Thêm project root vào path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.pose_and_gait import PoseGaitTracker, LANDMARK_MAP
from modules.terrain_perception import TerrainPerception
from utils.geometry_math import distance_foot_to_nearest_hazard

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PrepareDataset")


def extract_skeleton_vector(keypoints: dict[str, tuple[int, int]]) -> list[float]:
    """Chuyển dict keypoints → vector 36-d cố định thứ tự."""
    JOINT_NAMES = [
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
        "left_ear",
    ]
    center_x, center_y = 640.0, 400.0
    lh = keypoints.get("left_hip")
    rh = keypoints.get("right_hip")
    if lh and rh:
        center_x = (lh[0] + rh[0]) / 2.0
        center_y = (lh[1] + rh[1]) / 2.0
    elif lh:
        center_x, center_y = lh
    elif rh:
        center_x, center_y = rh
    
    scale = 200.0
    vec: list[float] = []
    for name in JOINT_NAMES:
        if name in keypoints:
            pt = keypoints[name]
            vec.extend([
                (float(pt[0]) - center_x) / scale,
                (float(pt[1]) - center_y) / scale
            ])
        else:
            vec.extend([0.0, 0.0])
    return vec


def extract_geometric_features(
    keypoints: dict[str, tuple[int, int]],
    gait: dict[str, float],
    hazard_mask: np.ndarray | None,
) -> list[float]:
    """Trích xuất 4 đặc trưng hình học cho KAN."""
    # 1. Khoảng cách chân → vật cản
    foot = keypoints.get("left_ankle") or keypoints.get("right_ankle")
    dist = float("inf")
    if foot is not None and hazard_mask is not None:
        dist = distance_foot_to_nearest_hazard(hazard_mask, foot)
    norm_dist = max(0.0, 1.0 - dist / 300.0)

    # 2. Body tilt
    norm_tilt = min(1.0, abs(gait.get("body_tilt", 0.0)) / 45.0)

    # 3. Step velocity (chuẩn hóa thô)
    norm_vel = min(1.0, gait.get("step_velocity", 0.0) / 20.0)

    # 4. Gait symmetry (đảo: bất đối xứng cao → nguy hiểm)
    asym = 1.0 - gait.get("gait_symmetry", 1.0)

    return [norm_dist, norm_tilt, norm_vel, asym]


def process_video(
    video_path: str,
    pose_tracker: PoseGaitTracker,
    terrain: TerrainPerception | None,
    seq_len: int = 30,
    stride: int = 10,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Xử lý 1 video → danh sách chuỗi skeleton (cho SNN) và feature (cho KAN).

    Returns:
        skeleton_sequences: list of [seq_len, 36]
        feature_sequences:  list of [seq_len, 4]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Không mở được video: %s", video_path)
        return [], []

    all_skeletons: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    frame_skeletons: list[list[float]] = []
    frame_features: list[list[float]] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose
        kp, found = pose_tracker.track_pose(rgb)
        if not found:
            skeleton_vec = [0.0] * 36
            geo_vec = [0.0] * 4
        else:
            skeleton_vec = extract_skeleton_vector(kp)
            gait = pose_tracker.calculate_gait_features(kp)

            # Hazard mask (chạy mỗi 5 frame)
            hazard_mask = None
            if terrain is not None and frame_idx % 5 == 0:
                hazard_mask = terrain.detect_hazards(rgb)

            geo_vec = extract_geometric_features(kp, gait, hazard_mask)

        frame_skeletons.append(skeleton_vec)
        frame_features.append(geo_vec)
        frame_idx += 1

    cap.release()

    # Cắt thành chuỗi sliding window
    total = len(frame_skeletons)
    for start in range(0, total - seq_len + 1, stride):
        end = start + seq_len
        skel_seq = np.array(frame_skeletons[start:end], dtype=np.float32)
        feat_seq = np.array(frame_features[start:end], dtype=np.float32)
        all_skeletons.append(skel_seq)
        all_features.append(feat_seq)

    logger.info("Video %s: %d frames → %d sequences", video_path, total, len(all_skeletons))
    return all_skeletons, all_features


def load_labels(label_file: str) -> dict[str, int]:
    """
    Đọc file CSV gán nhãn: video_name, label (0=safe, 1=warning, 2=danger).
    """
    labels: dict[str, int] = {}
    with open(label_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("video") or row.get("video_name") or row.get("filename", "")
            lab = int(row.get("label", row.get("risk_level", 0)))
            labels[name] = lab
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training dataset from videos.")
    parser.add_argument("--video-dir", required=True, help="Thư mục chứa video.")
    parser.add_argument("--output", default="data/dataset.npz", help="File .npz đầu ra.")
    parser.add_argument("--label-file", default=None, help="CSV gán nhãn risk (tùy chọn).")
    parser.add_argument("--seq-len", type=int, default=30, help="Chiều dài chuỗi.")
    parser.add_argument("--stride", type=int, default=10, help="Bước nhảy sliding window.")
    parser.add_argument("--terrain-model", default="yolov8n-seg.pt", help="Weights YOLOv8.")
    args = parser.parse_args()

    # Khởi tạo modules
    pose = PoseGaitTracker(model_complexity=1)

    terrain: TerrainPerception | None = None
    try:
        terrain = TerrainPerception(model_path=args.terrain_model)
    except Exception:
        logger.warning("Không load được terrain model → bỏ qua hazard features.")

    # Tìm video
    video_dir = Path(args.video_dir)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in video_exts)
    logger.info("Tìm thấy %d video trong %s", len(video_files), video_dir)

    # Labels (nếu có)
    labels_map: dict[str, int] | None = None
    if args.label_file and os.path.exists(args.label_file):
        labels_map = load_labels(args.label_file)
        logger.info("Loaded %d labels from %s", len(labels_map), args.label_file)

    # Xử lý từng video
    all_skel: list[np.ndarray] = []
    all_feat: list[np.ndarray] = []
    all_labels: list[int] = []

    for vpath in video_files:
        skel_seqs, feat_seqs = process_video(
            str(vpath), pose, terrain,
            seq_len=args.seq_len, stride=args.stride,
        )
        all_skel.extend(skel_seqs)
        all_feat.extend(feat_seqs)

        # Gán label nếu có
        if labels_map is not None:
            lab = labels_map.get(vpath.name, labels_map.get(vpath.stem, 0))
            all_labels.extend([lab] * len(skel_seqs))

    if not all_skel:
        logger.error("Không trích xuất được dữ liệu nào!")
        return

    # Lưu
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = {
        "skeletons": np.array(all_skel),    # [N, seq_len, 36]
        "features": np.array(all_feat),     # [N, seq_len, 4]
    }
    if all_labels:
        save_dict["labels"] = np.array(all_labels)  # [N]

    np.savez_compressed(str(out_path), **save_dict)
    logger.info("Dataset saved → %s  |  %d samples", out_path, len(all_skel))


if __name__ == "__main__":
    main()
