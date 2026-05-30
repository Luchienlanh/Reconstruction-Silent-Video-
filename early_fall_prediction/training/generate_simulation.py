# -*- coding: utf-8 -*-

"""
Generate Simulation Data for Early Fall Prediction System.
Sinh dữ liệu giả lập dáng đi (Safe, Warning, Danger) bằng mô hình toán học.
Xuất ra file data/dataset.npz và data/labels.csv phục vụ huấn luyện.
"""

import argparse
import csv
import logging
import math
import os
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("GenerateSimulation")

# 18 joints matching modules/temporal_snn.py and prepare_dataset.py
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

def generate_sequence(label: int, seq_len: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinh 1 chuỗi dữ liệu (skeleton và features) cho một nhãn cụ thể.
    Labels:
      0 = Safe (đi bộ bình thường, thẳng đứng, không có vật cản)
      1 = Warning (đi bộ hơi nghiêng, có chướng ngại vật phía xa)
      2 = Danger (mất thăng bằng, vấp ngã hoặc sát chướng ngại vật)
    """
    # Khởi tạo mảng
    skeletons = np.zeros((seq_len, len(JOINT_NAMES) * 2), dtype=np.float32)
    features = np.zeros((seq_len, 4), dtype=np.float32)

    # Random hóa thông số ban đầu của người
    # Chiều cao và vị trí ngẫu nhiên
    h = 720
    w = 1280
    cx_start = np.random.uniform(400, 600)
    cy_hip = np.random.uniform(380, 420)
    
    # Chu kỳ đi bộ (walk cycle)
    walk_freq = np.random.uniform(0.8, 1.2)  # Hz
    dt = 1.0 / 30.0  # 30 FPS
    omega = 2 * math.pi * walk_freq
    start_phase = np.random.uniform(0, 2 * math.pi)

    # Vận tốc di chuyển của nhân vật (pixel/frame)
    if label == 0:
        vel_x = np.random.uniform(3.0, 5.0)
    elif label == 1:
        vel_x = np.random.uniform(2.5, 4.0)
    else:  # Danger
        vel_x = np.random.uniform(2.0, 3.5)

    # Khoảng cách ban đầu tới chướng ngại vật (pixel)
    if label == 0:
        initial_hazard_dist = np.random.uniform(400, 600)
    elif label == 1:
        initial_hazard_dist = np.random.uniform(200, 350)
    else:  # Danger
        initial_hazard_dist = np.random.uniform(100, 180)

    # Định nghĩa offset mặc định của các khớp so với hip (0, 0)
    base_offsets = {
        "nose": (0, -130),
        "left_ear": (-10, -135),
        "left_shoulder": (-35, -90),
        "right_shoulder": (35, -90),
        "left_elbow": (-50, -45),
        "right_elbow": (50, -45),
        "left_wrist": (-55, -5),
        "right_wrist": (55, -5),
        "left_hip": (-22, 0),
        "right_hip": (22, 0),
        "left_knee": (-25, 50),
        "right_knee": (25, 50),
        "left_ankle": (-28, 100),
        "right_ankle": (28, 100),
        "left_heel": (-30, 105),
        "right_heel": (30, 105),
        "left_foot_index": (-22, 108),
        "right_foot_index": (22, 108),
    }

    # Sinh từng frame
    cy_hip_start = cy_hip
    for t in range(seq_len):
        phase = start_phase + t * omega * dt
        cy_hip = cy_hip_start
        
        # Dynamic offsets dictionary to allow updates during falls
        current_offsets = base_offsets.copy()
        
        # 1. Tính toán độ nghiêng thân và vận tốc hiện tại
        if label == 0:
            # Safe: Thẳng đứng, dao động rất nhỏ
            tilt_deg = np.random.normal(0, 1.5)
            curr_vel = vel_x
            asymmetry = np.random.uniform(0.01, 0.05)
            # Hazard cách xa
            dist_to_hazard = initial_hazard_dist - t * vel_x
            dist_to_hazard = max(10.0, dist_to_hazard)
            cx = cx_start + t * vel_x
        elif label == 1:
            # Warning: Hơi nghiêng người về phía trước
            tilt_deg = np.random.normal(8.0, 2.5)
            curr_vel = vel_x * (1.0 - 0.005 * t)  # Hơi giảm tốc độ
            asymmetry = np.random.uniform(0.05, 0.15)
            # Hazard tiến gần
            dist_to_hazard = initial_hazard_dist - t * curr_vel
            dist_to_hazard = max(10.0, dist_to_hazard)
            cx = cx_start + t * curr_vel
        else:  # Danger
            # Danger: Mất thăng bằng nghiêm trọng, ngã đổ hoặc dừng đột ngột
            # Mô phỏng quá trình ngã từ frame 10 trở đi
            if t < 10:
                tilt_deg = np.random.normal(12.0, 3.0)
                curr_vel = vel_x
                asymmetry = np.random.uniform(0.1, 0.25)
                cx = cx_start + t * curr_vel
            else:
                # Quá trình ngã vật lý (Inverted Pendulum collapse)
                fall_factor = (t - 10) / (seq_len - 10)
                tilt_deg = 12.0 + fall_factor * np.random.uniform(35.0, 48.0)
                # Tốc độ giảm mạnh về 0 (va chạm/stumble)
                curr_vel = vel_x * max(0.0, 1.0 - 1.2 * fall_factor)
                asymmetry = 0.2 + fall_factor * np.random.uniform(0.3, 0.5)
                cx = cx_start + 10 * vel_x + (t - 10) * curr_vel
                
                # --- COLLAPSE VẬT LÝ ---
                # Trọng tâm hạ thấp, mông sụp xuống sàn
                cy_hip = cy_hip_start + 120.0 * fall_factor
                
                # Cập nhật relative offsets để chân không bị lún xuống sàn
                ankle_dy = 100.0 - 110.0 * fall_factor
                knee_dy = ankle_dy * 0.5
                
                current_offsets.update({
                    # Đầu gối gập mạnh (khuỵu xuống)
                    "left_knee": (-25 + 35.0 * fall_factor, knee_dy),
                    "right_knee": (25 + 35.0 * fall_factor, knee_dy),
                    # Bàn chân co cụm sát mặt đất và thụt lùi so với hông đang đổ về trước
                    "left_ankle": (-28 - 15.0 * fall_factor, ankle_dy),
                    "right_ankle": (28 - 15.0 * fall_factor, ankle_dy),
                    "left_heel": (-30 - 15.0 * fall_factor, ankle_dy + 5.0),
                    "right_heel": (30 - 15.0 * fall_factor, ankle_dy + 5.0),
                    "left_foot_index": (-22 - 15.0 * fall_factor, ankle_dy + 8.0),
                    "right_foot_index": (22 - 15.0 * fall_factor, ankle_dy + 8.0),
                    # Tay vung lên phía trước để cản mặt đất (flailing arms)
                    "left_elbow": (-50 - 20.0 * fall_factor, -45 - 20.0 * fall_factor),
                    "right_elbow": (50 + 20.0 * fall_factor, -45 - 20.0 * fall_factor),
                    "left_wrist": (-55 - 40.0 * fall_factor, -5 - 55.0 * fall_factor),
                    "right_wrist": (55 + 40.0 * fall_factor, -5 - 55.0 * fall_factor),
                })

            dist_to_hazard = initial_hazard_dist - (10 * vel_x + (t - 10) * curr_vel)
            dist_to_hazard = max(0.0, dist_to_hazard)

        tilt_rad = math.radians(tilt_deg)

        # 2. Xây dựng tọa độ khớp cho frame t
        frame_keypoints = {}
        
        # Upper body joints (rotate around hip pivot)
        upper_body = {"nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_ear"}
        
        for name, (dx, dy) in current_offsets.items():
            if name in upper_body:
                # Rotate upper body by tilt_rad
                rx = cx + dx * math.cos(tilt_rad) - dy * math.sin(tilt_rad)
                ry = cy_hip + dx * math.sin(tilt_rad) + dy * math.cos(tilt_rad)
            else:
                # Lower body: Apply walking dynamics (oscillation)
                x_osc = 0.0
                y_osc = 0.0
                
                # Đối với Danger khi đang ngã, ta dùng trực tiếp offsets tĩnh đã gập để mô phỏng sụp đổ,
                # còn khi bình thường/safe/warning thì dùng walk cycle dao động.
                if not (label == 2 and t >= 10):
                    is_left = name.startswith("left")
                    p = phase if is_left else (phase + math.pi)

                    if "knee" in name:
                        x_osc = 10 * math.sin(p)
                        y_osc = -6 * math.cos(2 * p)
                    elif "ankle" in name or "heel" in name or "foot_index" in name:
                        x_osc = 25 * math.sin(p)
                        # Lift the foot during swing phase
                        y_osc = -18 * math.sin(p) if math.sin(p) > 0 else 0.0

                # Apply offset and oscillation
                rx = cx + dx + x_osc
                ry = cy_hip + dy + y_osc

            # Thêm nhiễu Gaussian nhỏ (nhiễu camera)
            noise_x = np.random.normal(0, 1.5)
            noise_y = np.random.normal(0, 1.5)
            
            frame_keypoints[name] = (int(rx + noise_x), int(ry + noise_y))

        # Flatten and normalize frame keypoints into the 36-d vector in exact order
        lh = frame_keypoints.get("left_hip")
        rh = frame_keypoints.get("right_hip")
        center_x, center_y = 640.0, 400.0
        if lh and rh:
            center_x = (lh[0] + rh[0]) / 2.0
            center_y = (lh[1] + rh[1]) / 2.0
        elif lh:
            center_x, center_y = lh
        elif rh:
            center_x, center_y = rh
        
        scale = 200.0
        vec = []
        for name in JOINT_NAMES:
            pt = frame_keypoints.get(name, (0, 0))
            vec.extend([
                (float(pt[0]) - center_x) / scale,
                (float(pt[1]) - center_y) / scale
            ])
        skeletons[t] = vec

        # 3. Tính toán 4 đặc trưng hình học cho KAN
        norm_dist = max(0.0, 1.0 - dist_to_hazard / 300.0)
        norm_tilt = min(1.0, abs(tilt_deg) / 45.0)
        norm_vel = min(1.0, curr_vel / 20.0)
        asym = min(1.0, asymmetry)

        features[t] = [norm_dist, norm_tilt, norm_vel, asym]

    return skeletons, features

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gait dataset.")
    parser.add_argument("--num-safe", type=int, default=2000, help="Number of safe samples.")
    parser.add_argument("--num-warning", type=int, default=2000, help="Number of warning samples.")
    parser.add_argument("--num-danger", type=int, default=2000, help="Number of danger samples.")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory.")
    args = parser.parse_args()

    # Tạo thư mục đầu ra
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_samples = args.num_safe + args.num_warning + args.num_danger
    logger.info("Bắt đầu sinh %d mẫu dữ liệu mô phỏng...", total_samples)

    all_skeletons = []
    all_features = []
    all_labels = []
    csv_rows = []

    # Safe (0)
    logger.info("Sinh %d mẫu Safe...", args.num_safe)
    for i in range(args.num_safe):
        skel, feat = generate_sequence(label=0, seq_len=args.seq_len)
        all_skeletons.append(skel)
        all_features.append(feat)
        all_labels.append(0)
        csv_rows.append([f"sim_safe_{i:04d}", 0])

    # Warning (1)
    logger.info("Sinh %d mẫu Warning...", args.num_warning)
    for i in range(args.num_warning):
        skel, feat = generate_sequence(label=1, seq_len=args.seq_len)
        all_skeletons.append(skel)
        all_features.append(feat)
        all_labels.append(1)
        csv_rows.append([f"sim_warn_{i:04d}", 1])

    # Danger (2)
    logger.info("Sinh %d mẫu Danger...", args.num_danger)
    for i in range(args.num_danger):
        skel, feat = generate_sequence(label=2, seq_len=args.seq_len)
        all_skeletons.append(skel)
        all_features.append(feat)
        all_labels.append(2)
        csv_rows.append([f"sim_danger_{i:04d}", 2])

    # Convert to numpy arrays
    skeletons_arr = np.array(all_skeletons, dtype=np.float32)
    features_arr = np.array(all_features, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Save to NPZ
    npz_path = output_path / "dataset.npz"
    np.savez_compressed(
        str(npz_path),
        skeletons=skeletons_arr,
        features=features_arr,
        labels=labels_arr
    )
    logger.info("Đã lưu file NPZ: %s (skeletons: %s, features: %s, labels: %s)", 
                npz_path, skeletons_arr.shape, features_arr.shape, labels_arr.shape)

    # Save labels to CSV
    csv_path = output_path / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "label"])
        writer.writerows(csv_rows)
    logger.info("Đã lưu file labels CSV: %s", csv_path)

    logger.info("Hoàn tất sinh dữ liệu mô phỏng!")

if __name__ == "__main__":
    main()
