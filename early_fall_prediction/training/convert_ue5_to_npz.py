import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.pose_and_gait import PoseGaitTracker

# Define the exact joint names used in train_snn / prepare_dataset
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

def extract_skeleton_vector(keypoints: dict[str, tuple[int, int]]) -> list[float]:
    """Chuyen dict keypoints -> vector 36-d chuan hoa theo hip."""
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
    vec = []
    for name in JOINT_NAMES:
        if name in keypoints:
            pt = keypoints[name]
            norm_x = (pt[0] - center_x) / scale
            norm_y = (pt[1] - center_y) / scale
            vec.extend([norm_x, norm_y])
        else:
            vec.extend([0.0, 0.0])
    return vec

def calculate_tilt_angle(keypoints: dict[str, tuple[int, int]]) -> float:
    """Tinh goc nghieng cua than de tu dong gan nhan."""
    l_shoulder = keypoints.get("left_shoulder")
    r_shoulder = keypoints.get("right_shoulder")
    l_hip = keypoints.get("left_hip")
    rh_hip = keypoints.get("right_hip")
    
    if l_shoulder and r_shoulder and l_hip and rh_hip:
        neck = ((l_shoulder[0] + r_shoulder[0]) / 2.0, (l_shoulder[1] + r_shoulder[1]) / 2.0)
        com = ((l_hip[0] + rh_hip[0]) / 2.0, (l_hip[1] + rh_hip[1]) / 2.0)
        dx = com[0] - neck[0]
        dy = com[1] - neck[1]
        angle = np.degrees(np.arctan2(dx, dy))
        return abs(angle)
    return 0.0

def extract_features_from_skeleton(skeleton_seq):
    """
    Tich hop trich xuat dac trung tuong tu generate_simulation.py
    """
    # skeleton_seq: shape (seq_len, 36) -> (seq_len, 18, 2)
    seq = skeleton_seq.reshape(-1, len(JOINT_NAMES), 2)
    features = []
    
    for frame_idx in range(len(seq)):
        frame = seq[frame_idx]
        
        # Vi tri index cua cac khop tuong ung trong JOINT_NAMES
        nose = frame[0]
        l_shoulder, r_shoulder = frame[1], frame[2]
        l_hip, r_hip = frame[7], frame[8]
        l_ankle, r_ankle = frame[11], frame[12]
        
        # 1. Normalized Center of Mass Distance
        com = (l_hip + r_hip) / 2
        feet_center = (l_ankle + r_ankle) / 2
        dist_y = abs(com[1] - feet_center[1])
        norm_dist = np.clip(dist_y / 1.5, 0, 1) # Voi toa do da chuan hoa
        
        # 2. Body Tilt Angle
        neck = (l_shoulder + r_shoulder) / 2
        dx = com[0] - neck[0]
        dy = com[1] - neck[1]
        angle = np.degrees(np.arctan2(dx, dy))
        norm_tilt = np.clip(abs(angle) / 45.0, 0, 1)
        
        # 3. Horizontal Velocity
        if frame_idx > 0:
            prev_frame = seq[frame_idx - 1]
            prev_com = (prev_frame[7] + prev_frame[8]) / 2
            vel_x = abs(com[0] - prev_com[0])
            norm_vel = np.clip(vel_x / 0.5, 0, 1)
        else:
            norm_vel = 0.0
            
        # 4. Asymmetry Index
        asym = np.clip(abs(l_shoulder[1] - r_shoulder[1]), 0, 1)
        
        features.append([norm_dist, norm_tilt, norm_vel, asym])
        
    return np.array(features)

def convert_ue5_rgb_to_npz(input_dir, output_npz, output_csv, seq_len=30):
    input_path = Path(input_dir)
    
    # Get all RGB images sorted
    rgb_files = sorted(list(input_path.glob("rgb_*.png")))
    if not rgb_files:
        print(f"Error: No rgb_*.png files found in {input_dir}")
        return
        
    print(f"Found {len(rgb_files)} frames. Running MediaPipe pose estimation...")
    
    tracker = PoseGaitTracker(model_complexity=1)
    
    raw_skeletons = []
    frame_labels = []
    
    for idx, rgb_file in enumerate(rgb_files):
        # Load image
        img = cv2.imread(str(rgb_file))
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        keypoints, found = tracker.track_pose(img_rgb)
        
        if found:
            # 36-d skeleton vector
            vec = extract_skeleton_vector(keypoints)
            raw_skeletons.append(vec)
            
            # Calculate tilt to auto-label
            tilt = calculate_tilt_angle(keypoints)
            if tilt < 15.0:
                frame_labels.append(0) # Safe
            elif tilt < 30.0:
                frame_labels.append(1) # Warning
            else:
                frame_labels.append(2) # Danger
        else:
            # Fallback to zero vector if pose is missing in a single frame
            if raw_skeletons:
                raw_skeletons.append(raw_skeletons[-1])
                frame_labels.append(frame_labels[-1])
            else:
                raw_skeletons.append([0.0] * 36)
                frame_labels.append(0)
                
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(rgb_files)} frames...")

    print(f"Pose extraction complete. Total valid frames: {len(raw_skeletons)}")
    
    skeletons_list = []
    labels_list = []
    
    # Slice into sequences of size seq_len (30) with 50% overlap
    step_size = seq_len // 2
    for i in range(0, len(raw_skeletons) - seq_len + 1, step_size):
        seq = raw_skeletons[i:i+seq_len]
        seq_labels = frame_labels[i:i+seq_len]
        
        # The label of the sequence is the maximum warning level in the sequence
        seq_label = max(seq_labels)
        
        skeletons_list.append(seq)
        labels_list.append(seq_label)
        
    if not skeletons_list:
        print("Error: Not enough frames to make a single sequence.")
        return
        
    skeletons = np.array(skeletons_list) # shape (N, seq_len, 36)
    labels = np.array(labels_list)
    
    # Calculate geometric features
    print("Calculating geometric features...")
    features_list = []
    for seq in skeletons:
        feat = extract_features_from_skeleton(seq)
        features_list.append(feat)
        
    features = np.array(features_list) # shape (N, seq_len, 4)
    
    # Save NPZ
    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, skeletons=skeletons, features=features, labels=labels)
    
    # Save CSV
    df = pd.DataFrame({
        "sample_index": np.arange(len(labels)),
        "label": labels,
        "label_name": [["Safe", "Warning", "Danger"][l] for l in labels]
    })
    df.to_csv(output_csv, index=False)
    
    print(f"Conversion completed successfully: {len(skeletons)} sequences generated.")
    print(f"Skeletons shape: {skeletons.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Labels distribution: {pd.Series(labels).value_counts().to_dict()}")
    print(f"Saved: {output_npz}")
    print(f"Saved: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/ue5_raw", help="Folder containing raw UE5 rgb images")
    parser.add_argument("--out-npz", type=str, default="data/ue5_dataset.npz")
    parser.add_argument("--out-csv", type=str, default="data/ue5_labels.csv")
    
    args = parser.parse_args()
    convert_ue5_rgb_to_npz(args.input, args.out_npz, args.out_csv)
