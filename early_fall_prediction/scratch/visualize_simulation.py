# -*- coding: utf-8 -*-

"""
Visualize Simulation Data to verify the correctness of the generated skeletons.
Draws the skeleton sequences for Safe, Warning, and Danger classes and saves them as images.
"""

import os
from pathlib import Path
import numpy as np
import cv2

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 18 joints index mapping (to draw skeleton connections)
JOINT_CONNECTIONS = [
    # Body
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Arms
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_foot_index"),
    # Head details
    ("nose", "left_ear")
]

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

def draw_skeleton(img, coords_2d, color=(0, 255, 0), thickness=2):
    """coords_2d: dict of joint_name -> (x, y)"""
    # Draw connections
    for joint_a, joint_b in JOINT_CONNECTIONS:
        if joint_a in coords_2d and joint_b in coords_2d:
            pt_a = coords_2d[joint_a]
            pt_b = coords_2d[joint_b]
            cv2.line(img, pt_a, pt_b, color, thickness)
            
    # Draw joints
    for name, pt in coords_2d.items():
        cv2.circle(img, pt, 4, (0, 0, 255), -1)

def main():
    dataset_path = PROJECT_ROOT / "data" / "dataset.npz"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}!")
        return

    data = np.load(dataset_path)
    skeletons = data["skeletons"]  # [N, seq_len, 36]
    features = data["features"]    # [N, seq_len, 4]
    labels = data["labels"]        # [N]

    print(f"Loaded dataset: skeletons={skeletons.shape}, labels={labels.shape}")

    # Output directory
    output_dir = PROJECT_ROOT / "data" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find one sample for each label
    for label_val, label_name in [(0, "Safe"), (1, "Warning"), (2, "Danger")]:
        indices = np.where(labels == label_val)[0]
        if len(indices) == 0:
            continue
        
        idx = indices[0]
        sample_skel = skeletons[idx]  # [seq_len, 36]
        sample_feat = features[idx]  # [seq_len, 4]
        
        print(f"Visualizing {label_name} sample index {idx}...")
        
        # We will create an image that shows 5 frames of the sequence (e.g. frame 0, 7, 14, 21, 29)
        # side-by-side to visualize the temporal movement
        frame_indices = [0, 7, 14, 21, 29]
        h, w = 400, 300
        canvas = np.zeros((h, w * len(frame_indices), 3), dtype=np.uint8)
        
        # Color based on label
        if label_val == 0:
            color = (0, 255, 0)  # Green
        elif label_val == 1:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
            
        for count, f_idx in enumerate(frame_indices):
            # Extract 2d coords from flat array
            flat_coords = sample_skel[f_idx]
            
            # Since SNN training data is normalized, we need to denormalize it
            # using default hip center (640, 200) to draw it inside the canvas.
            # Let's put hip center at (w/2, h/2) inside each sub-frame
            center_x = w // 2
            center_y = h // 2 - 50
            scale = 120.0  # Scale down to fit sub-frame
            
            coords_2d = {}
            for j_idx, name in enumerate(JOINT_NAMES):
                norm_x = flat_coords[j_idx * 2]
                norm_y = flat_coords[j_idx * 2 + 1]
                # Denormalize relative to center
                x = int(center_x + norm_x * scale)
                y = int(center_y + norm_y * scale)
                coords_2d[name] = (x, y)
                
            # Create sub-frame region
            sub_img = np.zeros((h, w, 3), dtype=np.uint8)
            # Add grid
            cv2.rectangle(sub_img, (0, 0), (w, h), (40, 40, 40), 1)
            # Add text
            cv2.putText(sub_img, f"Frame {f_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Draw KAN features for this frame
            feat = sample_feat[f_idx]
            cv2.putText(sub_img, f"Dist: {feat[0]:.2f}", (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(sub_img, f"Tilt: {feat[1]:.2f}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(sub_img, f"Vel: {feat[2]:.2f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(sub_img, f"Asym: {feat[3]:.2f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Draw skeleton
            draw_skeleton(sub_img, coords_2d, color=color, thickness=2)
            
            # --- PHYSICS CALCULATION FOR VISUALIZATION ---
            # 1. Center of Mass (CoM)
            # Torso midpoint
            torso_x = (coords_2d["left_shoulder"][0] + coords_2d["right_shoulder"][0] + coords_2d["left_hip"][0] + coords_2d["right_hip"][0]) / 4.0
            torso_y = (coords_2d["left_shoulder"][1] + coords_2d["right_shoulder"][1] + coords_2d["left_hip"][1] + coords_2d["right_hip"][1]) / 4.0
            # Head
            head_x = coords_2d["nose"][0]
            head_y = coords_2d["nose"][1]
            # Simple weighted CoM
            com_x = int(0.7 * torso_x + 0.3 * head_x)
            com_y = int(0.7 * torso_y + 0.3 * head_y)
            
            # 2. Base of Support (BoS) from ankles
            ankle_l_x = coords_2d["left_ankle"][0]
            ankle_r_x = coords_2d["right_ankle"][0]
            bos_min = min(ankle_l_x, ankle_r_x) - 10
            bos_max = max(ankle_l_x, ankle_r_x) + 10
            bos_y = h - 90
            
            # Draw BoS Line (Yellow)
            cv2.line(sub_img, (bos_min, bos_y), (bos_max, bos_y), (0, 255, 255), 3)
            # Draw ankle ticks to ground
            cv2.line(sub_img, (ankle_l_x, coords_2d["left_ankle"][1]), (ankle_l_x, bos_y), (120, 120, 120), 1)
            cv2.line(sub_img, (ankle_r_x, coords_2d["right_ankle"][1]), (ankle_r_x, bos_y), (120, 120, 120), 1)
            
            # Draw CoM indicator (Blue dot)
            cv2.circle(sub_img, (com_x, com_y), 6, (255, 0, 0), -1)
            # Vertical projection line from CoM to ground
            is_stable = (bos_min <= com_x <= bos_max)
            proj_color = (0, 255, 0) if is_stable else (0, 0, 255)
            cv2.line(sub_img, (com_x, com_y), (com_x, bos_y), proj_color, 1)
            cv2.circle(sub_img, (com_x, bos_y), 4, proj_color, -1)
            
            # Draw Stability Status Text
            status_text = "STABLE" if is_stable else "FALLING"
            cv2.putText(sub_img, status_text, (w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, proj_color, 1)
            
            # Copy to canvas
            canvas[:, count * w : (count + 1) * w] = sub_img
            
        # Draw class name on canvas
        cv2.putText(canvas, f"Class: {label_name}", (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Save canvas
        output_file = output_dir / f"{label_name.lower()}_simulation.jpg"
        cv2.imwrite(str(output_file), canvas)
        print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    main()
