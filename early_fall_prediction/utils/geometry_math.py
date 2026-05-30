# -*- coding: utf-8 -*-

"""
Geometry Math Utilities.
Tính toán không gian giữa người (bàn chân, quỹ đạo) và vật cản (hazard mask):
  - Kiểm tra chân có nằm trong vùng nguy hiểm không.
  - Khoảng cách pixel ngắn nhất từ chân tới vật cản.
  - Kiểm tra đoạn quỹ đạo có cắt qua vùng nguy hiểm không.
  - Tính diện tích vùng nguy hiểm nằm trên đường đi.
  - Ước lượng thời gian va chạm (time-to-collision) đơn giản.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ======================================================================
#  1. Kiểm tra chân có nằm trong vùng nguy hiểm
# ======================================================================
def foot_inside_hazard(
    hazard_mask: np.ndarray | None,
    foot_pos: tuple[int, int],
    check_radius: int = 8,
) -> bool:
    """
    Kiểm tra bán kính xung quanh bàn chân có chạm vùng hazard không.

    Args:
        hazard_mask: Mask nhị phân [H, W], pixel > 0 = vật cản.
        foot_pos: Tọa độ pixel (x, y).
        check_radius: Bán kính kiểm tra (pixel).

    Returns:
        True nếu chân chạm/gần chạm vùng nguy hiểm.
    """
    if hazard_mask is None or not hazard_mask.any():
        return False

    h, w = hazard_mask.shape[:2]
    fx, fy = foot_pos

    if not (0 <= fx < w and 0 <= fy < h):
        return False

    y1 = max(0, fy - check_radius)
    y2 = min(h, fy + check_radius + 1)
    x1 = max(0, fx - check_radius)
    x2 = min(w, fx + check_radius + 1)

    return bool(hazard_mask[y1:y2, x1:x2].any())


# ======================================================================
#  2. Khoảng cách ngắn nhất từ chân đến vật cản
# ======================================================================
def distance_foot_to_nearest_hazard(
    hazard_mask: np.ndarray | None,
    foot_pos: tuple[int, int],
    max_search_radius: int = 400,
) -> float:
    """
    Tính khoảng cách Euclid (pixel) ngắn nhất từ chân tới vật cản.

    Sử dụng distance transform thay vì brute-force argwhere để tăng tốc
    trên mask lớn.

    Returns:
        Khoảng cách pixel. float('inf') nếu không có vật cản.
    """
    if hazard_mask is None or not hazard_mask.any():
        return float("inf")

    h, w = hazard_mask.shape[:2]
    fx = max(0, min(foot_pos[0], w - 1))
    fy = max(0, min(foot_pos[1], h - 1))

    # Cắt vùng tìm kiếm cục bộ để giảm tải
    y1 = max(0, fy - max_search_radius)
    y2 = min(h, fy + max_search_radius)
    x1 = max(0, fx - max_search_radius)
    x2 = min(w, fx + max_search_radius)

    local_mask = hazard_mask[y1:y2, x1:x2]
    if not local_mask.any():
        return float("inf")

    # Distance transform: tính khoảng cách mỗi pixel nền (==0) tới pixel vật cản gần nhất
    # Ta đảo mask: nền vật cản → 0, vùng trống → 1, rồi distanceTransform
    inverted = (local_mask == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

    # Tọa độ chân trong hệ cục bộ
    local_fx = fx - x1
    local_fy = fy - y1
    local_fx = max(0, min(local_fx, dist_map.shape[1] - 1))
    local_fy = max(0, min(local_fy, dist_map.shape[0] - 1))

    return float(dist_map[local_fy, local_fx])


# ======================================================================
#  3. Kiểm tra đoạn quỹ đạo cắt qua vùng nguy hiểm
# ======================================================================
def trajectory_intersects_hazard(
    hazard_mask: np.ndarray | None,
    start: tuple[int, int],
    end: tuple[int, int],
    num_samples: int = 20,
) -> bool:
    """
    Lấy mẫu đều trên đoạn thẳng (start → end), kiểm tra có mẫu nào
    rơi vào vùng hazard không.

    Args:
        hazard_mask: Mask nhị phân [H, W].
        start: Tọa độ bắt đầu (x, y) – vị trí chân hiện tại.
        end: Tọa độ kết thúc (x, y) – vị trí chân dự đoán.
        num_samples: Số điểm lấy mẫu trên đoạn.

    Returns:
        True nếu quỹ đạo cắt qua vùng nguy hiểm.
    """
    if hazard_mask is None or not hazard_mask.any():
        return False

    h, w = hazard_mask.shape[:2]

    for i in range(num_samples + 1):
        t = i / max(num_samples, 1)
        sx = int(round(start[0] + t * (end[0] - start[0])))
        sy = int(round(start[1] + t * (end[1] - start[1])))

        sx = max(0, min(sx, w - 1))
        sy = max(0, min(sy, h - 1))

        if hazard_mask[sy, sx] > 0:
            return True

    return False


# ======================================================================
#  4. Diện tích vùng hazard trên đường đi
# ======================================================================
def hazard_area_in_path(
    hazard_mask: np.ndarray | None,
    start: tuple[int, int],
    end: tuple[int, int],
    corridor_half_width: int = 30,
) -> float:
    """
    Tính tỷ lệ diện tích hazard nằm trong hành lang di chuyển.

    Args:
        hazard_mask: Mask nhị phân [H, W].
        start: Tọa độ bắt đầu (x, y).
        end: Tọa độ kết thúc (x, y).
        corridor_half_width: Nửa chiều rộng hành lang (pixel).

    Returns:
        Tỷ lệ pixel hazard trong hành lang (0.0 → 1.0).
    """
    if hazard_mask is None or not hazard_mask.any():
        return 0.0

    h, w = hazard_mask.shape[:2]

    # Tạo mask hành lang bằng cách vẽ đường dày
    corridor = np.zeros((h, w), dtype=np.uint8)
    cv2.line(corridor, start, end, 255, thickness=corridor_half_width * 2)

    # Giao giữa hành lang và hazard
    intersection = cv2.bitwise_and(hazard_mask, corridor)
    corridor_area = float(np.count_nonzero(corridor))

    if corridor_area < 1.0:
        return 0.0

    return float(np.count_nonzero(intersection)) / corridor_area


# ======================================================================
#  5. Ước lượng thời gian va chạm (Time-to-Collision)
# ======================================================================
def estimate_time_to_collision(
    foot_pos: tuple[int, int],
    velocity_px_per_frame: float,
    heading_rad: float | None,
    hazard_mask: np.ndarray | None,
    fps: float = 30.0,
    max_lookahead_frames: int = 90,
    step_px: int = 5,
) -> float:
    """
    Ước lượng số giây cho đến khi chân chạm vật cản nếu đi thẳng
    theo hướng heading hiện tại.

    Returns:
        Số giây. float('inf') nếu không va chạm trong phạm vi lookahead.
    """
    if (
        hazard_mask is None
        or not hazard_mask.any()
        or heading_rad is None
        or velocity_px_per_frame < 0.5
    ):
        return float("inf")

    h, w = hazard_mask.shape[:2]
    dx = math.cos(heading_rad)
    dy = math.sin(heading_rad)

    total_distance = 0.0
    cx, cy = float(foot_pos[0]), float(foot_pos[1])

    max_distance = velocity_px_per_frame * max_lookahead_frames

    while total_distance < max_distance:
        cx += dx * step_px
        cy += dy * step_px
        total_distance += step_px

        ix, iy = int(round(cx)), int(round(cy))

        # Ra ngoài khung hình
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            break

        if hazard_mask[iy, ix] > 0:
            # Quy đổi pixel distance → thời gian
            frames_to_hit = total_distance / max(velocity_px_per_frame, 0.1)
            return frames_to_hit / fps

    return float("inf")


# ======================================================================
#  6. Tính toán vật lý: Center of Mass (CoM) & Base of Support (BoS)
# ======================================================================
def calculate_center_of_mass(keypoints: dict[str, tuple[int, int]]) -> tuple[float, float]:
    """
    Tính tọa độ 2D Center of Mass (CoM) của cơ thể từ các khớp xương.
    Sử dụng tỷ lệ khối lượng cơ thể chuẩn (Anthropometric data):
      - Torso (Shoulders/Hips): 50%
      - Head (Nose/Ears): 8%
      - Mỗi chân (Hip/Knee/Ankle/Heel/Foot): 16% (x2 = 32%)
      - Mỗi tay (Shoulder/Elbow/Wrist): 5% (x2 = 10%)
    """
    segments = []
    weights = []
    
    # 1. Head
    head_pts = [keypoints[k] for k in ["nose", "left_ear"] if k in keypoints]
    if head_pts:
        segments.append(np.mean(head_pts, axis=0))
        weights.append(0.08)
        
    # 2. Torso
    torso_pts = [keypoints[k] for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"] if k in keypoints]
    if torso_pts:
        segments.append(np.mean(torso_pts, axis=0))
        weights.append(0.50)
        
    # 3. Left Arm
    l_arm_pts = [keypoints[k] for k in ["left_shoulder", "left_elbow", "left_wrist"] if k in keypoints]
    if l_arm_pts:
        segments.append(np.mean(l_arm_pts, axis=0))
        weights.append(0.05)
        
    # 4. Right Arm
    r_arm_pts = [keypoints[k] for k in ["right_shoulder", "right_elbow", "right_wrist"] if k in keypoints]
    if r_arm_pts:
        segments.append(np.mean(r_arm_pts, axis=0))
        weights.append(0.05)
        
    # 5. Left Leg
    l_leg_pts = [keypoints[k] for k in ["left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index"] if k in keypoints]
    if l_leg_pts:
        segments.append(np.mean(l_leg_pts, axis=0))
        weights.append(0.16)
        
    # 6. Right Leg
    r_leg_pts = [keypoints[k] for k in ["right_hip", "right_knee", "right_ankle", "right_heel", "right_foot_index"] if k in keypoints]
    if r_leg_pts:
        segments.append(np.mean(r_leg_pts, axis=0))
        weights.append(0.16)
        
    if not segments:
        hip = keypoints.get("left_hip") or keypoints.get("right_hip")
        if hip:
            return float(hip[0]), float(hip[1])
        return 640.0, 360.0
        
    weights = np.array(weights)
    weights /= np.sum(weights)
    com = np.sum([s * w for s, w in zip(segments, weights)], axis=0)
    return float(com[0]), float(com[1])


def calculate_base_of_support(keypoints: dict[str, tuple[int, int]]) -> tuple[float, float]:
    """
    Tính khoảng Base of Support (BoS) trên trục X dựa trên vị trí tiếp xúc của 2 chân.
    Trả về: (min_x, max_x) của chân đế nâng đỡ cơ thể.
    """
    foot_x = []
    for k in ["left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]:
        if k in keypoints:
            foot_x.append(keypoints[k][0])
            
    if not foot_x:
        lh = keypoints.get("left_hip")
        rh = keypoints.get("right_hip")
        if lh and rh:
            return float(min(lh[0], rh[0]) - 30), float(max(lh[0], rh[0]) + 30)
        return 600.0, 680.0
        
    min_x = float(min(foot_x)) - 15.0  # 15px padding
    max_x = float(max(foot_x)) + 15.0
    return min_x, max_x


def evaluate_stability(
    keypoints: dict[str, tuple[int, int]],
    prev_keypoints: dict[str, tuple[int, int]] | None = None,
    fps: float = 30.0
) -> dict:
    """
    Đánh giá độ ổn định cơ thể dựa trên Center of Mass (CoM) và Base of Support (BoS).
    Tính cả Extrapolated Center of Mass (XCoM) cho động học.
    """
    com_x, com_y = calculate_center_of_mass(keypoints)
    bos_min, bos_max = calculate_base_of_support(keypoints)
    
    # Static Stability Margin (SM)
    sm = min(com_x - bos_min, bos_max - com_x)
    is_statically_stable = sm > 0
    
    # Dynamic Stability using Extrapolated CoM (XCoM)
    xcom_x = com_x
    if prev_keypoints is not None:
        prev_com_x, _ = calculate_center_of_mass(prev_keypoints)
        xcom_x = com_x + 4.0 * (com_x - prev_com_x)
        
    dsm = min(xcom_x - bos_min, bos_max - xcom_x)
    is_dynamically_stable = dsm > 0
    
    # Chuẩn hóa stability score từ 0.0 (rất vững) đến 1.0 (mất thăng bằng hoàn toàn)
    if dsm >= 30.0:
        stability_score = 0.0
    elif dsm <= -30.0:
        stability_score = 1.0
    else:
        stability_score = 1.0 - (dsm + 30.0) / 60.0
        
    return {
        "com": (com_x, com_y),
        "bos": (bos_min, bos_max),
        "sm": sm,
        "dsm": dsm,
        "is_stable": is_statically_stable,
        "is_dynamically_stable": is_dynamically_stable,
        "stability_score": float(stability_score)
    }

