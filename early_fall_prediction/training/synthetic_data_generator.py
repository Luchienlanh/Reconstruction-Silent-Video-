# -*- coding: utf-8 -*-

"""
Synthetic Data Generator using BlenderProc.

Mô phỏng môi trường 3D để sinh dữ liệu tự động (Sim-to-Real).
Tạo ra các cặp dữ liệu hoàn hảo gồm:
  - Ảnh RGB (RGB rendering)
  - Bản đồ độ sâu (Depth Map) chính xác tuyệt đối
  - Semantic/Instance Segmentation Mask (chỉ ra vùng an toàn, vật cản, lỗ hổng)
  - Metadata (thông tin bounding box, tọa độ 3D)

Yêu cầu cài đặt: pip install blenderproc

Cách chạy:
  blenderproc run training/synthetic_data_generator.py \
      --scene data/3d_assets/room.obj \
      --output data/synthetic_dataset/
"""

import argparse
import os
import json
import logging

try:
    import blenderproc as bproc
except ImportError:
    logging.warning("Chưa cài đặt blenderproc. Cài đặt bằng: pip install blenderproc")
    bproc = None

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyntheticGenerator")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='data/3d_assets/room.obj', help="Đường dẫn tới file scene 3D (.obj, .blend)")
    parser.add_argument('--output', type=str, default='data/synthetic_dataset', help="Thư mục lưu dữ liệu đầu ra")
    parser.add_argument('--num_poses', type=int, default=10, help="Số lượng góc camera ngẫu nhiên cần render")
    args = parser.parse_args()

    if bproc is None:
        logger.error("Không thể chạy script này vì thiếu blenderproc.")
        return

    bproc.init()

    # 1. Load scene (ví dụ một con đường hoặc căn phòng có sẵn chướng ngại vật)
    logger.info(f"Loading scene từ {args.scene}...")
    if os.path.exists(args.scene):
        objs = bproc.loader.load_obj(args.scene)
    else:
        logger.warning(f"File scene {args.scene} không tồn tại. Tạo một scene mặc định (sàn + vật cản).")
        # Tạo sàn nhà (safe ground)
        floor = bproc.object.create_primitive("PLANE", scale=[5, 5, 1])
        floor.set_cp("category_id", 0) # 0 = safe ground
        
        # Tạo chướng ngại vật (obstacle)
        obstacle1 = bproc.object.create_primitive("CUBE", scale=[0.5, 0.5, 0.5], location=[1, 1, 0.5])
        obstacle1.set_cp("category_id", 1) # 1 = obstacle
        
        # Tạo một gờ bậc (step edge)
        step = bproc.object.create_primitive("CUBE", scale=[2, 0.2, 0.1], location=[0, -1, 0.1])
        step.set_cp("category_id", 3) # 3 = step edge

    # 2. Cài đặt nguồn sáng ngẫu nhiên để tăng tính đa dạng (domain randomization)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([0, 0, 4])
    light.set_energy(1000)

    # 3. Lấy mẫu ngẫu nhiên các góc camera
    # Mô phỏng góc nhìn của một camera đeo trên người (wearable) hoặc cầm tay, hướng xuống đường
    logger.info(f"Đang lấy mẫu {args.num_poses} góc camera...")
    for i in range(args.num_poses):
        # Tọa độ x, y ngẫu nhiên trên sàn, cao độ z khoảng 1.2m - 1.6m (ngực/đầu)
        location = np.random.uniform([-2, -2, 1.2], [2, 2, 1.6])
        
        # Tính toán điểm nhìn (look-at point) - nhìn xuống phía trước mặt khoảng 2-4m
        poi = location + np.array([np.random.uniform(2, 4), np.random.uniform(-1, 1), -1.2])
        
        # Tính toán rotation matrix từ location tới poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.1, 0.1))
        
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

    # 4. Kích hoạt render RGB, Depth và Segmentation (bằng category_id)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

    # 5. Thực hiện Render
    logger.info("Bắt đầu render dữ liệu (RGB, Depth, Segmentation)...")
    data = bproc.renderer.render()

    # 6. Lưu dữ liệu ra định dạng chuẩn (ví dụ HDF5 hoặc từng file rời)
    os.makedirs(args.output, exist_ok=True)
    bproc.writer.write_bop(args.output, target_objects=bproc.object.get_all_mesh_objects(), dataset="synthetic_terrain", depth_scale=1.0, depth_type=np.float32, color_file_format="JPEG", ignore_dist_z=True)
    bproc.writer.write_hdf5(args.output, data)
    
    logger.info(f"Đã lưu synthetic dataset tại: {args.output}")

if __name__ == "__main__":
    main()
