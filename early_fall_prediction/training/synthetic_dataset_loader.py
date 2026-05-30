# -*- coding: utf-8 -*-

"""
Synthetic Dataset Loader.

Đọc dữ liệu do BlenderProc sinh ra (định dạng HDF5).
Trích xuất ảnh RGB, Depth Map thực, và Segmentation Mask (Hazard Mask) 
để chuẩn bị cho việc train/fine-tune các mô hình (đặc biệt là Terrain Perception và Depth Estimation).
"""

import os
import glob
import logging
import h5py
import numpy as np
import cv2

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyntheticLoader")


class SyntheticTerrainDataset(Dataset if TORCH_AVAILABLE else object): # type: ignore
    """
    PyTorch Dataset để load ảnh RGB, Depth map và Segmentation Mask từ các file HDF5.
    """
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Thư mục chứa các file .hdf5 do BlenderProc xuất ra.
            transform: Các phép biến đổi (augmentation/tensorize) từ torchvision.transforms.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Tìm tất cả các file hdf5 trong thư mục
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        
        if not self.file_paths:
            logger.warning(f"Không tìm thấy file .hdf5 nào trong {data_dir}.")
        else:
            logger.info(f"Đã tìm thấy {len(self.file_paths)} mẫu dữ liệu synthetic.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with h5py.File(file_path, "r") as data:
            # Lấy ảnh RGB (BlenderProc mặc định lưu là uint8 [H, W, 3])
            colors = np.array(data["colors"])
            
            # Lấy bản đồ độ sâu (khoảng cách thực tế bằng mét, float32)
            distance = np.array(data["distance"])
            
            # Lấy segmentation mask (dựa trên category_id đã set trong Blender)
            # category_id = 1, 3 (obstacle, step edge) -> hazard
            category_id = np.array(data["category_id"])
            
            # Tạo binary hazard mask (1 nếu là vật cản nguy hiểm, 0 là an toàn)
            # Giả sử trong file generator, ta set: 1=obstacle, 3=step_edge
            hazard_mask = np.isin(category_id, [1, 3]).astype(np.uint8) * 255

        # Resize hoặc tiền xử lý cơ bản nếu cần
        # ...
            
        sample = {
            "image": colors,
            "depth": distance,
            "mask": hazard_mask
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# Helper function để test loader
def visualize_sample(sample, save_path=None):
    """Hiển thị hoặc lưu một mẫu dữ liệu (RGB, Depth, Mask) để kiểm tra."""
    img = sample["image"]
    depth = sample["depth"]
    mask = sample["mask"]
    
    # Chuẩn hóa depth map để hiển thị
    d_min, d_max = np.min(depth), np.max(depth)
    if d_max > d_min:
        depth_vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # Nối 3 ảnh lại theo chiều ngang
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Mask màu đỏ để dễ nhìn
    mask_colored[mask > 0] = [0, 0, 255]

    combined = np.hstack((cv2.cvtColor(img, cv2.COLOR_RGB2BGR), depth_vis, mask_colored))
    
    if save_path:
        cv2.imwrite(save_path, combined)
        logger.info(f"Đã lưu hình ảnh trực quan tại {save_path}")
    else:
        cv2.imshow("Synthetic Sample (RGB | Depth | Hazard Mask)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test script đơn giản
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/synthetic_dataset')
    parser.add_argument('--vis', action='store_true', help="Hiển thị ảnh mẫu")
    args = parser.parse_args()
    
    dataset = SyntheticTerrainDataset(args.data)
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"Kích thước RGB: {sample['image'].shape}, Depth: {sample['depth'].shape}, Mask: {sample['mask'].shape}")
        if args.vis:
            visualize_sample(sample, save_path="sample_synthetic_vis.jpg")
