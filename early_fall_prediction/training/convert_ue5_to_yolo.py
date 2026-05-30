import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil
import yaml

# Cau hinh map mau (B, G, R) tu UnrealCV sang Class ID cua YOLO
# Luu y: OpenCV doc anh theo thu tu BGR, nen mau RGB can doi nguoc lai.
#   obstacle    RGB(255, 0,   0  ) -> BGR(0,   0,   255) -> class 1
#   pothole     RGB(0,   0,   255) -> BGR(255, 0,   0  ) -> class 2
#   step_edge   RGB(0,   255, 0  ) -> BGR(0,   255, 0  ) -> class 3
#   wet_surface RGB(255, 255, 0  ) -> BGR(0,   255, 255) -> class 4
COLOR_TO_CLASS = {
    (0,   0,   255): 1,   # Mau Do   (Red)        -> class 1: obstacle
    (255, 0,   0  ): 2,   # Mau Xanh Duong (Blue) -> class 2: pothole
    (0,   255, 0  ): 3,   # Mau Xanh La (Green)   -> class 3: step_edge
    (0,   255, 255): 4,   # Mau Vang (Yellow)      -> class 4: wet_surface
}

# Nguong dung sai mau sac – xu ly nhieu nhe khi luu anh PNG
COLOR_TOLERANCE = 30

CLASS_NAMES = ['safe_ground', 'obstacle', 'pothole', 'step_edge', 'wet_surface']

def mask_to_yolo_polygons(mask_img):
    """
    Tim polygons (contours) cho tung class dua vao mau sac cua mask_img.
    Tra ve list of strings theo format YOLO: "<class_id> <x1> <y1> <x2> <y2> ..."
    Su dung dung sai mau sac (COLOR_TOLERANCE) de xu ly nhieu nhe khi luu PNG.
    """
    height, width, _ = mask_img.shape
    yolo_annotations = []
    
    for color_bgr, class_id in COLOR_TO_CLASS.items():
        # Tao mask nhi phan voi dung sai mau sac
        lower = np.clip(np.array(color_bgr, dtype=np.int32) - COLOR_TOLERANCE, 0, 255).astype(np.uint8)
        upper = np.clip(np.array(color_bgr, dtype=np.int32) + COLOR_TOLERANCE, 0, 255).astype(np.uint8)
        binary_mask = cv2.inRange(mask_img, lower, upper)
        
        # Tim contours (polygons)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Loc bot contour qua nho (nhieu)
            if cv2.contourArea(contour) < 50:
                continue
                
            # Don gian hoa contour de giam dung luong file txt
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Neu contour qua it diem thi bo qua
            if len(approx) < 3:
                continue
            
            # Chuan hoa toa do ve khoang [0, 1]
            polygon_str = f"{class_id}"
            for point in approx:
                x, y = point[0]
                norm_x = x / width
                norm_y = y / height
                polygon_str += f" {norm_x:.6f} {norm_y:.6f}"
                
            yolo_annotations.append(polygon_str)
            
    return yolo_annotations

def convert_ue5_to_yolo(input_dir, output_dir, split_ratio=0.8):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Tạo cấu trúc thư mục YOLO
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    # Tạo file dataset.yaml
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(output_path / 'terrain_dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print("YOLO dataset structure created successfully.")
    
    # Lay danh sach anh RGB
    rgb_files = sorted(list(input_path.glob("rgb_*.png")))
    if not rgb_files:
        print(f"No image files found in {input_dir}")
        return
        
    num_train = int(len(rgb_files) * split_ratio)
    
    success_count = 0
    class_counts = {cid: 0 for cid in COLOR_TO_CLASS.values()}

    for idx, rgb_file in enumerate(rgb_files):
        frame_id = rgb_file.stem.split('_')[1]
        mask_file = input_path / f"mask_{frame_id}.png"
        
        if not mask_file.exists():
            print(f"Missing mask_file for frame {frame_id}")
            continue
            
        split = 'train' if idx < num_train else 'val'
        
        # Doc mask image va chuyen sang polygons
        mask_img = cv2.imread(str(mask_file))
        if mask_img is None:
            continue
            
        annotations = mask_to_yolo_polygons(mask_img)
        
        # Dem so luong annotation theo tung class
        for ann in annotations:
            cid = int(ann.split()[0])
            if cid in class_counts:
                class_counts[cid] += 1
        
        # Neu co annotations, copy anh va ghi file nhan
        shutil.copy(rgb_file, output_path / 'images' / split / f"{frame_id}.png")
        
        with open(output_path / 'labels' / split / f"{frame_id}.txt", "w") as f:
            f.write("\n".join(annotations))
            
        success_count += 1
        
    print(f"\nConversion completed: {success_count}/{len(rgb_files)} frames -> YOLO Format.")
    print(f"YOLO dataset folder: {output_path}")
    print("\nAnnotation count by class:")
    names = {1: 'obstacle', 2: 'pothole', 3: 'step_edge', 4: 'wet_surface'}
    for cid, cnt in class_counts.items():
        status = "OK" if cnt > 0 else "WARNING: 0 annotations - check spawn colors!"
        print(f"  Class {cid} ({names.get(cid, '?'):12s}): {cnt:4d} instances  [{status}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển Mask UE5 sang YOLO format")
    parser.add_argument("--input", type=str, default="data/ue5_raw", help="Thư mục chứa rgb_*.png và mask_*.png")
    parser.add_argument("--output", type=str, default="data/terrain", help="Thư mục xuất YOLO dataset")
    
    args = parser.parse_args()
    convert_ue5_to_yolo(args.input, args.output)
