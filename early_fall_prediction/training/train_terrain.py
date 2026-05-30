# -*- coding: utf-8 -*-

"""
Train Terrain Segmentation – Fine-tune YOLOv8-seg cho nhận diện vật cản.

Cách dùng:
  python training/train_terrain.py --data data/terrain_dataset.yaml --epochs 50
  python training/train_terrain.py --data data/terrain_dataset.yaml --model yolov8n-seg.pt --epochs 100 --imgsz 640

Cấu trúc thư mục dữ liệu (YOLO format):
  data/terrain/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/       (cho detection) hoặc masks/ (cho segmentation)
  │   ├── train/
  │   └── val/
  └── terrain_dataset.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TrainTerrain")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("Cần cài đặt ultralytics: pip install ultralytics")


def create_sample_dataset_yaml(output_path: Path) -> None:
    """Tạo file YAML mẫu cho dataset terrain."""
    content = """# Terrain Hazard Segmentation Dataset
# Chỉnh sửa các đường dẫn phù hợp với dữ liệu của bạn

path: data/terrain          # Thư mục gốc
train: images/train         # Ảnh train (tương đối so với path)
val: images/val             # Ảnh val

# Danh sách class (segmentation)
names:
  0: safe_ground            # Mặt đường an toàn
  1: obstacle               # Vật cản (ghế, hộp, đá, vali, ...)
  2: pothole                # Ổ gà, vùng lõm
  3: step_edge              # Gờ bậc, bậc thang
  4: wet_surface            # Mặt đường trơn/ướt
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Sample dataset YAML created: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8-seg for terrain hazard segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Pretrained model hoặc checkpoint.")
    parser.add_argument("--data", default="data/terrain/terrain_dataset.yaml", help="Dataset YAML.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01, help="Learning rate khởi đầu.")
    parser.add_argument("--project", default="runs/terrain", help="Thư mục lưu kết quả.")
    parser.add_argument("--name", default="train", help="Tên experiment.")
    parser.add_argument("--device", default="", help="cuda device (0, 0,1, cpu).")
    parser.add_argument("--resume", action="store_true", help="Resume training từ checkpoint cuối.")
    parser.add_argument("--create-sample-yaml", action="store_true", help="Tạo file YAML mẫu rồi thoát.")
    args = parser.parse_args()

    if args.create_sample_yaml:
        create_sample_dataset_yaml(Path(args.data))
        return

    if not YOLO_AVAILABLE:
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset YAML không tìm thấy: %s", data_path)
        logger.info("Chạy --create-sample-yaml để tạo file mẫu.")
        sys.exit(1)

    # Load model
    logger.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    # Train
    logger.info("Starting training: %d epochs, imgsz=%d, batch=%d", args.epochs, args.imgsz, args.batch)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
        resume=args.resume,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Segmentation-specific
        overlap_mask=True,
        mask_ratio=4,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Early stopping
        patience=15,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    logger.info("Training complete. Results saved in: %s/%s", args.project, args.name)

    # Validate
    logger.info("Running validation...")
    metrics = model.val()
    logger.info("mAP50: %.4f | mAP50-95: %.4f", metrics.seg.map50, metrics.seg.map)

    # Export best model
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.exists():
        logger.info("Best model: %s", best_path)
        logger.info("Copy vào weights/ để pipeline sử dụng:")
        logger.info("  cp %s weights/yolov8n-seg.pt", best_path)


if __name__ == "__main__":
    main()
