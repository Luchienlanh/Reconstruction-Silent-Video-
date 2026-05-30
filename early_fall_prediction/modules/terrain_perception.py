# -*- coding: utf-8 -*-

"""
Terrain Perception Module.
Nhận diện các vật thể nguy hiểm trên đường đi (như ghế, bàn, chai lọ, gờ đá)
và trả về một mặt nạ nhị phân (Hazard Mask) đánh dấu các vị trí không an toàn.
Sử dụng YOLOv8-seg (Ultralytics) để phân đoạn đối tượng (instance segmentation).
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TerrainPerception")

# Thử import ultralytics YOLO
try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(
        "Thư viện 'ultralytics' chưa được cài đặt. "
        "TerrainPerception sẽ chạy ở chế độ giả lập (simulation/fallback)."
    )


class TerrainPerception:
    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        hazard_classes: list[int] | None = None,
        confidence: float = 0.45,
    ) -> None:
        """
        Khởi tạo mô hình phân đoạn vật cản YOLOv8-seg.

        Args:
            model_path: Đường dẫn tới file trọng số (.pt). Nếu không có sẵn, YOLO sẽ tự động tải xuống.
            hazard_classes: Danh sách class ID (COCO) coi là nguy hiểm (mặc định: chair, couch, table, bottle...).
            confidence: Ngưỡng tin cậy (confidence threshold) để lọc vật thể.
        """
        self.model_path = model_path
        self.confidence = confidence
        
        # Mặc định danh sách class COCO nguy hiểm: 
        # 56: chair, 57: couch, 60: diningtable, 39: bottle, 24: backpack, 28: suitcase
        self.hazard_classes = hazard_classes if hazard_classes is not None else [56, 57, 60, 39, 24, 28]
        
        self.model = None
        if ULTRALYTICS_AVAILABLE:
            try:
                logger.info("Đang tải mô hình YOLOv8-seg từ: %s", model_path)
                self.model = YOLO(model_path)
                # Di chuyển model sang GPU nếu khả dụng
                if torch.cuda.is_available():
                    self.model.to("cuda")
                    logger.info("YOLOv8-seg đang chạy trên GPU (CUDA).")
                else:
                    logger.info("YOLOv8-seg đang chạy trên CPU.")
            except Exception as e:
                logger.error("Lỗi khi load mô hình YOLOv8-seg: %s. Chuyển sang chế độ giả lập.", e)
                self.model = None

    def detect_hazards(self, frame: np.ndarray) -> np.ndarray:
        """
        Phát hiện vật cản nguy hiểm trên frame RGB.

        Args:
            frame: Ảnh RGB đầu vào dạng numpy array [H, W, 3].

        Returns:
            hazard_mask: Mặt nạ nhị phân [H, W] dạng np.uint8, trong đó pixel > 0 đại diện cho vật cản.
        """
        h, w = frame.shape[:2]
        
        # Nếu mô hình không khả dụng, chạy fallback giả lập (tạo một vùng hazard giả định ở giữa màn hình)
        if not ULTRALYTICS_AVAILABLE or self.model is None:
            return self._fallback_simulation(h, w)

        try:
            # YOLOv8 yêu cầu frame đầu vào (có thể là BGR hoặc RGB tùy cấu hình, mặc định OpenCV/PIL đều hoạt động tốt)
            # Chạy inference với tham số lọc confidence
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            if not results:
                return np.zeros((h, w), dtype=np.uint8)
                
            result = results[0]
            
            # Kiểm tra xem có detect được mask nào không
            if result.masks is None:
                return np.zeros((h, w), dtype=np.uint8)

            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Lấy thông tin về mask data và class IDs tương ứng
            masks_data = result.masks.data  # Tensor shape [N, mask_h, mask_w]
            class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None else []

            for i, cls_id in enumerate(class_ids):
                if int(cls_id) in self.hazard_classes:
                    mask = masks_data[i]
                    
                    # Chuyển tensor sang numpy array nếu cần
                    if hasattr(mask, "cpu"):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = np.array(mask)
                        
                    # Resize mask về kích thước ảnh gốc nếu kích thước không khớp
                    if mask_np.shape != (h, w):
                        mask_resized = cv2.resize(
                            mask_np, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        mask_resized = mask_np
                        
                    # Chuyển đổi thành mặt nạ nhị phân (0 hoặc 255)
                    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Hợp nhất (OR) các mặt nạ của các đối tượng lại với nhau
                    combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
            
            return combined_mask

        except Exception as e:
            logger.error("Lỗi trong quá trình detect_hazards: %s. Chuyển sang fallback.", e)
            return self._fallback_simulation(h, w)

    def _fallback_simulation(self, h: int, w: int) -> np.ndarray:
        """Sinh ra một vùng nguy hiểm giả định ở giữa/dưới màn hình để kiểm thử hệ thống."""
        mask = np.zeros((h, w), dtype=np.uint8)
        # Giả định một vật cản nằm ở góc dưới bên trái của lối đi (từ 70% đến 90% chiều cao và 30% đến 50% chiều rộng)
        cv2.rectangle(
            mask,
            (int(w * 0.3), int(h * 0.7)),
            (int(w * 0.5), int(h * 0.9)),
            255,
            -1
        )
        return mask
