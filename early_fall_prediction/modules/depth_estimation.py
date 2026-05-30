# -*- coding: utf-8 -*-

"""
Depth Estimation Module.
Ước lượng bản đồ độ sâu tương đối từ ảnh RGB đơn (monocular depth estimation).
Sử dụng Depth Anything V2 hoặc MiDaS làm backbone.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger("DepthEstimation")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch chưa được cài đặt. DepthEstimation chạy ở chế độ giả lập.")

# --- Thử load Depth Anything V2 ---
DEPTH_ANYTHING_AVAILABLE = False
DepthAnythingV2Model = None

if TORCH_AVAILABLE:
    try:
        from transformers import pipeline as hf_pipeline
        DEPTH_ANYTHING_AVAILABLE = True
        logger.info("HuggingFace Transformers sẵn sàng cho Depth Anything V2.")
    except ImportError:
        pass

# --- Thử load MiDaS (fallback nếu không có Depth Anything) ---
MIDAS_AVAILABLE = False
if TORCH_AVAILABLE and not DEPTH_ANYTHING_AVAILABLE:
    try:
        _midas_test = torch.hub.list("intel-isl/MiDaS", force_reload=False)
        MIDAS_AVAILABLE = True
        logger.info("MiDaS (torch.hub) sẵn sàng làm depth backbone dự phòng.")
    except Exception:
        pass


class DepthEstimation:
    """Ước lượng độ sâu tương đối từ ảnh RGB đơn."""

    def __init__(
        self,
        model_path: str = "",
        encoder: str = "vits",
        input_size: int = 518,
    ) -> None:
        self.model_path = model_path
        self.encoder = encoder
        self.input_size = input_size
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        self._pipe = None        # HuggingFace pipeline (Depth Anything V2)
        self._midas = None       # MiDaS model
        self._midas_transform = None
        self._backend = "none"

        if DEPTH_ANYTHING_AVAILABLE:
            self._init_depth_anything()
        elif MIDAS_AVAILABLE:
            self._init_midas()
        else:
            logger.warning(
                "Không tìm thấy backbone depth nào (Depth Anything V2, MiDaS). "
                "Sử dụng chế độ giả lập gradient."
            )

    # ------------------------------------------------------------------
    # Khởi tạo Depth Anything V2 qua HuggingFace Transformers
    # ------------------------------------------------------------------
    def _init_depth_anything(self) -> None:
        try:
            model_id = self.model_path if self.model_path and os.path.exists(self.model_path) \
                else f"depth-anything/Depth-Anything-V2-Small-hf"
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=model_id,
                device=0 if self.device == "cuda" else -1,
            )
            self._backend = "depth_anything_v2"
            logger.info("Depth Anything V2 loaded (%s).", model_id)
        except Exception as e:
            logger.error("Lỗi khi load Depth Anything V2: %s", e)
            self._pipe = None
            if MIDAS_AVAILABLE:
                self._init_midas()

    # ------------------------------------------------------------------
    # Khởi tạo MiDaS qua torch.hub
    # ------------------------------------------------------------------
    def _init_midas(self) -> None:
        try:
            model_type = "MiDaS_small"
            self._midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self._midas.to(self.device).eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._midas_transform = midas_transforms.small_transform
            self._backend = "midas"
            logger.info("MiDaS (%s) loaded successfully.", model_type)
        except Exception as e:
            logger.error("Lỗi khi load MiDaS: %s", e)
            self._midas = None

    # ------------------------------------------------------------------
    # API chính: estimate_depth
    # ------------------------------------------------------------------
    def estimate_depth(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Nhận ảnh RGB [H, W, 3] uint8, trả về depth map [H, W] float32
        đã chuẩn hóa về khoảng [0.0, 1.0] (0 = gần, 1 = xa).
        """
        h, w = frame_rgb.shape[:2]

        if self._backend == "depth_anything_v2" and self._pipe is not None:
            return self._run_depth_anything(frame_rgb, h, w)

        if self._backend == "midas" and self._midas is not None:
            return self._run_midas(frame_rgb, h, w)

        # Fallback: sinh depth map giả lập dạng gradient dọc
        return self._fallback_gradient(h, w)

    # ------------------------------------------------------------------
    # Inference Depth Anything V2
    # ------------------------------------------------------------------
    def _run_depth_anything(self, frame_rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        try:
            from PIL import Image
            pil_img = Image.fromarray(frame_rgb)
            result = self._pipe(pil_img)
            depth_pil = result["depth"]
            depth_np = np.array(depth_pil, dtype=np.float32)

            # Resize về kích thước gốc nếu cần
            if depth_np.shape[:2] != (h, w):
                depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

            # Chuẩn hóa 0-1
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-5:
                depth_np = (depth_np - d_min) / (d_max - d_min)
            else:
                depth_np = np.zeros((h, w), dtype=np.float32)

            return depth_np

        except Exception as e:
            logger.error("Depth Anything V2 inference error: %s", e)
            return self._fallback_gradient(h, w)

    # ------------------------------------------------------------------
    # Inference MiDaS
    # ------------------------------------------------------------------
    def _run_midas(self, frame_rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        try:
            input_batch = self._midas_transform(frame_rgb).to(self.device)
            with torch.no_grad():
                prediction = self._midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_np = prediction.cpu().numpy().astype(np.float32)

            # MiDaS trả về inverse depth (giá trị lớn = gần), đảo ngược lại
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-5:
                depth_np = 1.0 - (depth_np - d_min) / (d_max - d_min)
            else:
                depth_np = np.zeros((h, w), dtype=np.float32)

            return depth_np

        except Exception as e:
            logger.error("MiDaS inference error: %s", e)
            return self._fallback_gradient(h, w)

    # ------------------------------------------------------------------
    # Fallback gradient giả lập
    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_gradient(h: int, w: int) -> np.ndarray:
        """Sinh depth map gradient dọc: trên = xa (1.0), dưới = gần (0.0)."""
        gradient = np.linspace(1.0, 0.0, num=h, dtype=np.float32)
        depth_map = np.tile(gradient[:, np.newaxis], (1, w))
        return depth_map
