# -*- coding: utf-8 -*-

"""
Temporal SNN (Spiking Neural Network) Module.
Xử lý chuỗi thời gian skeleton + khoảng cách tới vật cản để trích xuất
đặc trưng bất ổn dáng đi qua mô hình nơ-ron xung nhiều lớp.

Kiến trúc:
  Input (36-d)  ──► LIF Layer 1 (64-d)  ──► LIF Layer 2 (32-d)  ──► Output (16-d)
  Mỗi lớp LIF giữ trạng thái điện thế màng (membrane potential) liên tục giữa các frame.
"""

from __future__ import annotations

import collections
import logging
import os
import time

import numpy as np

logger = logging.getLogger("TemporalSNN")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch chưa được cài đặt. TemporalSNN chạy ở chế độ giả lập.")

# Danh sách 18 khớp xương chính (x, y) → 36 chiều đầu vào
_JOINT_NAMES: list[str] = [
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

INPUT_DIM = len(_JOINT_NAMES) * 2  # 36


# ======================================================================
#  Leaky Integrate-and-Fire Cell
# ======================================================================
if TORCH_AVAILABLE:
    class SurrogateSpike(torch.autograd.Function):
        """Surrogate gradient for Heaviside step function using Fast Sigmoid."""
        @staticmethod
        def forward(ctx, x, threshold=1.0):
            ctx.save_for_backward(x - threshold)
            return (x >= threshold).float()

        @staticmethod
        def backward(ctx, grad_output):
            (diff,) = ctx.saved_tensors
            beta = 10.0
            grad_input = grad_output * (0.5 * beta / (1.0 + beta * torch.abs(diff))**2)
            return grad_input, None

    class LIFCell(nn.Module):
        """Tế bào nơ-ron xung Leaky Integrate-and-Fire."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            decay: float = 0.85,
            threshold: float = 1.0,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.decay = decay
            self.threshold = threshold
            self.synapse = nn.Linear(input_dim, hidden_dim)
            nn.init.xavier_uniform_(self.synapse.weight)

        def forward(
            self,
            x: torch.Tensor,
            mem: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x   – [batch, input_dim]
                mem – [batch, hidden_dim]  điện thế màng hiện tại
            Returns:
                spikes   – [batch, hidden_dim]  0/1
                next_mem – [batch, hidden_dim]  điện thế sau reset
            """
            mem = mem * self.decay + self.synapse(x)
            # Sử dụng surrogate gradient để lan truyền ngược qua hàm xung
            spikes = SurrogateSpike.apply(mem, self.threshold)
            next_mem = mem * (1.0 - spikes)  # reset nơ-ron đã phát xung
            return spikes, next_mem

    # ==================================================================
    #  Mạng SNN nhiều lớp
    # ==================================================================
    class SpikingNetwork(nn.Module):
        """SNN 2 lớp LIF + 1 lớp readout tuyến tính."""

        def __init__(
            self,
            input_dim: int = INPUT_DIM,
            hidden1: int = 64,
            hidden2: int = 32,
            output_dim: int = 16,
            decay: float = 0.85,
            threshold: float = 1.0,
        ) -> None:
            super().__init__()
            self.hidden1 = hidden1
            self.hidden2 = hidden2
            self.output_dim = output_dim

            self.lif1 = LIFCell(input_dim, hidden1, decay, threshold)
            self.lif2 = LIFCell(hidden1, hidden2, decay, threshold)
            self.readout = nn.Linear(hidden2, output_dim)

        def forward(
            self,
            x: torch.Tensor,
            mem1: torch.Tensor,
            mem2: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Một bước thời gian (1 frame).
            Returns:
                feature – [batch, output_dim]
                mem1    – cập nhật
                mem2    – cập nhật
            """
            spk1, mem1 = self.lif1(x, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            feature = self.readout(spk2)
            return feature, mem1, mem2

else:
    # Stub khi không có PyTorch
    LIFCell = None          # type: ignore[assignment,misc]
    SpikingNetwork = None   # type: ignore[assignment,misc]


# ======================================================================
#  Wrapper API cho pipeline
# ======================================================================
class TemporalSNN:
    """Bọc SpikingNetwork, quản lý trạng thái màng và lịch sử đặc trưng."""

    def __init__(
        self,
        model_path: str = "",
        output_dim: int = 16,
        decay: float = 0.85,
        threshold: float = 1.0,
    ) -> None:
        self.model_path = model_path
        self.output_dim = output_dim
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        self.model: SpikingNetwork | None = None  # type: ignore[assignment]
        self._mem1: torch.Tensor | None = None
        self._mem2: torch.Tensor | None = None

        # Lưu lịch sử đặc trưng đầu ra để phân tích xu hướng
        self._feature_history: collections.deque[np.ndarray] = collections.deque(maxlen=60)

        # Bộ đệm simulation
        self._sim_t0 = time.time()

        if TORCH_AVAILABLE:
            self._build(decay, threshold)
            self._try_load_weights()

    # ------------------------------------------------------------------
    def _build(self, decay: float, threshold: float) -> None:
        self.model = SpikingNetwork(
            input_dim=INPUT_DIM,
            hidden1=64,
            hidden2=32,
            output_dim=self.output_dim,
            decay=decay,
            threshold=threshold,
        )
        self.model.to(self.device)
        self.model.eval()

        self._mem1 = torch.zeros(1, 64, device=self.device)
        self._mem2 = torch.zeros(1, 32, device=self.device)

    # ------------------------------------------------------------------
    def _try_load_weights(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info("SNN: không tìm thấy weights → dùng trọng số khởi tạo.")
            return
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)
            logger.info("SNN weights loaded from %s", self.model_path)
        except Exception as e:
            logger.error("SNN load weights failed: %s", e)

    # ------------------------------------------------------------------
    # Reset trạng thái màng (khi đổi video / mất người)
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Xóa điện thế màng và lịch sử."""
        if TORCH_AVAILABLE and self.model is not None:
            self._mem1 = torch.zeros(1, 64, device=self.device)
            self._mem2 = torch.zeros(1, 32, device=self.device)
        self._feature_history.clear()

    # ------------------------------------------------------------------
    # API chính: forward
    # ------------------------------------------------------------------
    def forward(
        self,
        keypoints: dict[str, tuple[int, int]],
        hazard_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Xử lý 1 frame.
        Returns:
            feature_vector – np.ndarray shape [output_dim]
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._fallback_simulation()

        try:
            x_vec = self._encode_keypoints(keypoints)
            x_tensor = torch.tensor([x_vec], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                feat, self._mem1, self._mem2 = self.model(x_tensor, self._mem1, self._mem2)

            out = feat.cpu().numpy()[0]
            self._feature_history.append(out.copy())
            return out

        except Exception as e:
            logger.error("SNN forward error: %s", e)
            return np.zeros(self.output_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Trích xuất xu hướng bất ổn từ lịch sử đặc trưng
    # ------------------------------------------------------------------
    def get_instability_score(self) -> float:
        """
        Tính độ bất ổn dáng đi dựa trên phương sai của chuỗi đặc trưng SNN gần nhất.
        Trả về giá trị 0.0 (ổn định) đến 1.0 (rất bất ổn).
        """
        if len(self._feature_history) < 5:
            return 0.0

        recent = np.array(list(self._feature_history)[-20:])
        variance = np.mean(np.var(recent, axis=0))
        # Chuẩn hóa mềm bằng sigmoid
        score = float(1.0 / (1.0 + np.exp(-10.0 * (variance - 0.15))))
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Encode keypoints → vector 36-d
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_keypoints(keypoints: dict[str, tuple[int, int]]) -> list[float]:
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
        vec: list[float] = []
        for name in _JOINT_NAMES:
            if name in keypoints:
                pt = keypoints[name]
                vec.extend([
                    (float(pt[0]) - center_x) / scale,
                    (float(pt[1]) - center_y) / scale
                ])
            else:
                vec.extend([0.0, 0.0])
        return vec

    # ------------------------------------------------------------------
    # Fallback mô phỏng
    # ------------------------------------------------------------------
    def _fallback_simulation(self) -> np.ndarray:
        t = time.time() - self._sim_t0
        vec = np.zeros(self.output_dim, dtype=np.float32)
        for i in range(self.output_dim):
            freq = 1.0 + i * 0.15
            vec[i] = 0.5 * (np.sin(t * freq) + 1.0)
        self._feature_history.append(vec.copy())
        return vec
