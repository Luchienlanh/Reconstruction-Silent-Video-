# -*- coding: utf-8 -*-

"""
Risk Fusion KAN (Kolmogorov-Arnold Network) Module.
Dung hợp đặc trưng từ SNN temporal, hazard mask, depth map, trajectory,
và gait analysis để tính Risk Score (0.0 → 1.0).

Kiến trúc KAN:
  Input (20-d) ──► KANLayer(20→12, grid=6) ──► SiLU
                ──► KANLayer(12→6, grid=4)  ──► SiLU
                ──► Linear(6→1)             ──► Sigmoid
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("RiskFusionKAN")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch chưa cài đặt. RiskFusionKAN chạy ở chế độ rule-based.")


# ======================================================================
#  KAN Layer – học hàm phi tuyến trên mỗi cạnh bằng cơ sở Fourier
# ======================================================================
if TORCH_AVAILABLE:
    class KANLayer(nn.Module):
        """
        Lớp Kolmogorov-Arnold sử dụng hàm cơ sở dạng sin/cos.
        Mỗi cạnh (i→o) có grid_size hệ số learnable cho sin(k·x) và cos(k·x).
        """

        def __init__(self, in_features: int, out_features: int, grid_size: int = 5) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.grid_size = grid_size

            # Hệ số sin: [in, out, grid]
            self.sin_coef = nn.Parameter(torch.randn(in_features, out_features, grid_size) * 0.05)
            # Hệ số cos: [in, out, grid]
            self.cos_coef = nn.Parameter(torch.randn(in_features, out_features, grid_size) * 0.05)
            # Thành phần tuyến tính cơ sở (residual)
            self.linear = nn.Linear(in_features, out_features)
            # Bias bổ sung
            self.bias = nn.Parameter(torch.zeros(out_features))

            nn.init.xavier_uniform_(self.linear.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: [batch, in_features] → [batch, out_features]"""
            batch = x.shape[0]

            # Mở rộng x để nhân với từng bậc k
            x_exp = x.unsqueeze(2)  # [B, in, 1]

            # Tạo bậc k = 1, 2, ..., grid_size
            k = torch.arange(1, self.grid_size + 1, dtype=x.dtype, device=x.device)
            k = k.view(1, 1, -1)  # [1, 1, grid]

            kx = k * x_exp  # [B, in, grid]

            sin_basis = torch.sin(kx)  # [B, in, grid]
            cos_basis = torch.cos(kx)  # [B, in, grid]

            # Tính đóng góp phi tuyến: einsum("big,iog->bo")
            y_sin = torch.einsum("big,iog->bo", sin_basis, self.sin_coef)
            y_cos = torch.einsum("big,iog->bo", cos_basis, self.cos_coef)

            # Cộng thành phần tuyến tính cơ sở
            y_lin = self.linear(x)

            return y_sin + y_cos + y_lin + self.bias

    # ==================================================================
    #  Mạng KAN hoàn chỉnh
    # ==================================================================
    class KANFusionNet(nn.Module):
        """Mạng KAN 2 lớp + readout sigmoid."""

        def __init__(
            self,
            input_dim: int = 20,
            hidden1: int = 12,
            hidden2: int = 6,
            grid1: int = 6,
            grid2: int = 4,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                KANLayer(input_dim, hidden1, grid1),
                nn.SiLU(),
                nn.LayerNorm(hidden1),
                KANLayer(hidden1, hidden2, grid2),
                nn.SiLU(),
                nn.Linear(hidden2, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:
    KANLayer = None       # type: ignore[assignment,misc]
    KANFusionNet = None   # type: ignore[assignment,misc]


# ======================================================================
#  Wrapper API cho pipeline
# ======================================================================
class RiskFusionKAN:
    """Quản lý mạng KAN, trích xuất đặc trưng hình học, và dự đoán risk."""

    # Kích thước vector đầu vào:
    #   16 (SNN temporal) + 1 (norm distance) + 1 (future intersect)
    #   + 1 (body tilt) + 1 (depth diff) = 20
    FEATURE_DIM = 20

    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model: KANFusionNet | None = None  # type: ignore[assignment]

        if TORCH_AVAILABLE:
            self._build()
            self._try_load_weights()

    def _build(self) -> None:
        self.model = KANFusionNet(input_dim=self.FEATURE_DIM)
        self.model.to(self.device)
        self.model.eval()

    def _try_load_weights(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info("KAN: không tìm thấy weights → dùng trọng số khởi tạo.")
            return
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)
            logger.info("KAN weights loaded from %s", self.model_path)
        except Exception as e:
            logger.error("KAN load weights failed: %s", e)

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def predict_risk(
        self,
        temporal_feature: np.ndarray | None,
        hazard_mask: np.ndarray | None,
        depth_map: np.ndarray | None,
        foot_position: tuple[int, int] | None,
        future_foot: tuple[int, int] | None,
        gait_features: dict[str, float] | None,
    ) -> float:
        """Trả về risk score 0.0 (an toàn) → 1.0 (nguy hiểm)."""

        # ---- Trích xuất 4 đặc trưng hình học ----
        geo = self._extract_geometric_features(
            hazard_mask, depth_map, foot_position, future_foot, gait_features
        )

        # ---- Nếu không có PyTorch / model → fallback ----
        if not TORCH_AVAILABLE or self.model is None:
            return self._rule_based_risk(geo)

        try:
            # Ghép vector: SNN(16) + geo(4) = 20
            snn_vec = temporal_feature if temporal_feature is not None else np.zeros(16, dtype=np.float32)
            features = np.concatenate([snn_vec, geo]).astype(np.float32)

            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                score = self.model(x)
            return float(score.cpu().item())

        except Exception as e:
            logger.error("KAN predict error: %s → fallback", e)
            return self._rule_based_risk(geo)

    # ------------------------------------------------------------------
    # Trích xuất đặc trưng hình học (4-d)
    # ------------------------------------------------------------------
    def _extract_geometric_features(
        self,
        hazard_mask: np.ndarray | None,
        depth_map: np.ndarray | None,
        foot_position: tuple[int, int] | None,
        future_foot: tuple[int, int] | None,
        gait_features: dict[str, float] | None,
    ) -> np.ndarray:
        """Trả về vector [norm_dist, future_intersect, norm_tilt, norm_depth]."""

        # 1) Khoảng cách chân → vật cản (chuẩn hóa 0-1, càng gần càng lớn)
        norm_dist = 0.0
        if hazard_mask is not None and foot_position is not None and hazard_mask.any():
            try:
                from utils.geometry_math import distance_foot_to_nearest_hazard
                raw = distance_foot_to_nearest_hazard(hazard_mask, foot_position)
                norm_dist = max(0.0, 1.0 - raw / 300.0)
            except Exception:
                pass

        # 2) Quỹ đạo tương lai cắt vật cản?
        future_intersect = 0.0
        if hazard_mask is not None and future_foot is not None and hazard_mask.any():
            try:
                from utils.geometry_math import foot_inside_hazard
                if foot_inside_hazard(hazard_mask, future_foot):
                    future_intersect = 1.0
            except Exception:
                pass

        # 3) Độ nghiêng thân (chuẩn hóa theo 45°)
        norm_tilt = 0.0
        if gait_features is not None:
            norm_tilt = min(1.0, abs(gait_features.get("body_tilt", 0.0)) / 45.0)

        # 4) Chênh lệch độ sâu tại vị trí chân
        norm_depth = 0.0
        if depth_map is not None and foot_position is not None:
            h, w = depth_map.shape[:2]
            fx = max(0, min(foot_position[0], w - 1))
            fy = max(0, min(foot_position[1], h - 1))
            # Lấy depth tại chân và depth trung bình vùng lân cận
            r = 15
            y1, y2 = max(0, fy - r), min(h, fy + r)
            x1, x2 = max(0, fx - r), min(w, fx + r)
            local_patch = depth_map[y1:y2, x1:x2]
            if local_patch.size > 0:
                local_std = float(np.std(local_patch))
                norm_depth = min(1.0, local_std / 0.15)  # std > 0.15 coi là chênh lệch lớn

        return np.array([norm_dist, future_intersect, norm_tilt, norm_depth], dtype=np.float32)

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _rule_based_risk(geo: np.ndarray) -> float:
        """Tính risk bằng trọng số cố định khi không có mạng KAN."""
        norm_dist, future_intersect, norm_tilt, norm_depth = geo
        risk = 0.0
        risk += 0.40 * norm_dist
        risk += 0.30 * future_intersect
        risk += 0.15 * norm_tilt
        risk += 0.15 * norm_depth
        return min(float(risk), 1.0)
