from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2):
        if channels % g == 0 and channels >= g:
            return g
    return 1


class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, spatial_stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels,
                mid_channels,
                kernel_size=(1, 3, 3),
                stride=(1, spatial_stride, spatial_stride),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(_groups(mid_channels), mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                mid_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class R2Plus1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_stride: int = 1):
        super().__init__()
        mid_channels = max(1, (in_channels * out_channels * 27) // (in_channels * 9 + out_channels * 3))
        self.conv1 = nn.Sequential(
            Conv2Plus1D(in_channels, out_channels, mid_channels, spatial_stride=spatial_stride),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            Conv2Plus1D(out_channels, out_channels, mid_channels, spatial_stride=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.downsample = None
        if spatial_stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, spatial_stride, spatial_stride),
                    bias=False,
                ),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class R2Plus1DVisualEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 512,
        width: int = 32,
        layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        spatial_pool_size: int = 2,
        temporal_layers: int = 1,
        dropout: float = 0.05,
    ):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(inplace=True),
            nn.Conv3d(width, width, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, c1, layers[0], spatial_stride=1)
        self.layer2 = self._make_layer(c1, c2, layers[1], spatial_stride=2)
        self.layer3 = self._make_layer(c2, c3, layers[2], spatial_stride=2)
        self.layer4 = self._make_layer(c3, c4, layers[3], spatial_stride=2)
        self.pool = nn.AdaptiveAvgPool3d((None, spatial_pool_size, spatial_pool_size))
        self.proj = nn.Sequential(
            nn.Linear(c4 * spatial_pool_size * spatial_pool_size, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.pos_scale = nn.Parameter(torch.tensor(0.02))
        if temporal_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=max(1, min(8, dim // 64)),
                dim_feedforward=dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        else:
            self.temporal = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self._init_weights()

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, num_blocks: int, spatial_stride: int) -> nn.Sequential:
        blocks = [R2Plus1DBlock(in_channels, out_channels, spatial_stride=spatial_stride)]
        for _ in range(1, int(num_blocks)):
            blocks.append(R2Plus1DBlock(out_channels, out_channels, spatial_stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, video: torch.Tensor, video_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(video)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1, 3, 4).flatten(2)
        x = self.proj(x)
        pos = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1)
        x = x + self.pos_scale.to(x.dtype) * torch.sin(2.0 * torch.pi * pos)
        key_padding_mask = None
        if video_mask is not None:
            key_padding_mask = ~video_mask.to(x.device, dtype=torch.bool)
        if isinstance(self.temporal, nn.TransformerEncoder):
            x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.temporal(x)
        return self.norm(x)


class LandmarkMotionEncoder(nn.Module):
    def __init__(
        self,
        num_points: int = 40,
        dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.05,
        temporal_layers: int = 1,
    ):
        super().__init__()
        self.num_points = int(num_points)
        feature_dim = self.num_points * 6 + 8
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.tcn = nn.ModuleList()
        for i in range(4):
            dilation = 2 ** i
            self.tcn.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(dim),
                        "conv": nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * 2),
                            nn.SiLU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * 2, dim),
                        ),
                    }
                )
            )
        if temporal_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=max(1, min(8, dim // 64)),
                dim_feedforward=dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        else:
            self.temporal = nn.Identity()
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _diff(x: torch.Tensor) -> torch.Tensor:
        first = torch.zeros_like(x[:, :1])
        return torch.cat([first, x[:, 1:] - x[:, :-1]], dim=1)

    def _features(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = torch.nan_to_num(landmarks.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xy = landmarks[..., :2]
        center = xy.mean(dim=2, keepdim=True)
        centered = xy - center
        scale = centered.flatten(2).std(dim=-1, keepdim=True).view(xy.shape[0], xy.shape[1], 1, 1).clamp_min(1e-3)
        xy_n = centered / scale
        d1 = self._diff(xy_n)
        d2 = self._diff(d1)
        motion = torch.cat([xy_n, d1, d2], dim=-1)

        if motion.shape[2] < self.num_points:
            pad = torch.zeros(
                motion.shape[0],
                motion.shape[1],
                self.num_points - motion.shape[2],
                motion.shape[3],
                device=motion.device,
                dtype=motion.dtype,
            )
            motion = torch.cat([motion, pad], dim=2)
        motion = motion[:, :, : self.num_points]
        flat = motion.flatten(2)

        min_xy = xy_n.amin(dim=2)
        max_xy = xy_n.amax(dim=2)
        width_height = max_xy - min_xy
        mouth_open = xy_n[..., 1].std(dim=2, keepdim=True)
        mouth_width = width_height[..., :1]
        mouth_height = width_height[..., 1:2]
        area = mouth_width * mouth_height
        speed = d1.norm(dim=-1).mean(dim=2, keepdim=True)
        accel = d2.norm(dim=-1).mean(dim=2, keepdim=True)
        geom = torch.cat([center.squeeze(2), width_height, mouth_open, area, speed, accel], dim=-1)
        return torch.cat([flat, geom], dim=-1)

    def forward(self, landmarks: torch.Tensor, video_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input(self._features(landmarks))
        for block in self.tcn:
            y = block["norm"](x).transpose(1, 2)
            y = block["conv"](y).transpose(1, 2)
            x = x + F.silu(y)
            x = x + block["ffn"](x)
        key_padding_mask = None
        if video_mask is not None:
            key_padding_mask = ~video_mask.to(x.device, dtype=torch.bool)
        if isinstance(self.temporal, nn.TransformerEncoder):
            x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.temporal(x)
        return self.norm(x)


class Fusion(nn.Module):
    def __init__(self, dim: int = 512, fusion_type: str = "landmark_first", dropout: float = 0.05):
        super().__init__()
        self.fusion_type = fusion_type
        self.video_proj = nn.Linear(dim, dim)
        self.landmark_proj = nn.Linear(dim, dim)
        self.concat = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.constant_(self.gate[-1].bias, -0.5)
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "visual_only":
            return self.norm(self.video_proj(visual))
        if self.fusion_type == "landmark_only":
            return self.norm(self.landmark_proj(landmarks))
        joined = torch.cat([visual, landmarks], dim=-1)
        if self.fusion_type == "concat":
            return self.norm(self.concat(joined))
        if self.fusion_type == "gated":
            gate = torch.sigmoid(self.gate(joined))
            return self.norm(self.video_proj(visual) + gate * self.landmark_proj(landmarks))
        if self.fusion_type != "landmark_first":
            raise ValueError(f"Unsupported fusion_type={self.fusion_type}")
        gate = torch.sigmoid(self.gate(joined))
        return self.norm(self.landmark_proj(landmarks) + gate * self.video_proj(visual))


class DirectTCNMelDecoder(nn.Module):
    def __init__(
        self,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 80,
        num_layers: int = 6,
        dropout: float = 0.05,
        output_bias_init: float = -4.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(condition_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 4)
            padding = dilation * 2
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "conv": nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=5,
                            padding=padding,
                            dilation=dilation,
                        ),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.SiLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                    }
                )
            )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, output_bias_init)

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        with torch.no_grad():
            self.output.bias.copy_(mel_mean.to(device=self.output.bias.device, dtype=self.output.bias.dtype))

    def forward(self, condition: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if target_len is not None and condition.shape[1] != int(target_len):
            condition = F.interpolate(
                condition.transpose(1, 2),
                size=int(target_len),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
        x = self.input_proj(condition)
        for block in self.blocks:
            y = block["norm"](x).transpose(1, 2)
            y = block["conv"](y).transpose(1, 2)
            x = x + F.silu(y)
            x = x + block["ffn"](x)
        return self.output(self.output_norm(x))


class SirenMelDecoder(nn.Module):
    def __init__(
        self,
        condition_dim: int = 512,
        hidden_dim: int = 256,
        out_dim: int = 80,
        num_layers: int = 4,
        output_bias_init: float = -4.0,
    ):
        super().__init__()
        self.condition = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, hidden_dim), nn.SiLU())
        layers: list[nn.Module] = []
        in_dim = hidden_dim + 32
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.omega = nn.Parameter(torch.tensor(20.0))
        nn.init.constant_(self.out.bias, output_bias_init)

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        with torch.no_grad():
            self.out.bias.copy_(mel_mean.to(device=self.out.bias.device, dtype=self.out.bias.dtype))

    @staticmethod
    def _time_features(batch: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype).view(1, length, 1).expand(batch, -1, -1)
        freqs = torch.linspace(1.0, 16.0, 16, device=device, dtype=dtype).view(1, 1, -1)
        angles = t * freqs * torch.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, condition: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        if target_len is not None and condition.shape[1] != int(target_len):
            condition = F.interpolate(
                condition.transpose(1, 2),
                size=int(target_len),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
        c = self.condition(condition)
        t = self._time_features(c.shape[0], c.shape[1], c.device, c.dtype)
        x = torch.cat([c, t], dim=-1)
        omega = self.omega.clamp(1.0, 40.0).to(dtype=x.dtype)
        for layer in self.layers:
            x = torch.sin(omega * layer(x))
        return self.out(x)


class WindowedSpeechModel(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_landmark_points: int = 40,
        decoder_type: str = "direct_tcn",
        fusion_type: str = "landmark_first",
        encoder_width: int = 32,
        resnet_layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        visual_temporal_layers: int = 1,
        landmark_temporal_layers: int = 1,
        decoder_layers: int = 6,
        dropout: float = 0.05,
        output_bias_init: float = -4.0,
    ):
        super().__init__()
        self.visual = R2Plus1DVisualEncoder(
            dim=dim,
            width=encoder_width,
            layers=resnet_layers,
            temporal_layers=visual_temporal_layers,
            dropout=dropout,
        )
        self.landmarks = LandmarkMotionEncoder(
            num_points=num_landmark_points,
            dim=dim,
            temporal_layers=landmark_temporal_layers,
            dropout=dropout,
        )
        self.fusion = Fusion(dim=dim, fusion_type=fusion_type, dropout=dropout)
        if decoder_type == "direct_tcn":
            self.decoder = DirectTCNMelDecoder(
                condition_dim=dim,
                hidden_dim=dim,
                num_layers=decoder_layers,
                dropout=dropout,
                output_bias_init=output_bias_init,
            )
        elif decoder_type == "siren":
            self.decoder = SirenMelDecoder(
                condition_dim=dim,
                hidden_dim=max(128, dim // 2),
                num_layers=max(2, min(6, decoder_layers)),
                output_bias_init=output_bias_init,
            )
        else:
            raise ValueError(f"Unsupported decoder_type={decoder_type}")
        self.decoder_type = decoder_type

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        if hasattr(self.decoder, "set_output_bias"):
            self.decoder.set_output_bias(mel_mean)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        video = batch["video"]
        landmarks = batch["landmarks"]
        video_mask = batch.get("video_mask")
        visual = self.visual(video, video_mask=video_mask)
        landmark_features = self.landmarks(landmarks, video_mask=video_mask)
        return self.fusion(visual, landmark_features)

    def forward(self, batch: dict[str, torch.Tensor], target_len: Optional[int] = None) -> torch.Tensor:
        if target_len is None:
            mel = batch.get("mel")
            if mel is not None:
                target_len = int(mel.shape[1])
        memory = self.encode(batch)
        return self.decoder(memory, target_len=target_len)

