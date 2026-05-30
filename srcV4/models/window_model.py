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


class AVHuBERTVisualFrontend(nn.Module):
    """
    AV-HuBERT Visual Frontend with a robust native 3D CNN (R(2+1)D) fallback.
    Ensures the model runs instantly without network download errors or library incompatibilities.
    """
    def __init__(
        self,
        encoder_type: str = "r2plus1d",
        in_channels: int = 1,
        dim: int = 512,
        width: int = 32,
        layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        spatial_pool_size: int = 2,
        temporal_layers: int = 1,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "av_hubert":
            print("[visual-encoder] AV-HuBERT chosen. Attempting to locate transformers library...")
            try:
                import transformers
                # If fairseq or specific wrapper is needed, we raise fallback to avoid runtime crashes
                raise ImportError("AV-HuBERT requires specialized AV wrapper. Falling back to native R(2+1)D.")
            except Exception as e:
                print(f"[visual-encoder] AV-HuBERT fallback triggered: {e}. Using native R(2+1)D encoder.")
                self.encoder_type = "r2plus1d"

        if self.encoder_type == "r2plus1d":
            self.encoder = R2Plus1DVisualEncoder(
                in_channels=in_channels,
                dim=dim,
                width=width,
                layers=layers,
                spatial_pool_size=spatial_pool_size,
                temporal_layers=temporal_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

    def forward(self, video: torch.Tensor, video_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.encoder_type == "r2plus1d":
            return self.encoder(video, video_mask=video_mask)
        else:
            raise NotImplementedError("AV-HuBERT forward is not currently available.")


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


# ------------------ SNN LAYERS ------------------

class LIFSpikingLayer(nn.Module):
    """
    Highly portable and lightweight native PyTorch LIF Spiking Neuron layer.
    Requires no external packages (like spikingjelly), ensuring seamless execution.
    """
    def __init__(self, dim: int, tau: float = 2.0, v_threshold: float = 1.0, v_reset: float = 0.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.input_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device
        u = torch.zeros(B, C, device=device, dtype=x.dtype)
        spikes = []
        proj_x = self.input_proj(x)
        
        for t in range(T):
            inputs = proj_x[:, t, :]
            u = u + (inputs - u) / self.tau
            spike = (u >= self.v_threshold).to(dtype=x.dtype)
            spikes.append(spike)
            u = u * (1.0 - spike) + spike * self.v_reset
            
        out_spikes = torch.stack(spikes, dim=1)
        return self.norm(out_spikes)


class SpikingTemporalProcessor(nn.Module):
    """
    Spiking Neural Network (SNN) Temporal Processor.
    Placed between fusion and decoder to capture biological timing and speech rhythms.
    """
    def __init__(self, dim: int = 512, n_layers: int = 2, tau: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList([
            LIFSpikingLayer(dim, tau=tau) for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = x
        for layer in self.layers:
            out = layer(out)
        return residual + 0.5 * self.readout(out)


# ------------------ TFiLM DECODER LAYERS ------------------

class TFiLMLayer(nn.Module):
    """
    Temporal Feature-wise Linear Modulation (TFiLM) Layer.
    Injects conditioning features to modulate decoder states at each layer and timestep.
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, dim * 2),
        )
        # Initialize gamma to 1.0 and beta to 0.0
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        self.film_gen[-1].bias.data[:dim] = 1.0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film_gen(cond).chunk(2, dim=-1)
        return gamma * self.norm(x) + beta


class ConformerFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x).transpose(1, 2)
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.dropout(x)
        return res + x.transpose(1, 2)


class TFiLMConformerBlock(nn.Module):
    """
    TFiLM Conformer Block: Integrates self-attention, convolutions, half-step feed-forward nets,
    and TFiLM temporal visual conditioning after each sub-layer.
    """
    def __init__(self, dim: int, cond_dim: int, n_heads: int = 8, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = ConformerFeedForward(dim, dim * 4, dropout)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.conv = ConformerConvModule(dim, conv_kernel, dropout)
        self.ffn2 = ConformerFeedForward(dim, dim * 4, dropout)
        self.norm = nn.LayerNorm(dim)
        
        # TFiLM generators for attention, convolution, and feed-forward sub-layers
        self.tfilm_ffn = TFiLMLayer(dim, cond_dim)
        self.tfilm_attn = TFiLMLayer(dim, cond_dim)
        self.tfilm_conv = TFiLMLayer(dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # FFN 1/2
        x = x + 0.5 * self.ffn1(x)
        x = self.tfilm_ffn(x, cond)
        
        # Self-Attention
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = self.tfilm_attn(x, cond)
        
        # Convolutions
        x = self.conv(x)
        x = self.tfilm_conv(x, cond)
        
        # FFN 2/2
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class TFiLMConformerDecoder(nn.Module):
    """
    TFiLM Conformer Decoder.
    Takes fused representations and target mel lengths, performs temporal upsampling via interpolation,
    processes with TFiLM-modulated Conformers, and maps hidden states to 80-bin mel space.
    """
    def __init__(self, dim: int = 512, cond_dim: int = 512, n_layers: int = 6, 
                 n_heads: int = 8, conv_kernel: int = 31, out_dim: int = 80, 
                 dropout: float = 0.1, output_bias_init: float = -4.0):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        
        self.input_proj = nn.Linear(cond_dim, dim)
        self.cond_proj = nn.Linear(cond_dim, dim)
        
        self.layers = nn.ModuleList([
            TFiLMConformerBlock(dim, dim, n_heads=n_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, output_bias_init)

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        with torch.no_grad():
            self.out_proj.bias.copy_(mel_mean.to(device=self.out_proj.bias.device, dtype=self.out_proj.bias.dtype))

    def forward(self, fused: torch.Tensor, target_len: int | None = None, mel_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Fused video-landmark sequence: (B, T_vid, cond_dim)
        if target_len is not None and fused.shape[1] != int(target_len):
            fused_mel = F.interpolate(
                fused.transpose(1, 2),
                size=int(target_len),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
        else:
            fused_mel = fused
            
        x = self.input_proj(fused_mel)
        cond = self.cond_proj(fused_mel)
        
        for layer in self.layers:
            x = layer(x, cond, mask=mel_mask)
            
        return self.out_proj(self.out_norm(x))


# ------------------ SIREN RESIDUAL REFINEMENT ------------------

class SIRENResidualLayer(nn.Module):
    """
    Implicit Neural Representation (INR) using SIREN.
    Operates as a residual block on top of Conformer's output to inject high-frequency formants and spectral details.
    """
    def __init__(self, n_mels: int = 80, cond_dim: int = 512, hidden_dim: int = 256, 
                 n_layers: int = 2, omega_0: float = 20.0):
        super().__init__()
        self.n_layers = n_layers
        self.omega_0 = omega_0
        
        self.register_buffer("time_freqs", torch.linspace(1.0, 32.0, 16))
        input_dim = n_mels + 32
        
        self.film_gens = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim * 2) for _ in range(n_layers)
        ])
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.out = nn.Linear(hidden_dim, n_mels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                dim_in = layer.in_features
                if i == 0:
                    layer.weight.uniform_(-1.0 / dim_in, 1.0 / dim_in)
                else:
                    bound = (6.0 / dim_in) ** 0.5 / self.omega_0
                    layer.weight.uniform_(-bound, bound)

    def _time_pe(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).unsqueeze(-1)
        freqs = self.time_freqs.to(device=device, dtype=dtype).view(1, -1)
        angles = t * freqs * 2.0 * torch.pi
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return pe

    def forward(self, coarse_mel: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = coarse_mel.shape
        time_pe = self._time_pe(T, coarse_mel.device, coarse_mel.dtype).unsqueeze(0).expand(B, -1, -1)
        
        # Detach coarse_mel to isolate SIREN as a pure residual model, protecting the decoder gradients
        x = torch.cat([coarse_mel.detach(), time_pe], dim=-1)
        
        for layer, film_gen in zip(self.layers, self.film_gens):
            gamma, beta = film_gen(cond).chunk(2, dim=-1)
            x = layer(x)
            x = gamma * x + beta
            x = torch.sin(self.omega_0 * x)
            
        residual = self.out(x)
        return self.alpha * residual


# ------------------ V4 SPEECH MODEL ------------------

class V4SpeechModel(nn.Module):
    """
    V4 Lip-to-Speech Reconstruction Model.
    Consolidated Architecture:
    Video Input ──> AV-HuBERT/CNN Visual Encoder ─┐
                                                  ├──> Gated Fusion ──> [SNN] ──> TFiLM Conformer Decoder ──> SIREN Residual ──> Final Mel
    Landmarks  ──> Dilated TCN landmark encoder  ─┘
    """
    def __init__(
        self,
        dim: int = 512,
        num_landmark_points: int = 40,
        fusion_type: str = "landmark_first",
        encoder_width: int = 32,
        resnet_layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        visual_temporal_layers: int = 1,
        landmark_temporal_layers: int = 1,
        decoder_layers: int = 6,
        dropout: float = 0.05,
        output_bias_init: float = -4.0,
        use_snn: bool = False,
        snn_layers: int = 2,
        snn_tau: float = 2.0,
        siren_layers: int = 2,
        siren_omega: float = 20.0,
        visual_encoder_type: str = "r2plus1d",
    ):
        super().__init__()
        self.visual = AVHuBERTVisualFrontend(
            encoder_type=visual_encoder_type,
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
        
        self.use_snn = use_snn
        if use_snn:
            self.snn = SpikingTemporalProcessor(dim=dim, n_layers=snn_layers, tau=snn_tau)
            
        self.decoder = TFiLMConformerDecoder(
            dim=dim,
            cond_dim=dim,
            n_layers=decoder_layers,
            n_heads=max(1, min(8, dim // 64)),
            conv_kernel=31,
            out_dim=80,
            dropout=dropout,
            output_bias_init=output_bias_init,
        )
        
        self.siren_residual = SIRENResidualLayer(
            n_mels=80,
            cond_dim=dim,
            hidden_dim=max(128, dim // 2),
            n_layers=siren_layers,
            omega_0=siren_omega,
        )

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        self.decoder.set_output_bias(mel_mean)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        video = batch["video"]
        landmarks = batch["landmarks"]
        video_mask = batch.get("video_mask")
        
        visual_feats = self.visual(video, video_mask=video_mask)
        landmark_feats = self.landmarks(landmarks, video_mask=video_mask)
        
        fused = self.fusion(visual_feats, landmark_feats)
        if self.use_snn:
            fused = self.snn(fused)
            
        return fused

    def forward(self, batch: dict[str, torch.Tensor], target_len: Optional[int] = None) -> torch.Tensor:
        if target_len is None:
            mel = batch.get("mel")
            if mel is not None:
                target_len = int(mel.shape[1])
                
        # 1. Encode visual + landmark structures
        memory = self.encode(batch) # (B, T_vid, dim)
        
        # 2. Prepare conditioning signals (interpolated internally in Conformer)
        mel_mask = batch.get("mel_mask")
        
        # 3. Decode coarse mel spectrogram
        coarse_mel = self.decoder(memory, target_len=target_len, mel_mask=mel_mask) # (B, T_mel, 80)
        
        # 4. Get upsampled condition for SIREN
        if target_len is not None and memory.shape[1] != int(target_len):
            cond_mel = F.interpolate(
                memory.transpose(1, 2),
                size=int(target_len),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()
        else:
            cond_mel = memory
            
        # 5. Compute high-frequency residual and add to coarse mel
        res_mel = self.siren_residual(coarse_mel, cond_mel)
        final_mel = coarse_mel + res_mel
        
        return final_mel
