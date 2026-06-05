from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from srcV2.models.r2plus1d_inr import MelTemporalBlock, R2INRMemoryEncoder, TimeFourier
from srcV4.models.window_model import SIRENResidualLayer, TFiLMConformerDecoder


class AVFeatureAdapter(nn.Module):
    """Project cached AV-HuBERT visual features into the reconstruction space."""

    def __init__(self, input_dim: int = 768, dim: int = 512, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        if int(layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=max(1, min(8, dim // 64)),
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=int(layers))
        else:
            self.temporal = nn.Identity()

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0))
        key_padding_mask = None if mask is None else ~mask.bool()
        if isinstance(self.temporal, nn.TransformerEncoder):
            x = self.temporal(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.temporal(x)
        if mask is not None:
            x = x * mask.to(x.device, x.dtype).unsqueeze(-1)
        return x


class ProsodyHead(nn.Module):
    def __init__(self, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowMatchingMelRefiner(nn.Module):
    """Small conditional rectified-flow head trained as an auxiliary mel generator."""

    def __init__(self, dim: int = 512, n_mels: int = 80, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_mels = int(n_mels)
        self.time = TimeFourier(dim, num_freqs=32, max_freq=8.0)
        self.input = nn.Sequential(
            nn.LayerNorm(n_mels + dim * 2),
            nn.Linear(n_mels + dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([MelTemporalBlock(dim, dilation=2**i, dropout=dropout) for i in range(int(layers))])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_mels))
        nn.init.zeros_(self.out[-1].bias)
        nn.init.normal_(self.out[-1].weight, 0.0, 1e-3)

    def velocity(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if t.ndim == 1:
            t = t.view(-1, 1).expand(-1, x_t.shape[1])
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t.expand(-1, x_t.shape[1])
        t_embed = self.time(t.to(device=x_t.device, dtype=x_t.dtype))
        h = self.input(torch.cat([x_t.float(), cond.float(), t_embed.float()], dim=-1))
        for block in self.blocks:
            h = block(h, mask)
        v = self.out(h)
        if mask is not None:
            v = v * mask.to(v.device, v.dtype).unsqueeze(-1)
        return v

    def loss(self, target: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(target.float())
        t = torch.rand(target.shape[0], 1, 1, device=target.device, dtype=target.dtype)
        x_t = (1.0 - t) * noise + t * target.float()
        v_target = target.float() - noise
        v_pred = self.velocity(x_t, cond, t.squeeze(-1), mask)
        if mask is None:
            return F.mse_loss(v_pred, v_target)
        mask_f = mask.to(v_pred.device, v_pred.dtype).unsqueeze(-1)
        denom = (mask_f.sum() * target.shape[-1]).clamp_min(1.0)
        return ((v_pred - v_target).pow(2) * mask_f).sum() / denom

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, mask: torch.Tensor | None = None, steps: int = 8) -> torch.Tensor:
        steps = max(1, int(steps))
        x = torch.randn(cond.shape[0], cond.shape[1], self.n_mels, device=cond.device, dtype=cond.dtype)
        for idx in range(steps):
            t = torch.full((cond.shape[0], 1), float(idx) / float(steps), device=cond.device, dtype=cond.dtype)
            x = x + self.velocity(x, cond, t, mask=mask) / float(steps)
            if mask is not None:
                x = x * mask.to(x.device, x.dtype).unsqueeze(-1)
        return x


class V10PretrainedFusionSpeechModel(nn.Module):
    """V6 alignment/decoder upgraded with optional pretrained AV-HuBERT features.

    The default path works on existing R2INR cache files. When `av_features`
    are present in the batch, the model injects them as additional memory
    tokens for mel-time cross-attention and fuses them into frame memory.
    """

    def __init__(
        self,
        dim: int = 512,
        spatial_tokens: int = 4,
        num_points: int = 40,
        dropout: float = 0.1,
        decoder_layers: int = 6,
        heads: int = 8,
        num_units: int = 0,
        use_content_units: bool = True,
        unit_temperature: float = 1.0,
        unit_teacher_prob: float = 0.0,
        detach_unit_condition: bool = True,
        detach_content_hidden: bool = True,
        siren_layers: int = 2,
        siren_omega: float = 20.0,
        use_avhubert_features: bool = False,
        av_feature_dim: int = 768,
        av_feature_layers: int = 1,
        av_feature_scale: float = 1.0,
        use_prosody_head: bool = True,
        use_flow_refiner: bool = False,
        flow_layers: int = 4,
        flow_steps: int = 8,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_units = int(num_units)
        self.use_content_units = bool(use_content_units and self.num_units > 0)
        self.unit_temperature = float(unit_temperature)
        self.unit_teacher_prob = float(unit_teacher_prob)
        self.detach_unit_condition = bool(detach_unit_condition)
        self.detach_content_hidden = bool(detach_content_hidden)
        self.use_avhubert_features = bool(use_avhubert_features)
        self.av_feature_scale = float(av_feature_scale)
        self.use_prosody_head = bool(use_prosody_head)
        self.use_flow_refiner = bool(use_flow_refiner)
        self.flow_steps = int(flow_steps)

        self.encoder = R2INRMemoryEncoder(
            dim=dim,
            spatial_tokens=spatial_tokens,
            num_points=num_points,
            dropout=dropout,
        )
        if self.use_avhubert_features:
            self.av_adapter = AVFeatureAdapter(
                input_dim=max(1, int(av_feature_dim)),
                dim=dim,
                layers=av_feature_layers,
                dropout=dropout,
            )
            self.av_frame_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
            nn.init.constant_(self.av_frame_gate[-1].bias, -1.0)
            self.av_frame_norm = nn.LayerNorm(dim)
            self.av_global_norm = nn.LayerNorm(dim)

        self.time = TimeFourier(dim, num_freqs=64, max_freq=96.0)
        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads=max(1, int(heads)), dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.aligned_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.pre_refine = nn.ModuleList(
            [
                MelTemporalBlock(dim, dilation=1, dropout=dropout),
                MelTemporalBlock(dim, dilation=2, dropout=dropout),
                MelTemporalBlock(dim, dilation=4, dropout=dropout),
                MelTemporalBlock(dim, dilation=8, dropout=dropout),
            ]
        )

        if self.use_content_units:
            self.unit_head = nn.Linear(dim, self.num_units)
            self.unit_embedding = nn.Embedding(self.num_units, dim)
            self.content_fusion = nn.Sequential(
                nn.Linear(dim * 3, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )

        if self.use_prosody_head:
            self.prosody_head = ProsodyHead(dim=dim, dropout=dropout)

        self.decoder = TFiLMConformerDecoder(
            dim=dim,
            cond_dim=dim,
            n_layers=decoder_layers,
            n_heads=max(1, min(8, dim // 64)),
            conv_kernel=31,
            out_dim=80,
            dropout=dropout,
            output_bias_init=-4.0,
        )
        self.siren_residual = SIRENResidualLayer(
            n_mels=80,
            cond_dim=dim,
            hidden_dim=max(128, dim // 2),
            n_layers=siren_layers,
            omega_0=siren_omega,
        )
        if self.use_flow_refiner:
            self.flow_refiner = FlowMatchingMelRefiner(dim=dim, n_mels=80, layers=flow_layers, dropout=dropout)

    def set_output_bias(self, mel_mean: torch.Tensor) -> None:
        self.decoder.set_output_bias(mel_mean)

    @staticmethod
    def _resize_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.shape[1] == int(target_len):
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=int(target_len),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).contiguous()

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= 1:
            return torch.zeros_like(x)
        d = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)

    @staticmethod
    def _resize_units(units: torch.Tensor, target_len: int) -> torch.Tensor:
        if units.shape[1] == int(target_len):
            return units
        return F.interpolate(units.float().unsqueeze(1), size=int(target_len), mode="nearest").squeeze(1).long()

    def _target_len(self, batch: dict[str, torch.Tensor]) -> int:
        if batch.get("mel_mask") is not None:
            return int(batch["mel_mask"].shape[1])
        if batch.get("mel") is not None:
            return int(batch["mel"].shape[1])
        return int(batch["mel_times"].shape[1])

    def _augment_encoded_with_av(
        self,
        encoded: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        if not self.use_avhubert_features or "av_features" not in batch:
            return encoded, None
        feats = batch["av_features"]
        if feats.shape[-1] != self.av_adapter.input_dim:
            raise ValueError(f"Expected av_feature_dim={self.av_adapter.input_dim}, got {feats.shape[-1]}")
        mask = batch.get("av_feature_mask")
        if mask is None:
            mask = torch.ones(feats.shape[:2], device=feats.device, dtype=torch.bool)
        else:
            mask = mask.to(feats.device, dtype=torch.bool)
        present = batch.get("av_feature_present")
        if present is not None:
            mask = mask & present.to(mask.device, dtype=torch.bool).unsqueeze(1)
        if not bool(mask.any()):
            return encoded, None

        av = self.av_adapter(feats, mask=mask) * self.av_feature_scale
        memory = torch.cat([encoded["memory"], av], dim=1)
        memory_mask = torch.cat([encoded["memory_mask"].bool(), mask], dim=1)

        frame_av = self._resize_time(av, encoded["frame_memory"].shape[1])
        frame_mask = encoded["frame_mask"].to(frame_av.device, dtype=torch.bool)
        gate = torch.sigmoid(self.av_frame_gate(torch.cat([encoded["frame_memory"], frame_av], dim=-1)))
        if present is not None:
            gate = gate * present.to(gate.device, gate.dtype).view(-1, 1, 1)
        frame_memory = self.av_frame_norm(encoded["frame_memory"] + gate * frame_av)
        frame_memory = frame_memory * frame_mask.to(frame_memory.device, frame_memory.dtype).unsqueeze(-1)

        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        av_global = (av * mask.unsqueeze(-1).float()).sum(dim=1) / denom
        if present is not None:
            av_global = av_global * present.to(av_global.device, av_global.dtype).unsqueeze(-1)
        global_token = self.av_global_norm(encoded["global"] + av_global)

        out = dict(encoded)
        out["memory"] = memory
        out["memory_mask"] = memory_mask
        out["frame_memory"] = frame_memory
        out["global"] = global_token
        return out, av

    def _unit_condition(self, unit_logits: torch.Tensor, h: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        probs = torch.softmax(unit_logits.float() / max(0.1, self.unit_temperature), dim=-1).to(h.dtype)
        if self.detach_unit_condition:
            probs = probs.detach()
        soft_units = probs @ self.unit_embedding.weight.to(device=h.device, dtype=h.dtype)
        if not self.training or self.unit_teacher_prob <= 0 or "speech_units" not in batch:
            return soft_units

        targets = self._resize_units(batch["speech_units"].to(h.device), unit_logits.shape[1])
        valid = targets.ge(0) & targets.lt(self.num_units)
        safe_targets = targets.clamp(0, max(0, self.num_units - 1))
        teacher_units = self.unit_embedding(safe_targets).to(h.dtype)
        mixed = (1.0 - self.unit_teacher_prob) * soft_units + self.unit_teacher_prob * teacher_units
        return torch.where(valid.unsqueeze(-1), mixed, soft_units)

    def _mel_aligned_memory(self, encoded: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], target_len: int) -> torch.Tensor:
        mel_times = batch["mel_times"][:, :target_len]
        q_time = self.time(mel_times)
        q = self.query(q_time + encoded["global"].unsqueeze(1))
        key_padding_mask = ~encoded["memory_mask"].bool()
        context, _ = self.attn(q, encoded["memory"], encoded["memory"], key_padding_mask=key_padding_mask, need_weights=False)
        aligned = self.aligned_proj(self._resize_time(encoded["frame_memory"], target_len))
        h = self.cross_norm(q + context + aligned)
        mel_mask = batch.get("mel_mask")
        for block in self.pre_refine:
            h = block(h, mel_mask)
        return h

    def flow_loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_flow_refiner or "flow_cond" not in outputs:
            return batch["mel"].new_tensor(0.0)
        return self.flow_refiner.loss(batch["mel"], outputs["flow_cond"], batch.get("mel_mask"))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        target_len = self._target_len(batch)
        encoded = self.encoder(batch)
        encoded, av_features = self._augment_encoded_with_av(encoded, batch)
        h = self._mel_aligned_memory(encoded, batch, target_len)
        mel_mask = batch.get("mel_mask")
        unit_logits = None

        if self.use_content_units:
            unit_logits = self.unit_head(h)
            unit_cond = self._unit_condition(unit_logits, h, batch)
            h_content = h.detach() if self.detach_content_hidden else h
            h = self.content_fusion(torch.cat([unit_cond, h_content, self._delta(h_content)], dim=-1))
            if mel_mask is not None:
                h = h * mel_mask.to(h.device, h.dtype).unsqueeze(-1)

        prosody = self.prosody_head(h) if self.use_prosody_head else None
        coarse = self.decoder(h, target_len=target_len, mel_mask=mel_mask)
        residual = self.siren_residual(coarse, h)
        mel = coarse + residual
        if mel_mask is not None:
            mel = mel * mel_mask.to(mel.device, mel.dtype).unsqueeze(-1)

        if self.use_flow_refiner and bool(batch.get("sample_flow", False)):
            mel = self.flow_refiner.sample(h, mask=mel_mask, steps=int(batch.get("flow_steps", self.flow_steps)))

        if bool(batch.get("return_aux", False)):
            return {
                "mel": mel,
                "coarse_mel": coarse,
                "residual_mel": residual,
                "unit_logits": unit_logits,
                "prosody": prosody,
                "flow_cond": h,
                "av_features": av_features,
            }
        return mel


def build_model_from_config(config: dict) -> V10PretrainedFusionSpeechModel:
    return V10PretrainedFusionSpeechModel(
        dim=int(config.get("dim", 512)),
        spatial_tokens=int(config.get("spatial_tokens", 4)),
        num_points=int(config.get("num_landmark_points", 40)),
        dropout=0.0,
        decoder_layers=int(config.get("decoder_layers", 6)),
        heads=int(config.get("heads", 8)),
        num_units=int(config.get("num_units", 0)),
        use_content_units=bool(config.get("use_content_units", False)),
        unit_temperature=float(config.get("unit_temperature", 1.0)),
        unit_teacher_prob=0.0,
        detach_unit_condition=bool(config.get("detach_unit_condition", True)),
        detach_content_hidden=bool(config.get("detach_content_hidden", True)),
        siren_layers=int(config.get("siren_layers", 2)),
        siren_omega=float(config.get("siren_omega", 20.0)),
        use_avhubert_features=bool(config.get("use_avhubert_features", False)),
        av_feature_dim=int(config.get("av_feature_dim", 768)),
        av_feature_layers=int(config.get("av_feature_layers", 1)),
        av_feature_scale=float(config.get("av_feature_scale", 1.0)),
        use_prosody_head=bool(config.get("use_prosody_head", True)),
        use_flow_refiner=bool(config.get("use_flow_refiner", False)),
        flow_layers=int(config.get("flow_layers", 4)),
        flow_steps=int(config.get("flow_steps", 8)),
    )
