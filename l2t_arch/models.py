from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = int(max_len or lengths.max().item())
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


def monotonic_memory_mask(
    tgt_len: int,
    src_len: int,
    device: torch.device,
    frames_per_token: float,
    left_window: float,
    right_window: float,
    strength: float,
    hard: bool,
) -> torch.Tensor:
    tgt_pos = torch.arange(tgt_len, device=device, dtype=torch.float32).unsqueeze(1)
    src_pos = torch.arange(src_len, device=device, dtype=torch.float32).unsqueeze(0)
    center = torch.clamp(tgt_pos * float(frames_per_token), max=max(src_len - 1, 0))
    rel = src_pos - center
    outside_left = rel < -float(left_window)
    outside_right = rel > float(right_window)
    if hard:
        return outside_left | outside_right
    penalty = torch.relu(-rel - float(left_window)) + torch.relu(rel - float(right_window))
    denom = max(float(left_window), float(right_window), 1.0)
    return -float(strength) * penalty / denom


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class PLIFMembraneBlock(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dim = int(config["hidden_dim"])
        init_beta = float(config.get("plif_init_beta", 0.9))
        init_beta = min(max(init_beta, 1e-4), 1.0 - 1e-4)
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.beta_min = float(config.get("plif_beta_min", 0.1))
        self.beta_max = float(config.get("plif_beta_max", 0.995))
        beta_pos = (init_beta - self.beta_min) / max(self.beta_max - self.beta_min, 1e-6)
        beta_pos = min(max(beta_pos, 1e-4), 1.0 - 1e-4)
        self.beta_logit = nn.Parameter(torch.full((dim,), math.log(beta_pos / (1.0 - beta_pos))))
        self.threshold = float(config.get("plif_threshold", 1.0))
        self.slope = float(config.get("plif_slope", 5.0))
        self.use_spike_gate = bool(config.get("plif_spike_gate", True))
        self.gate_scale = nn.Parameter(torch.zeros(dim))
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
        self.norm = nn.LayerNorm(dim)

    def beta(self) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.beta_logit)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mem = x.new_zeros(x.shape[0], x.shape[2])
        beta = self.beta().to(dtype=x.dtype)
        mem_states = []
        spikes = []
        for t in range(x.shape[1]):
            current = self.in_proj(x[:, t])
            mem = beta * mem + (1.0 - beta) * current
            spike = torch.sigmoid((mem - self.threshold) * self.slope)
            mem_states.append(mem)
            spikes.append(spike)
            mem = mem * (1.0 - spike.detach())
        mem_seq = torch.stack(mem_states, dim=1)
        spike_seq = torch.stack(spikes, dim=1)
        if self.use_spike_gate:
            gate = 1.0 + torch.tanh(self.gate_scale).view(1, 1, -1) * spike_seq.detach()
            mem_seq = mem_seq * gate
        return self.norm(x + self.out_proj(self.dropout(mem_seq))), mem_seq, spike_seq


class VisualEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dim = int(config["hidden_dim"])
        self.kind = str(config.get("visual_encoder", "transformer"))
        self.input = nn.Linear(int(config.get("visual_dim", 512)), dim)
        self.pos = PositionalEncoding(dim)
        layers = int(config.get("visual_layers", 3))
        dropout = float(config.get("dropout", 0.1))
        if self.kind == "gru":
            self.encoder = nn.GRU(dim, dim // 2, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True, bidirectional=True)
        elif self.kind == "plif":
            self.encoder = nn.ModuleList([PLIFMembraneBlock(config) for _ in range(layers)])
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=int(config.get("num_heads", 4)),
                dim_feedforward=int(config.get("ffn_dim", dim * 4)),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, visuals: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.pos(self.input(visuals))
        stats = {}
        if self.kind == "gru":
            x, _ = self.encoder(x)
        elif self.kind == "plif":
            spike_rates = []
            beta_means = []
            valid = (~lengths_to_mask(lengths, x.shape[1])).unsqueeze(-1).to(dtype=x.dtype)
            valid_count = valid.sum().clamp_min(1.0) * x.shape[-1]
            for block in self.encoder:
                x, _, spikes = block(x)
                spike_rates.append((spikes * valid).sum() / valid_count)
                beta_means.append(block.beta().mean())
            stats["spike_rate"] = torch.stack(spike_rates).mean()
            stats["spike_reg"] = stats["spike_rate"]
            stats["beta_mean"] = torch.stack(beta_means).mean()
        else:
            x = self.encoder(x, src_key_padding_mask=lengths_to_mask(lengths, x.shape[1]))
        return self.norm(x), stats


class DualMemoryPLIFEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dim = int(config["hidden_dim"])
        self.input = nn.Linear(int(config.get("visual_dim", 512)), dim)
        self.pos = PositionalEncoding(dim)
        layers = int(config.get("visual_layers", 3))
        short_config = dict(config)
        long_config = dict(config)
        short_config["plif_init_beta"] = float(config.get("short_plif_init_beta", 0.75))
        long_config["plif_init_beta"] = float(config.get("long_plif_init_beta", 0.95))
        self.short_blocks = nn.ModuleList([PLIFMembraneBlock(short_config) for _ in range(layers)])
        self.long_blocks = nn.ModuleList([PLIFMembraneBlock(long_config) for _ in range(layers)])
        self.fuse = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
        self.norm = nn.LayerNorm(dim)

    def _run_branch(
        self,
        blocks: nn.ModuleList,
        x: torch.Tensor,
        valid: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spike_rates = []
        beta_means = []
        for block in blocks:
            x, _, spikes = block(x)
            spike_rates.append((spikes * valid).sum() / valid_count)
            beta_means.append(block.beta().mean())
        return x, torch.stack(spike_rates).mean(), torch.stack(beta_means).mean()

    def forward(self, visuals: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.pos(self.input(visuals))
        valid = (~lengths_to_mask(lengths, x.shape[1])).unsqueeze(-1).to(dtype=x.dtype)
        valid_count = valid.sum().clamp_min(1.0) * x.shape[-1]
        short, short_spike, short_beta = self._run_branch(self.short_blocks, x, valid, valid_count)
        long, long_spike, long_beta = self._run_branch(self.long_blocks, x, valid, valid_count)
        pair = torch.cat([short, long], dim=-1)
        gate = torch.sigmoid(self.gate(pair))
        mixed = gate * short + (1.0 - gate) * long
        fused = self.norm(mixed + self.fuse(self.dropout(pair)))
        spike_rate = 0.5 * (short_spike + long_spike)
        return fused, {
            "spike_rate": spike_rate,
            "spike_reg": spike_rate,
            "short_beta_mean": short_beta,
            "long_beta_mean": long_beta,
            "beta_mean": 0.5 * (short_beta + long_beta),
            "fusion_gate_mean": gate.mean(),
        }


class TextOnlyRefiner(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int) -> None:
        super().__init__()
        dim = int(config["hidden_dim"])
        dropout = float(config.get("dropout", 0.1))
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim)
        enc_layer = nn.TransformerEncoderLayer(dim, int(config.get("num_heads", 4)), int(config.get("ffn_dim", dim * 4)), dropout, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, int(config.get("num_heads", 4)), int(config.get("ffn_dim", dim * 4)), dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(config.get("text_layers", 3)))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(config.get("decoder_layers", 3)))
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, vtp_tokens: torch.Tensor, gt_in: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        src_mask = vtp_tokens.eq(0)
        tgt_mask = gt_in.eq(0)
        memory = self.encoder(self.pos(self.embed(vtp_tokens)), src_key_padding_mask=src_mask)
        tgt = self.pos(self.embed(gt_in))
        h = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask(tgt.shape[1], tgt.device),
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )
        return {"logits": self.out(h)}


class VisualCTCModel(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int) -> None:
        super().__init__()
        self.visual = VisualEncoder(config)
        self.out = nn.Linear(int(config["hidden_dim"]), vocab_size)

    def forward(self, visuals: torch.Tensor, visual_lengths: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        h, stats = self.visual(visuals, visual_lengths)
        return {"ctc_logits": self.out(h), "ctc_lengths": visual_lengths, **stats}


class VisualPLIFSeq2Seq(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int) -> None:
        super().__init__()
        dim = int(config["hidden_dim"])
        dropout = float(config.get("dropout", 0.1))
        self.visual = VisualEncoder(config)
        self.gt_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim)
        dec_layer = nn.TransformerDecoderLayer(
            dim,
            int(config.get("num_heads", 4)),
            int(config.get("ffn_dim", dim * 4)),
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(config.get("decoder_layers", 3)))
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, visuals: torch.Tensor, visual_lengths: torch.Tensor, gt_in: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        memory, stats = self.visual(visuals, visual_lengths)
        tgt_mask = gt_in.eq(0)
        tgt = self.pos(self.gt_embed(gt_in))
        h = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask(tgt.shape[1], tgt.device),
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=lengths_to_mask(visual_lengths, memory.shape[1]),
        )
        return {"logits": self.out(h), **stats}


class DualPLIFMonotonicSeq2Seq(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int) -> None:
        super().__init__()
        dim = int(config["hidden_dim"])
        dropout = float(config.get("dropout", 0.1))
        self.visual = DualMemoryPLIFEncoder(config)
        self.gt_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim)
        dec_layer = nn.TransformerDecoderLayer(
            dim,
            int(config.get("num_heads", 4)),
            int(config.get("ffn_dim", dim * 4)),
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(config.get("decoder_layers", 3)))
        self.out = nn.Linear(dim, vocab_size)
        self.ctc_aux = bool(config.get("ctc_aux", False))
        self.ctc_out = nn.Linear(dim, vocab_size) if self.ctc_aux else None
        self.monotonic_attention = bool(config.get("monotonic_attention", True))
        self.monotonic_frames_per_token = float(config.get("monotonic_frames_per_token", 2.0))
        self.monotonic_left_window = float(config.get("monotonic_left_window", 8.0))
        self.monotonic_right_window = float(config.get("monotonic_right_window", 16.0))
        self.monotonic_bias_strength = float(config.get("monotonic_bias_strength", 2.0))
        self.monotonic_hard_window = bool(config.get("monotonic_hard_window", False))

    def forward(self, visuals: torch.Tensor, visual_lengths: torch.Tensor, gt_in: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        memory, stats = self.visual(visuals, visual_lengths)
        tgt_mask = gt_in.eq(0)
        tgt = self.pos(self.gt_embed(gt_in))
        memory_mask = None
        if self.monotonic_attention:
            memory_mask = monotonic_memory_mask(
                tgt.shape[1],
                memory.shape[1],
                tgt.device,
                self.monotonic_frames_per_token,
                self.monotonic_left_window,
                self.monotonic_right_window,
                self.monotonic_bias_strength,
                self.monotonic_hard_window,
            )
        memory_padding_mask = lengths_to_mask(visual_lengths, memory.shape[1])
        if memory_mask is not None and torch.is_floating_point(memory_mask):
            memory_padding_mask = memory_padding_mask.to(dtype=memory_mask.dtype).masked_fill(memory_padding_mask, float("-inf"))
        h = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask(tgt.shape[1], tgt.device),
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        output = {"logits": self.out(h), **stats}
        if self.ctc_out is not None:
            output["ctc_logits"] = self.ctc_out(memory)
            output["ctc_lengths"] = visual_lengths
        return output


class DualPathRefiner(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int) -> None:
        super().__init__()
        dim = int(config["hidden_dim"])
        dropout = float(config.get("dropout", 0.1))
        self.visual = VisualEncoder(config)
        self.text_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gt_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim)
        text_layer = nn.TransformerEncoderLayer(dim, int(config.get("num_heads", 4)), int(config.get("ffn_dim", dim * 4)), dropout, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, int(config.get("num_heads", 4)), int(config.get("ffn_dim", dim * 4)), dropout, batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=int(config.get("text_layers", 2)))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(config.get("decoder_layers", 3)))
        self.out = nn.Linear(dim, vocab_size)
        self.ctc_out = nn.Linear(dim, vocab_size)

    def forward(self, visuals: torch.Tensor, visual_lengths: torch.Tensor, vtp_tokens: torch.Tensor, gt_in: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        visual_memory, stats = self.visual(visuals, visual_lengths)
        text_mask = vtp_tokens.eq(0)
        text_memory = self.text_encoder(self.pos(self.text_embed(vtp_tokens)), src_key_padding_mask=text_mask)
        memory = torch.cat([visual_memory, text_memory], dim=1)
        memory_mask = torch.cat([lengths_to_mask(visual_lengths, visual_memory.shape[1]), text_mask], dim=1)
        tgt_mask = gt_in.eq(0)
        tgt = self.pos(self.gt_embed(gt_in))
        h = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask(tgt.shape[1], tgt.device),
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )
        return {
            "logits": self.out(h),
            "ctc_logits": self.ctc_out(visual_memory),
            "ctc_lengths": visual_lengths,
            **stats,
        }


def build_model(config: dict[str, Any], vocab_size: int) -> nn.Module:
    kind = str(config["model"]["type"])
    if kind == "text_only":
        return TextOnlyRefiner(config["model"], vocab_size)
    if kind == "visual_ctc":
        return VisualCTCModel(config["model"], vocab_size)
    if kind == "visual_plif_seq2seq":
        return VisualPLIFSeq2Seq(config["model"], vocab_size)
    if kind == "dual_plif_monotonic":
        return DualPLIFMonotonicSeq2Seq(config["model"], vocab_size)
    if kind == "dual_path":
        return DualPathRefiner(config["model"], vocab_size)
    raise ValueError(f"Unknown model.type: {kind}")
