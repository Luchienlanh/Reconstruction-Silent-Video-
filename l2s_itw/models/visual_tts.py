from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from l2s_itw.utils import lengths_to_mask


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)


class TransformerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.positional = SinusoidalPositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.positional(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


class VisualTTS(nn.Module):
    """Visual text-to-speech model.

    Visual embeddings query text-unit embeddings with attention. The aligned
    representation is upsampled to mel frame rate, speaker-conditioned, then
    decoded into mel-spectrogram frames.
    """

    def __init__(self, config: dict[str, Any], vocab_size: int, pad_id: int = 0) -> None:
        super().__init__()
        model_config = config["model"]
        dim = int(model_config["hidden_dim"])
        heads = int(model_config["num_heads"])
        ffn_dim = int(model_config["ffn_dim"])
        dropout = float(model_config["dropout"])

        self.pad_id = pad_id
        self.mel_upsample = int(model_config["mel_upsample"])
        self.mel_frames_per_video_frame = float(model_config.get("mel_frames_per_video_frame", self.mel_upsample))
        self.n_mels = int(model_config["n_mels"])

        self.text_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.visual_projection = nn.Linear(int(model_config["visual_dim"]), dim)
        self.speaker_projection = nn.Linear(int(model_config["speaker_dim"]), dim)

        self.text_encoder = TransformerStack(
            dim=dim,
            layers=int(model_config["text_layers"]),
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.visual_encoder = TransformerStack(
            dim=dim,
            layers=int(model_config["visual_layers"]),
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.visual_text_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = TransformerStack(
            dim=dim,
            layers=int(model_config["decoder_layers"]),
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.mel_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.n_mels),
        )

    def forward(
        self,
        visuals: torch.Tensor,
        visual_lengths: torch.Tensor,
        tokens: torch.Tensor,
        text_lengths: torch.Tensor,
        speakers: torch.Tensor,
        target_mel_lengths: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        visual_padding = lengths_to_mask(visual_lengths, visuals.shape[1])
        text_padding = lengths_to_mask(text_lengths, tokens.shape[1])

        text = self.text_embedding(tokens)
        text = self.text_encoder(text, padding_mask=text_padding)

        visual = self.visual_projection(visuals)
        visual = self.visual_encoder(visual, padding_mask=visual_padding)

        aligned, attention = self.visual_text_attention(
            query=visual,
            key=text,
            value=text,
            key_padding_mask=text_padding,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        aligned, mel_lengths = self._resize_to_mel_rate(aligned, visual_lengths, target_mel_lengths)
        mel_padding = lengths_to_mask(mel_lengths, aligned.shape[1])

        speaker = self.speaker_projection(speakers).unsqueeze(1)
        decoded = self.decoder(aligned + speaker, padding_mask=mel_padding)
        mel = self.mel_head(decoded)

        output = {"mel": mel, "mel_lengths": mel_lengths}
        if return_attention:
            output["attention"] = attention
        return output

    def _resize_to_mel_rate(
        self,
        aligned: torch.Tensor,
        visual_lengths: torch.Tensor,
        target_mel_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_mel_lengths is not None:
            mel_lengths = target_mel_lengths.to(device=aligned.device, dtype=torch.long)
        else:
            mel_lengths = torch.clamp(
                torch.round(visual_lengths.float() * self.mel_frames_per_video_frame).long(),
                min=1,
            ).to(aligned.device)

        max_mel_len = int(mel_lengths.max().item())
        resized = aligned.new_zeros((aligned.shape[0], max_mel_len, aligned.shape[2]))

        for index in range(aligned.shape[0]):
            source_len = max(1, int(visual_lengths[index].item()))
            target_len = max(1, int(mel_lengths[index].item()))
            source = aligned[index, :source_len].transpose(0, 1).unsqueeze(0)
            resized_source = torch.nn.functional.interpolate(
                source,
                size=target_len,
                mode="linear",
                align_corners=False,
            )
            resized[index, :target_len] = resized_source.squeeze(0).transpose(0, 1)

        return resized, mel_lengths
