from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1_mel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    max_len = min(pred.shape[1], target.shape[1])
    pred = pred[:, :max_len]
    target = target[:, :max_len]
    lengths = torch.minimum(pred_lengths, target_lengths).clamp(max=max_len)

    frame_positions = torch.arange(max_len, device=pred.device).unsqueeze(0)
    mask = frame_positions < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)

    loss = F.l1_loss(pred, target, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp_min(1) * pred.shape[-1]
    return loss.sum() / denom
