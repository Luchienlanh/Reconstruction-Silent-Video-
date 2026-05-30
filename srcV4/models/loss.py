from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionMelSpectralLoss(nn.Module):
    """
    Multi-Resolution Mel Spectral Loss (MR-MSL) operating directly on Mel Spectrograms.
    Enforces time-frequency consistency across multiple temporal scales using multi-scale 1D poolings.
    This helps the model capture both fine-grained temporal transitions (phoneme boundaries)
    and coarse spectral envelopes (vowels, formants).
    """
    def __init__(self, scales: list[int] = [2, 4, 8]):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = pred.new_tensor(0.0)
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
        
        # Transpose to (B, channels, T) for avg_pool1d
        p_t = pred.transpose(1, 2)
        t_t = target.transpose(1, 2)
        m_t = mask_f.transpose(1, 2)
        
        for scale in self.scales:
            p_pooled = F.avg_pool1d(p_t, kernel_size=scale, stride=1, padding=scale // 2)
            t_pooled = F.avg_pool1d(t_t, kernel_size=scale, stride=1, padding=scale // 2)
            m_pooled = F.avg_pool1d(m_t, kernel_size=scale, stride=1, padding=scale // 2)
            
            # Trim potential trailing frames from padding
            T = pred.shape[1]
            p_pooled = p_pooled[..., :T]
            t_pooled = t_pooled[..., :T]
            m_pooled = m_pooled[..., :T]
            
            diff = (p_pooled - t_pooled).abs()
            diff_sq = (p_pooled - t_pooled).pow(2)
            
            denom = (m_pooled.sum() * pred.shape[-1]).clamp_min(1.0)
            l1 = (diff * m_pooled).sum() / denom
            l2 = (diff_sq * m_pooled).sum() / denom
            loss = loss + l1 + 0.5 * l2
            
        return loss / len(self.scales)


class V4MelLoss(nn.Module):
    """
    Advanced V4 Mel Loss function combining:
    1. Masked L1 loss on Mel spectrogram
    2. Delta & Delta-Delta spectral transition losses
    3. Spectral energy preservation loss
    4. Multi-Resolution Mel Spectral Loss (frequency-domain temporal pooling supervision)
    """
    def __init__(
        self,
        mel_mean: torch.Tensor | None = None,
        mel_std: torch.Tensor | None = None,
        lambda_mel: float = 1.0,
        lambda_delta: float = 0.15,
        lambda_delta2: float = 0.03,
        lambda_energy: float = 0.02,
        lambda_mr_spectral: float = 0.25,
        shift_window: int = 0,
    ):
        super().__init__()
        self.lambda_mel = float(lambda_mel)
        self.lambda_delta = float(lambda_delta)
        self.lambda_delta2 = float(lambda_delta2)
        self.lambda_energy = float(lambda_energy)
        self.lambda_mr_spectral = float(lambda_mr_spectral)
        self.shift_window = int(shift_window)
        
        if mel_mean is not None and mel_std is not None:
            self.register_buffer("mel_mean", mel_mean.float().view(1, 1, -1))
            self.register_buffer("mel_std", mel_std.float().view(1, 1, -1).clamp_min(1e-4))
        else:
            self.mel_mean = None
            self.mel_std = None
            
        self.mr_spectral_loss = MultiResolutionMelSpectralLoss(scales=[2, 4, 8, 16])

    def set_shift_window(self, shift_window: int) -> None:
        self.shift_window = int(max(0, shift_window))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mel_mean is None or self.mel_std is None:
            return x
        mean = self.mel_mean.to(x.device, x.dtype)
        std = self.mel_std.to(x.device, x.dtype)
        return (x - mean) / std

    @staticmethod
    def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(pred.device, dtype=pred.dtype).unsqueeze(-1)
        denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
        return ((pred - target).abs() * mask).sum() / denom

    def _shifted_l1(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.shift_window <= 0 or pred.shape[1] <= 2:
            return self._masked_l1(pred, target, mask)
        max_shift = min(self.shift_window, pred.shape[1] - 1)
        losses = []
        for shift in range(-max_shift, max_shift + 1):
            if shift < 0:
                p = pred[:, :shift]
                t = target[:, -shift:]
                m = mask[:, :shift] & mask[:, -shift:]
            elif shift > 0:
                p = pred[:, shift:]
                t = target[:, :-shift]
                m = mask[:, shift:] & mask[:, :-shift]
            else:
                p, t, m = pred, target, mask
            losses.append(self._masked_l1(p, t, m))
        return torch.stack(losses).min()

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]

    @staticmethod
    def _energy(x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=-1, keepdim=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mel_mask: torch.Tensor) -> torch.Tensor:
        pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        target = torch.nan_to_num(target.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        mel_mask = mel_mask.to(pred.device, dtype=torch.bool)
        
        pred_n = self.normalize(pred)
        target_n = self.normalize(target)

        loss = pred.new_tensor(0.0)
        
        if self.lambda_mel:
            loss = loss + self.lambda_mel * self._shifted_l1(pred_n, target_n, mel_mask)
            
        if self.lambda_delta and pred.shape[1] > 1:
            d_mask = mel_mask[:, 1:] & mel_mask[:, :-1]
            loss = loss + self.lambda_delta * self._masked_l1(self._delta(pred_n), self._delta(target_n), d_mask)
            
        if self.lambda_delta2 and pred.shape[1] > 2:
            d2_mask = mel_mask[:, 2:] & mel_mask[:, 1:-1] & mel_mask[:, :-2]
            loss = loss + self.lambda_delta2 * self._masked_l1(
                self._delta(self._delta(pred_n)),
                self._delta(self._delta(target_n)),
                d2_mask,
            )
            
        if self.lambda_energy:
            loss = loss + self.lambda_energy * self._masked_l1(self._energy(pred), self._energy(target), mel_mask)
            
        if self.lambda_mr_spectral:
            loss = loss + self.lambda_mr_spectral * self.mr_spectral_loss(pred_n, target_n, mel_mask)
            
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite mel loss: {float(loss.detach().cpu())}")
            
        return loss


@torch.no_grad()
def masked_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    mask = mask.to(pred.device, dtype=torch.bool)
    p = pred.detach().float()[mask]
    t = target.detach().float()[mask]
    if p.numel() == 0 or t.numel() == 0:
        return {}

    def delta_abs(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2:
            return x.new_tensor(0.0)
        return (x[1:] - x[:-1]).abs().mean()

    p_energy = torch.logsumexp(p, dim=-1)
    t_energy = torch.logsumexp(t, dim=-1)
    
    return {
        "pred_std": float(p.std(unbiased=False).cpu()),
        "target_std": float(t.std(unbiased=False).cpu()),
        "std_ratio": float((p.std(unbiased=False) / t.std(unbiased=False).clamp_min(1e-6)).cpu()),
        "pred_delta": float(delta_abs(p).cpu()),
        "target_delta": float(delta_abs(t).cpu()),
        "delta_ratio": float((delta_abs(p) / delta_abs(t).clamp_min(1e-6)).cpu()),
        "energy_ratio": float((p_energy.std(unbiased=False) / t_energy.std(unbiased=False).clamp_min(1e-6)).cpu()),
    }
