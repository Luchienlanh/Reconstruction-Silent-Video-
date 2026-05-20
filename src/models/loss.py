import os
import gc
import glob
import math
import random
import warnings
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
try:
    import torchaudio
except ImportError:
    pass

class MelReconstructionLoss(nn.Module):
    def __init__(self, lambda_mel=1.0, lambda_delta=0.25, lambda_delta2=0.05, lambda_energy=0.05):
        super().__init__()
        self.lambda_mel = lambda_mel
        self.lambda_delta = lambda_delta
        self.lambda_delta2 = lambda_delta2
        self.lambda_energy = lambda_energy

    def _mask(self, x, lengths):
        T = x.shape[1]
        return (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)

    def _masked_l1(self, pred, target, lengths):
        mask = self._mask(pred, lengths)
        denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
        return ((pred - target).abs() * mask).sum() / denom

    def _delta(self, x):
        return x[:, 1:] - x[:, :-1]

    def _energy(self, mel):
        return torch.logsumexp(mel.float(), dim=-1, keepdim=True)

    def forward(self, pred, target, lengths):
        pred = pred.float()
        target = target.float()
        lengths = lengths.to(device=pred.device, dtype=torch.long).clamp(min=1, max=pred.shape[1])
        loss = pred.new_tensor(0.0)
        if self.lambda_mel:
            loss = loss + self.lambda_mel * self._masked_l1(pred, target, lengths)
        if self.lambda_delta and pred.shape[1] > 1:
            loss = loss + self.lambda_delta * self._masked_l1(self._delta(pred), self._delta(target), (lengths - 1).clamp_min(1))
        if self.lambda_delta2 and pred.shape[1] > 2:
            loss = loss + self.lambda_delta2 * self._masked_l1(self._delta(self._delta(pred)), self._delta(self._delta(target)), (lengths - 2).clamp_min(1))
        if self.lambda_energy:
            loss = loss + self.lambda_energy * self._masked_l1(self._energy(pred), self._energy(target), lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite mel loss: {float(loss.detach().cpu())}")
        return loss

class STFTLoss(nn.Module):
    """Single-resolution STFT loss computed in FP32 for numerical stability."""
    def __init__(self, fft_size=1024, hop_size=256, win_size=1024, eps=1e-5):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.eps = eps
        self.register_buffer('window', torch.hann_window(win_size))

    def stft(self, x):
        # x: (B, T_samples) -> complex STFT. Keep STFT in FP32; FP16 log-magnitude can underflow to -inf.
        x = x.float()
        return torch.stft(
            x, self.fft_size, self.hop_size, self.win_size,
            self.window.to(device=x.device, dtype=torch.float32), return_complex=True
        )

    def forward(self, pred, target):
        # pred, target: (B, T_samples)
        pred = pred.float()
        target = target.float()
        pred_stft = self.stft(pred)
        target_stft = self.stft(target)
        pred_mag = pred_stft.abs().clamp_min(self.eps)
        target_mag = target_stft.abs().clamp_min(self.eps)
        denom = torch.norm(target_mag, p='fro').clamp_min(self.eps)
        sc_loss = torch.norm(target_mag - pred_mag, p='fro') / denom
        log_mag_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
        loss = sc_loss + log_mag_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite STFTLoss: sc={float(sc_loss.detach().cpu())}, "
                f"log={float(log_mag_loss.detach().cpu())}"
            )
        return loss

class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss on full waveform tensors."""
    def __init__(self, fft_sizes=[256, 512, 1024], hop_sizes=[64, 128, 256], win_sizes=[256, 512, 1024]):
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft, hop, win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, pred, target):
        total = 0.0
        for stft_loss in self.stft_losses:
            total = total + stft_loss(pred, target)
        return total / len(self.stft_losses)

class CombinedAudioLoss(nn.Module):
    """MSE tr?n chunks + MR-STFT tr?n waveform, h? tr? padded batch b?ng lengths."""
    def __init__(self, lambda_mse=1.0, lambda_stft=1.0, hop=640):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_stft = lambda_stft
        self.hop = hop
        self.mr_stft = MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024],
            hop_sizes=[64, 128, 256],
            win_sizes=[256, 512, 1024]
        )

    def _masked_mse(self, pred, target, lengths):
        B, T, C = pred.shape
        mask = torch.arange(T, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(dtype=pred.dtype)
        denom = mask.sum().clamp_min(1.0) * C
        return ((pred - target).pow(2) * mask).sum() / denom

    def _stft_loss_by_length(self, pred, target, lengths):
        losses = []
        for i, L in enumerate(lengths.tolist()):
            pred_wave = pred[i, :L].reshape(1, -1)
            target_wave = target[i, :L].reshape(1, -1)
            # STFT c?n t?i thi?u win_size; crop train hi?n ?? d?i, nh?ng guard ?? tr?nh clip qu? ng?n.
            if pred_wave.shape[1] < 1024:
                pad = 1024 - pred_wave.shape[1]
                pred_wave = F.pad(pred_wave, (0, pad))
                target_wave = F.pad(target_wave, (0, pad))
            losses.append(self.mr_stft(pred_wave, target_wave))
        return torch.stack(losses).mean()

    def forward(self, pred, target, lengths=None):
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError(f"CombinedAudioLoss expects (B,T,640), got pred={tuple(pred.shape)}, target={tuple(target.shape)}")
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

        # Compute the loss in FP32 even when the model forward uses AMP/FP16.
        pred = pred.float()
        target = target.float()

        if lengths is None:
            lengths = torch.full((pred.shape[0],), pred.shape[1], device=pred.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=pred.device, dtype=torch.long).clamp(min=1, max=pred.shape[1])

        loss = pred.new_tensor(0.0)
        if self.lambda_mse != 0:
            loss = loss + self.lambda_mse * self._masked_mse(pred, target, lengths)
        if self.lambda_stft != 0:
            loss = loss + self.lambda_stft * self._stft_loss_by_length(pred, target, lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite CombinedAudioLoss: {float(loss.detach().cpu())}")
        return loss

def make_criterion(target_type, device):
    if target_type == "mel_hifigan":
        print("Loss function: MelReconstructionLoss (masked mel L1 + delta + energy)")
        return MelReconstructionLoss(lambda_mel=1.0, lambda_delta=0.25, lambda_delta2=0.05, lambda_energy=0.05).to(device)
    if target_type == "waveform":
        print("Loss function: CombinedAudioLoss (masked MSE + length-aware MR-STFT)")
        print("STFT resolutions: 256, 512, 1024")
        return CombinedAudioLoss(lambda_mse=1.0, lambda_stft=1.0).to(device)
    raise ValueError(f"Unsupported TARGET_TYPE={target_type}")

class PhaseAwareMelLoss(nn.Module):
    """
    Curriculum-aware mel loss with 4 training phases.

    Phase 1 - F0 (Pitch):
        Focus on low-frequency mel channels (0-20) that capture F0/pitch contour,
        plus first-order temporal delta to learn intonation dynamics.

    Phase 2 - Voice Tone (Timbre):
        Full mel reconstruction L1 to learn the overall spectral envelope.

    Phase 3 - Oscillation/Vibration:
        Emphasize first and second-order temporal deltas for fine detail.

    Phase 4 - Energy:
        Focus on log-energy envelope matching for volume dynamics.
    """

    def __init__(
        self,
        n_mels: int = 80,
        f0_mel_end: int = 20,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.f0_mel_end = f0_mel_end

        # Phase-specific lambda weights:
        # [lambda_mel, lambda_f0_mel, lambda_delta, lambda_delta2, lambda_energy]
        self.phase_configs = {
            1: {"lambda_mel": 0.3, "lambda_f0_mel": 1.0, "lambda_delta": 0.5,
                "lambda_delta2": 0.0, "lambda_energy": 0.0, "name": "F0 (Pitch)"},
            2: {"lambda_mel": 1.0, "lambda_f0_mel": 0.3, "lambda_delta": 0.2,
                "lambda_delta2": 0.05, "lambda_energy": 0.1, "name": "Voice Tone (Timbre)"},
            3: {"lambda_mel": 0.5, "lambda_f0_mel": 0.1, "lambda_delta": 0.8,
                "lambda_delta2": 0.5, "lambda_energy": 0.1, "name": "Oscillation"},
            4: {"lambda_mel": 0.3, "lambda_f0_mel": 0.1, "lambda_delta": 0.2,
                "lambda_delta2": 0.1, "lambda_energy": 1.0, "name": "Energy"},
        }

    def _mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return (
            torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).unsqueeze(-1).to(x.dtype)

    def _masked_l1(self, pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = self._mask(pred, lengths)
        denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
        return ((pred - target).abs() * mask).sum() / denom

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]

    def _energy(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(mel.float(), dim=-1, keepdim=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        phase: int = 2,
    ) -> torch.Tensor:
        """
        Compute phase-aware mel loss.

        Args:
            pred: (B, T_mel, n_mels)
            target: (B, T_mel, n_mels)
            lengths: (B,) valid mel lengths
            phase: training phase (1-4)
        """
        pred = pred.float()
        target = target.float()
        lengths = lengths.to(device=pred.device, dtype=torch.long).clamp(min=1, max=pred.shape[1])

        cfg = self.phase_configs.get(phase, self.phase_configs[2])
        loss = pred.new_tensor(0.0)

        # Full mel L1
        if cfg["lambda_mel"]:
            loss = loss + cfg["lambda_mel"] * self._masked_l1(pred, target, lengths)

        # F0 low-frequency mel channels
        if cfg["lambda_f0_mel"]:
            f0_end = min(self.f0_mel_end, pred.shape[-1])
            loss = loss + cfg["lambda_f0_mel"] * self._masked_l1(
                pred[..., :f0_end], target[..., :f0_end], lengths
            )

        # First-order temporal delta
        if cfg["lambda_delta"] and pred.shape[1] > 1:
            loss = loss + cfg["lambda_delta"] * self._masked_l1(
                self._delta(pred), self._delta(target),
                (lengths - 1).clamp_min(1)
            )

        # Second-order temporal delta
        if cfg["lambda_delta2"] and pred.shape[1] > 2:
            loss = loss + cfg["lambda_delta2"] * self._masked_l1(
                self._delta(self._delta(pred)),
                self._delta(self._delta(target)),
                (lengths - 2).clamp_min(1)
            )

        # Energy envelope
        if cfg["lambda_energy"]:
            loss = loss + cfg["lambda_energy"] * self._masked_l1(
                self._energy(pred), self._energy(target), lengths
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite PhaseAwareMelLoss (phase={phase}): {float(loss.detach().cpu())}"
            )

        return loss

