from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMelLoss(nn.Module):
    def __init__(
        self,
        mel_mean: torch.Tensor | None = None,
        mel_std: torch.Tensor | None = None,
        lambda_mel: float = 1.0,
        lambda_delta: float = 0.25,
        lambda_delta2: float = 0.05,
        lambda_energy: float = 0.05,
        lambda_mfcc: float = 0.0,
        lambda_flux: float = 0.0,
        lambda_voicing: float = 0.0,
        n_mfcc: int = 20,
        shift_window: int = 0,
    ):
        super().__init__()
        self.lambda_mel = float(lambda_mel)
        self.lambda_delta = float(lambda_delta)
        self.lambda_delta2 = float(lambda_delta2)
        self.lambda_energy = float(lambda_energy)
        self.lambda_mfcc = float(lambda_mfcc)
        self.lambda_flux = float(lambda_flux)
        self.lambda_voicing = float(lambda_voicing)
        self.n_mfcc = int(n_mfcc)
        self.shift_window = int(shift_window)
        self.register_buffer("mfcc_basis", torch.empty(0), persistent=False)
        if mel_mean is not None and mel_std is not None:
            self.register_buffer("mel_mean", mel_mean.float().view(1, 1, -1))
            self.register_buffer("mel_std", mel_std.float().view(1, 1, -1).clamp_min(1e-4))
        else:
            self.mel_mean = None
            self.mel_std = None

    def set_shift_window(self, shift_window: int) -> None:
        self.shift_window = int(max(0, shift_window))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mel_mean is None or self.mel_std is None:
            return x
        return (x - self.mel_mean.to(x.device, x.dtype)) / self.mel_std.to(x.device, x.dtype)

    @staticmethod
    def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(pred.device, dtype=pred.dtype).unsqueeze(-1)
        denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
        return ((pred - target).abs() * mask).sum() / denom

    def _shifted_l1(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.shift_window <= 0 or pred.shape[1] <= 2:
            return self._masked_l1(pred, target, mask)
        losses = []
        max_shift = min(self.shift_window, pred.shape[1] - 1)
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

    def _mfcc_basis(self, n_mels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        n_mfcc = max(1, min(self.n_mfcc, n_mels))
        if (
            self.mfcc_basis.numel() != n_mels * n_mfcc
            or self.mfcc_basis.device != device
            or self.mfcc_basis.dtype != dtype
        ):
            n = torch.arange(n_mels, dtype=dtype, device=device).unsqueeze(1)
            k = torch.arange(n_mfcc, dtype=dtype, device=device).unsqueeze(0)
            basis = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)
            basis[:, 0] *= math.sqrt(1.0 / float(n_mels))
            if n_mfcc > 1:
                basis[:, 1:] *= math.sqrt(2.0 / float(n_mels))
            self.mfcc_basis = basis
        return self.mfcc_basis

    def _mfcc(self, x: torch.Tensor) -> torch.Tensor:
        basis = self._mfcc_basis(x.shape[-1], x.dtype, x.device)
        return x.float().matmul(basis.float())

    @staticmethod
    def _spectral_flux(x: torch.Tensor) -> torch.Tensor:
        d = x[:, 1:] - x[:, :-1]
        return d.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()

    @staticmethod
    def _masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(logits.device, dtype=logits.dtype).unsqueeze(-1)
        denom = mask_f.sum().clamp_min(1.0)
        return (F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none") * mask_f).sum() / denom

    @staticmethod
    def _voicing_target(target_energy: torch.Tensor, mel_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = mel_mask.to(target_energy.device, target_energy.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (target_energy * mask).sum(dim=1, keepdim=True) / denom
        var = ((target_energy - mean).pow(2) * mask).sum(dim=1, keepdim=True) / denom
        std = var.clamp_min(1e-6).sqrt()
        threshold = mean - 0.35 * std
        return (target_energy > threshold).float(), threshold, std.clamp_min(0.25)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mel_mask: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        mel_mask = mel_mask.to(pred.device, dtype=torch.bool)
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)

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
            pred_energy = self._energy(pred)
            target_energy = self._energy(target)
            loss = loss + self.lambda_energy * self._masked_l1(pred_energy, target_energy, mel_mask)
        else:
            pred_energy = target_energy = None
        if self.lambda_mfcc:
            loss = loss + self.lambda_mfcc * self._masked_l1(self._mfcc(pred), self._mfcc(target), mel_mask)
        if self.lambda_flux and pred.shape[1] > 1:
            d_mask = mel_mask[:, 1:] & mel_mask[:, :-1]
            loss = loss + self.lambda_flux * self._masked_l1(self._spectral_flux(pred), self._spectral_flux(target), d_mask)
        if self.lambda_voicing:
            if pred_energy is None or target_energy is None:
                pred_energy = self._energy(pred)
                target_energy = self._energy(target)
            voiced, threshold, scale = self._voicing_target(target_energy, mel_mask)
            logits = (pred_energy - threshold) / scale
            loss = loss + self.lambda_voicing * self._masked_bce_with_logits(logits, voiced, mel_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite mel loss: {float(loss.detach().cpu())}")
        return loss
