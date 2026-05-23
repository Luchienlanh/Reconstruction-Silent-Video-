"""Probe 05: supervised auxiliary mel-summary losses on FullFrame_test."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from probe_01_shift_tolerant_loss import (
    add_common_args,
    build_reconstruction_model,
    compute_mel_stats,
    device_from_args,
    init_decoder_bias,
    make_criterion,
    make_loaders,
    save_history,
    seed_everything,
    train_probe,
    unpack_batch,
)


class AuxWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.energy_head = nn.Linear(512, 1)
        self.low_head = nn.Linear(512, 1)
        self.high_head = nn.Linear(512, 1)

    def forward(self, video, landmarks, target_len: int):
        out = self.base(video, landmarks, target_len)
        z_up = F.interpolate(out["z"].transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        out["energy"] = self.energy_head(z_up)
        out["low"] = self.low_head(z_up)
        out["high"] = self.high_head(z_up)
        return out


def aux_loss_fn(model, batch, criterion, device, args):
    video, landmarks, target, lengths, _ = unpack_batch(batch, device, args)
    out = model(video, landmarks, target.shape[1])
    mel_loss = criterion(out["pred"], target, lengths)
    energy = torch.logsumexp(target.float(), dim=-1, keepdim=True)
    low = torch.logsumexp(target[..., :20].float(), dim=-1, keepdim=True)
    high = torch.logsumexp(target[..., 20:].float(), dim=-1, keepdim=True)
    aux = F.l1_loss(out["energy"].float(), energy) + F.l1_loss(out["low"].float(), low) + F.l1_loss(out["high"].float(), high)
    return mel_loss + args.aux_weight * aux, out


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_ds, _, train_loader, val_loader, num_points = make_loaders(args)
    mel_mean, mel_std = compute_mel_stats(train_ds, args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    base = build_reconstruction_model(args, device, num_points)
    init_decoder_bias(base.decoder, mel_mean)
    model = AuxWrapper(base).to(device)
    print(f"[probe] supervised_aux aux_weight={args.aux_weight}")
    history = train_probe(args, model, train_loader, val_loader, criterion, device, loss_fn=aux_loss_fn, mel_mean=mel_mean)
    save_history(args, "05_supervised_aux_pretrain", history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 05: supervised auxiliary mel-summary losses.")
    add_common_args(parser)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
