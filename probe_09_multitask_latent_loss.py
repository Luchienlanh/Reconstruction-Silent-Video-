"""Probe 09: multi-task latent heads for energy, delta-energy, and voiced mask."""

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


class MultiTaskWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.energy = nn.Linear(512, 1)
        self.delta_energy = nn.Linear(512, 1)
        self.voiced = nn.Linear(512, 1)

    def forward(self, video, landmarks, target_len: int):
        out = self.base(video, landmarks, target_len)
        z = F.interpolate(out["z"].transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        out["energy"] = self.energy(z)
        out["delta_energy"] = self.delta_energy(z)
        out["voiced"] = self.voiced(z)
        return out


def multitask_loss_fn(model, batch, criterion, device, args):
    video, landmarks, target, lengths, _ = unpack_batch(batch, device, args)
    out = model(video, landmarks, target.shape[1])
    mel = criterion(out["pred"], target, lengths)
    energy = torch.logsumexp(target.float(), dim=-1, keepdim=True)
    d_energy = torch.cat([torch.zeros_like(energy[:, :1]), energy[:, 1:] - energy[:, :-1]], dim=1)
    voiced = (energy > energy.mean(dim=1, keepdim=True)).float()
    aux = (
        F.l1_loss(out["energy"].float(), energy)
        + F.l1_loss(out["delta_energy"].float(), d_energy)
        + F.binary_cross_entropy_with_logits(out["voiced"].float(), voiced)
    )
    return mel + args.aux_weight * aux, out


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_ds, _, train_loader, val_loader, num_points = make_loaders(args)
    mel_mean, mel_std = compute_mel_stats(train_ds, args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    base = build_reconstruction_model(args, device, num_points)
    init_decoder_bias(base.decoder, mel_mean)
    model = MultiTaskWrapper(base).to(device)
    print(f"[probe] multitask_latent aux_weight={args.aux_weight}")
    history = train_probe(args, model, train_loader, val_loader, criterion, device, loss_fn=multitask_loss_fn, mel_mean=mel_mean)
    save_history(args, "09_multitask_latent_loss", history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 09: multi-task latent losses.")
    add_common_args(parser)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
