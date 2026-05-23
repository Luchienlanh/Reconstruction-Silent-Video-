"""Probe 06: align visual/landmark latent to a mel teacher latent."""

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


class MelTeacherWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.mel_teacher = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
        )

    def forward(self, video, landmarks, target_len: int, target=None):
        out = self.base(video, landmarks, target_len)
        if target is not None:
            z_mel = self.mel_teacher(target.transpose(1, 2)).transpose(1, 2)
            z_mel = F.interpolate(z_mel.transpose(1, 2), size=out["z"].shape[1], mode="linear", align_corners=False).transpose(1, 2)
            out["z_mel"] = z_mel
        return out


def teacher_loss_fn(model, batch, criterion, device, args):
    video, landmarks, target, lengths, _ = unpack_batch(batch, device, args)
    out = model(video, landmarks, target.shape[1], target=target)
    mel_loss = criterion(out["pred"], target, lengths)
    z = F.normalize(out["z"].float(), dim=-1)
    z_mel = F.normalize(out["z_mel"].float().detach() if args.detach_teacher else out["z_mel"].float(), dim=-1)
    align = 1.0 - (z * z_mel).sum(dim=-1).mean()
    return mel_loss + args.align_weight * align, out


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_ds, _, train_loader, val_loader, num_points = make_loaders(args)
    mel_mean, mel_std = compute_mel_stats(train_ds, args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    base = build_reconstruction_model(args, device, num_points)
    init_decoder_bias(base.decoder, mel_mean)
    model = MelTeacherWrapper(base).to(device)
    print(f"[probe] mel_teacher_alignment align_weight={args.align_weight}")
    history = train_probe(args, model, train_loader, val_loader, criterion, device, loss_fn=teacher_loss_fn, mel_mean=mel_mean)
    save_history(args, "06_mel_teacher_alignment", history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 06: mel teacher latent alignment.")
    add_common_args(parser)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--detach-teacher", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
