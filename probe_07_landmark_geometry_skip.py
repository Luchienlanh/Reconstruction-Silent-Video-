"""Probe 07: feed explicit landmark geometry skip features into the decoder."""

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


def geometry_features(landmarks: torch.Tensor) -> torch.Tensor:
    xy = landmarks[..., :2].float()
    center = xy.mean(dim=2, keepdim=True)
    xy = xy - center
    scale = xy.pow(2).sum(dim=-1).sqrt().amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
    xy = xy / scale
    width = (xy[..., 0].amax(dim=2) - xy[..., 0].amin(dim=2)).unsqueeze(-1)
    height = (xy[..., 1].amax(dim=2) - xy[..., 1].amin(dim=2)).unsqueeze(-1)
    area = width * height
    aspect = height / width.clamp_min(1e-4)
    d_height = torch.cat([torch.zeros_like(height[:, :1]), height[:, 1:] - height[:, :-1]], dim=1)
    d_area = torch.cat([torch.zeros_like(area[:, :1]), area[:, 1:] - area[:, :-1]], dim=1)
    return torch.cat([width, height, area, aspect, d_height, d_area], dim=-1)


class GeometrySkipWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.geom_proj = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128))
        self.fuse = nn.Sequential(nn.Linear(512 + 128, 512), nn.LayerNorm(512), nn.SiLU(), nn.Linear(512, 512))

    def forward(self, video, landmarks, target_len: int):
        z = self.base.encoder(video, landmarks)
        geom = geometry_features(landmarks)
        geom = F.interpolate(geom.transpose(1, 2), size=z.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        z = self.fuse(torch.cat([z, self.geom_proj(geom)], dim=-1))
        return {"pred": self.base.decoder(z, target_len=target_len), "z": z}


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_ds, _, train_loader, val_loader, num_points = make_loaders(args)
    mel_mean, mel_std = compute_mel_stats(train_ds, args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    base = build_reconstruction_model(args, device, num_points)
    init_decoder_bias(base.decoder, mel_mean)
    model = GeometrySkipWrapper(base).to(device)
    print("[probe] landmark_geometry_skip")
    history = train_probe(args, model, train_loader, val_loader, criterion, device, mel_mean=mel_mean)
    save_history(args, "07_landmark_geometry_skip", history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 07: landmark geometry skip features.")
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
