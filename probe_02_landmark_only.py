"""Probe 02: landmark-only baseline on FullFrame_test."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from probe_01_shift_tolerant_loss import (
    LandmarkMotionEncoder,
    add_common_args,
    compute_mel_stats,
    device_from_args,
    init_decoder_bias,
    make_criterion,
    make_loaders,
    save_history,
    seed_everything,
    train_main,
    train_probe,
)


class LandmarkOnlyModel(nn.Module):
    def __init__(self, num_landmark_points: int, decoder: nn.Module):
        super().__init__()
        self.landmark_encoder = LandmarkMotionEncoder(num_landmark_points, out_dim=512, dropout=0.0)
        self.decoder = decoder

    def forward(self, video, landmarks, target_len: int):
        z = self.landmark_encoder(landmarks)
        return {"pred": self.decoder(z, target_len=target_len), "z": z}


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_ds, _, train_loader, val_loader, num_points = make_loaders(args)
    mel_mean, mel_std = compute_mel_stats(train_ds, args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    decoder = train_main.MelTemporalUpsampleDecoder(
        train_main.build_base_decoder(
            args.decoder_type,
            hidden_dim=args.decoder_hidden_dim,
            num_layers=args.decoder_num_layers,
            dropout=args.decoder_dropout,
        ),
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    init_decoder_bias(decoder, mel_mean)
    model = LandmarkOnlyModel(num_points, decoder).to(device)
    print(f"[probe] landmark_only train_files={len(train_ds)}")
    history = train_probe(args, model, train_loader, val_loader, criterion, device, mel_mean=mel_mean)
    save_history(args, "02_landmark_only", history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 02: landmark-only -> decoder.")
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
