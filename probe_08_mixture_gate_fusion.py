"""Probe 08: explicit mixture gate fusion over landmark/video/joint latents."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from probe_01_shift_tolerant_loss import (
    LandmarkMotionEncoder,
    add_common_args,
    build_encoder,
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


class MixtureGateFusion(nn.Module):
    def __init__(self, encoder_type: str, num_landmark_points: int, decoder: nn.Module):
        super().__init__()
        self.visual = build_encoder(encoder_type)
        self.landmark = LandmarkMotionEncoder(num_landmark_points, out_dim=512, dropout=0.0)
        self.joint = nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512), nn.SiLU(), nn.Linear(512, 512))
        self.gate = nn.Sequential(nn.Linear(1024, 256), nn.SiLU(), nn.Linear(256, 3))
        self.decoder = decoder
        self.last_gate = None

    def forward(self, video, landmarks, target_len: int):
        zv = self.visual(video)
        zl = self.landmark(landmarks)
        if zv.shape[1] != zl.shape[1]:
            zv = F.interpolate(zv.transpose(1, 2), size=zl.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        zj = self.joint(torch.cat([zl, zv], dim=-1))
        gate = torch.softmax(self.gate(torch.cat([zl, zv], dim=-1)), dim=-1)
        self.last_gate = gate.detach()
        z = gate[..., 0:1] * zl + gate[..., 1:2] * zv + gate[..., 2:3] * zj
        return {"pred": self.decoder(z, target_len=target_len), "z": z, "gate": gate}


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_ds, _, train_loader, val_loader, num_points = make_loaders(args)
    mel_mean, mel_std = compute_mel_stats(train_ds, args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    decoder = train_main.MelTemporalUpsampleDecoder(
        train_main.build_base_decoder(args.decoder_type, hidden_dim=args.decoder_hidden_dim, num_layers=args.decoder_num_layers),
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    init_decoder_bias(decoder, mel_mean)
    model = MixtureGateFusion(args.encoder_type, num_points, decoder).to(device)

    def after_epoch(m, epoch):
        if m.last_gate is None:
            return {}
        g = m.last_gate.mean(dim=(0, 1)).detach().cpu().tolist()
        print(f"[gate] lm={g[0]:.3f} video={g[1]:.3f} joint={g[2]:.3f}")
        return {"gate_lm": g[0], "gate_video": g[1], "gate_joint": g[2]}

    print("[probe] mixture_gate_fusion")
    history = train_probe(args, model, train_loader, val_loader, criterion, device, after_epoch=after_epoch, mel_mean=mel_mean)
    save_history(args, "08_mixture_gate_fusion", history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 08: mixture gate fusion.")
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
