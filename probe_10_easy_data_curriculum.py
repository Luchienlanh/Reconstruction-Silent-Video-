"""Probe 10: easy-sample data curriculum on FullFrame_test."""

from __future__ import annotations

import argparse
import math
import torch
from torch.utils.data import DataLoader

from probe_01_shift_tolerant_loss import (
    add_common_args,
    build_reconstruction_model,
    collate_pad_v2,
    compute_mel_stats,
    device_from_args,
    init_decoder_bias,
    make_criterion,
    make_dataset,
    save_history,
    seed_everything,
    split_files,
    train_probe,
)


def easy_score(item) -> float:
    _, landmarks, target, _ = item
    motion = landmarks[..., 2:4].abs().mean().item() if landmarks.shape[-1] >= 4 else 0.0
    energy_std = torch.logsumexp(target.float(), dim=-1).std(unbiased=False).item()
    length_penalty = 0.001 * float(target.shape[0])
    return motion + 0.25 * energy_std - length_penalty


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_files, val_files, num_points = split_files(args)
    full_train = make_dataset(args, train_files, random_crop=True)
    scored = []
    for idx in range(len(full_train)):
        scored.append((easy_score(full_train[idx]), full_train.files[idx]))
    scored.sort(reverse=True)
    sorted_files = [name for _, name in scored]

    # Start with the easiest fraction, then expand every epoch.
    start_count = min(len(sorted_files), max(1, int(math.ceil(len(sorted_files) * args.start_fraction))))
    train_ds = make_dataset(args, sorted_files[:start_count], random_crop=True)
    val_ds = make_dataset(args, val_files, random_crop=False) if val_files else None
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size or args.batch_size, shuffle=False, collate_fn=collate_pad_v2) if val_ds else None
    mel_mean, mel_std = compute_mel_stats(make_dataset(args, sorted_files, random_crop=False), args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    model = build_reconstruction_model(args, device, num_points)
    init_decoder_bias(model.decoder, mel_mean)
    history = []
    print(f"[probe] easy_curriculum total_train={len(sorted_files)} start={start_count}")

    for epoch in range(1, args.epochs + 1):
        fraction = args.start_fraction + (1.0 - args.start_fraction) * (epoch - 1) / max(1, args.epochs - 1)
        active_count = min(len(sorted_files), max(start_count, int(math.ceil(len(sorted_files) * fraction))))
        train_ds.files = sorted_files[:active_count]
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad_v2)
        args_one = argparse.Namespace(**vars(args))
        args_one.epochs = 1
        sub_history = train_probe(args_one, model, train_loader, val_loader, criterion, device, mel_mean=mel_mean)
        row = dict(sub_history[-1])
        row["epoch"] = epoch
        row["active_files"] = active_count
        history.append(row)
        print(f"[curriculum] epoch={epoch:04d} active_files={active_count}")
    save_history(args, "10_easy_data_curriculum", history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 10: easy data curriculum.")
    add_common_args(parser)
    parser.add_argument("--start-fraction", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
