"""Probe 04: sample-id latent upper bound on FullFrame_test."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from probe_01_shift_tolerant_loss import (
    PROJECT_ROOT,
    VNLipDatasetV2,
    add_common_args,
    compute_mel_stats,
    device_from_args,
    init_decoder_bias,
    make_criterion,
    make_dataset,
    maybe_wrap_model,
    resolve_path,
    safe_text,
    save_history,
    seed_everything,
    split_files,
    train_main,
)


class SampleIdDataset(torch.utils.data.Dataset):
    def __init__(self, base: VNLipDatasetV2):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        _, _, target, path = self.base[idx]
        return torch.tensor(idx, dtype=torch.long), target, path


def collate_id(batch):
    ids, targets, paths = zip(*batch)
    lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    tmax = int(lengths.max().item())
    padded = []
    for t in targets:
        if t.shape[0] < tmax:
            t = torch.cat([t, torch.zeros(tmax - t.shape[0], t.shape[1], dtype=t.dtype)], dim=0)
        padded.append(t)
    return torch.stack(ids), torch.stack(padded), lengths, list(paths)


class SampleIdLatentModel(nn.Module):
    def __init__(self, num_samples: int, latent_len: int, decoder: nn.Module):
        super().__init__()
        self.latent_len = latent_len
        self.table = nn.Embedding(num_samples, latent_len * 512)
        nn.init.normal_(self.table.weight, std=0.02)
        self.decoder = decoder

    def forward(self, ids: torch.Tensor, target_len: int):
        z = self.table(ids).view(ids.shape[0], self.latent_len, 512)
        return self.decoder(z, target_len=target_len)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = device_from_args(args)
    train_files, val_files, _ = split_files(args)
    train_base = make_dataset(args, train_files, random_crop=False)
    val_base = make_dataset(args, val_files, random_crop=False) if val_files else None
    mel_mean, mel_std = compute_mel_stats(train_base, args.mel_stats_max_samples)
    criterion = make_criterion(args, device, mel_mean, mel_std)
    train_loader = DataLoader(SampleIdDataset(train_base), batch_size=args.batch_size, shuffle=True, collate_fn=collate_id)
    val_loader = DataLoader(SampleIdDataset(val_base), batch_size=args.batch_size, shuffle=False, collate_fn=collate_id) if val_base else None
    decoder = train_main.MelTemporalUpsampleDecoder(
        train_main.build_base_decoder(args.decoder_type, hidden_dim=args.decoder_hidden_dim, num_layers=args.decoder_num_layers),
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    init_decoder_bias(decoder, mel_mean)
    model = SampleIdLatentModel(len(train_base), args.latent_len, decoder).to(device)
    model = maybe_wrap_model(model, args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[probe] sample_id_embedding train_files={len(train_base)} latent_len={args.latent_len}")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for ids, target, lengths, _ in train_loader:
            ids, target, lengths = ids.to(device), target.to(device), lengths.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(ids, target.shape[1])
            loss = criterion(pred, target, lengths)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            count += 1
        val = None
        if val_loader is not None:
            # Val ids are local to val set, so this upper bound reports train memorization only.
            val = None
        row = {"epoch": epoch, "train": total / max(1, count), "val": val}
        history.append(row)
        print(f"[epoch {epoch:04d}] train={row['train']:.6f}")
    save_history(args, "04_sample_id_embedding", history, {"note": "Upper bound checks dataset memorization on train split; val ids are intentionally not comparable."})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 04: sample-id latent table upper bound.")
    add_common_args(parser)
    parser.add_argument("--latent-len", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
