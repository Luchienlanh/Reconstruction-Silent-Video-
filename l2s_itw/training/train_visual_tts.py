from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from l2s_itw.config import apply_overrides, load_config, save_config
from l2s_itw.data import VisualTTSDataset, collate_visual_tts
from l2s_itw.models import VisualTTS
from l2s_itw.text import CharTokenizer
from l2s_itw.training.losses import masked_l1_mel_loss
from l2s_itw.utils import ensure_dir, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean Visual TTS model.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted.key=value.")
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def make_loader(
    manifest: str,
    tokenizer: CharTokenizer,
    config: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    limit = int(config["training"].get("limit_samples", 0))
    dataset = VisualTTSDataset(manifest, tokenizer, config, limit=limit)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_visual_tts,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: VisualTTS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    log_every = int(config["training"].get("log_every", 20))
    amp_enabled = bool(config["training"].get("amp", False)) and device.type == "cuda"
    grad_clip = float(config["training"].get("grad_clip", 0.0))

    prefix = "train" if is_train else "val"
    iterator = tqdm(loader, desc=f"{prefix} epoch {epoch}", leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = move_batch(batch, device)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(
                    visuals=batch["visuals"],
                    visual_lengths=batch["visual_lengths"],
                    tokens=batch["tokens"],
                    text_lengths=batch["text_lengths"],
                    speakers=batch["speakers"],
                    target_mel_lengths=batch["mel_lengths"],
                )
                loss = masked_l1_mel_loss(
                    pred=output["mel"],
                    target=batch["mels"],
                    pred_lengths=output["mel_lengths"],
                    target_lengths=batch["mel_lengths"],
                )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = int(batch["visuals"].shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
        if step % log_every == 0:
            iterator.set_postfix(loss=total_loss / max(total_items, 1))

    return total_loss / max(total_items, 1)


def save_checkpoint(
    path: Path,
    model: VisualTTS,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    config: dict[str, Any],
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokenizer_symbols": config["text"]["symbols"],
            "tokenizer_lowercase": bool(config["text"].get("lowercase", True)),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
            "vocab_size": tokenizer.vocab_size,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)

    train_manifest = config["data"].get("train_manifest", "")
    val_manifest = config["data"].get("val_manifest", "")
    if not train_manifest or not val_manifest:
        raise ValueError("Set data.train_manifest and data.val_manifest in config or with --set.")

    seed_everything(int(config.get("seed", 1234)))
    device = resolve_device(str(config["training"].get("device", "cuda")))
    output_dir = ensure_dir(config["training"]["output_dir"])
    save_config(config, output_dir / "config.json")

    tokenizer = CharTokenizer(
        symbols=config["text"]["symbols"],
        lowercase=bool(config["text"].get("lowercase", True)),
    )
    train_loader = make_loader(train_manifest, tokenizer, config, shuffle=True)
    val_loader = make_loader(val_manifest, tokenizer, config, shuffle=False)

    model = VisualTTS(config, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"].get("amp", False)) and device.type == "cuda")

    best_val = float("inf")
    epochs = int(config["training"]["epochs"])
    start = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, config, device, epoch)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, None, config, device, epoch)

        latest_path = output_dir / "latest_model.pth"
        save_checkpoint(latest_path, model, optimizer, tokenizer, config, epoch, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, tokenizer, config, epoch, val_loss)

        elapsed = time.time() - start
        print(
            f"epoch={epoch} train_l1={train_loss:.5f} val_l1={val_loss:.5f} "
            f"best_val={best_val:.5f} elapsed={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
