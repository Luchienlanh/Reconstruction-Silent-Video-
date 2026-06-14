from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from l2t_arch.config import apply_overrides, load_config, save_config
from l2t_arch.data import ArchDataset, collate_arch
from l2t_arch.models import build_model
from l2t_arch.text import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-VTP lip-to-text architecture models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_loader(manifest: str, tokenizer: CharTokenizer, config: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = ArchDataset(manifest, tokenizer, limit=int(config["training"].get("limit_samples", 0)))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_arch,
        pin_memory=torch.cuda.is_available(),
    )


def ce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def ctc_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], blank_id: int) -> torch.Tensor:
    log_probs = output["ctc_logits"].log_softmax(dim=-1).transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=batch["ctc_targets"],
        input_lengths=output["ctc_lengths"].detach().cpu(),
        target_lengths=batch["ctc_target_lengths"].detach().cpu(),
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def compute_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], config: dict[str, Any], tokenizer: CharTokenizer) -> torch.Tensor:
    kind = str(config["model"]["type"])
    if kind in {"text_only", "dual_path", "visual_plif_seq2seq", "dual_plif_monotonic"}:
        loss = ce_loss(output["logits"], batch["gt_out"], tokenizer.blank_id)
        ctc_weight = float(config["training"].get("ctc_weight", 0.0))
        if ctc_weight > 0 and "ctc_logits" in output:
            loss = loss + ctc_weight * ctc_loss(output, batch, tokenizer.blank_id)
        spike_reg_weight = float(config["training"].get("spike_reg_weight", 0.0))
        if spike_reg_weight > 0 and "spike_reg" in output:
            loss = loss + spike_reg_weight * output["spike_reg"]
        return loss
    return ctc_loss(output, batch, tokenizer.blank_id)


def run_epoch(model, loader, optimizer, scaler, config, tokenizer, device, epoch: int) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    items = 0
    amp_enabled = bool(config["training"].get("amp", False)) and device.type == "cuda"
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    spike_sum = 0.0
    spike_count = 0
    beta_sum = 0.0
    beta_count = 0
    short_beta_sum = 0.0
    short_beta_count = 0
    long_beta_sum = 0.0
    long_beta_count = 0
    fusion_gate_sum = 0.0
    fusion_gate_count = 0

    iterator = tqdm(loader, desc=("train" if is_train else "val") + f" epoch {epoch}", leave=False)
    for batch in iterator:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(**batch)
                loss = compute_loss(output, batch, config, tokenizer)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = int(batch["visuals"].shape[0])
        total += float(loss.detach().item()) * batch_size
        items += batch_size
        if "spike_rate" in output:
            spike_sum += float(output["spike_rate"].detach().item())
            spike_count += 1
        if "beta_mean" in output:
            beta_sum += float(output["beta_mean"].detach().item())
            beta_count += 1
        if "short_beta_mean" in output:
            short_beta_sum += float(output["short_beta_mean"].detach().item())
            short_beta_count += 1
        if "long_beta_mean" in output:
            long_beta_sum += float(output["long_beta_mean"].detach().item())
            long_beta_count += 1
        if "fusion_gate_mean" in output:
            fusion_gate_sum += float(output["fusion_gate_mean"].detach().item())
            fusion_gate_count += 1
        iterator.set_postfix(loss=total / max(items, 1))

    metrics = {"loss": total / max(items, 1)}
    if spike_count:
        metrics["spike_rate"] = spike_sum / spike_count
    if beta_count:
        metrics["beta_mean"] = beta_sum / beta_count
    if short_beta_count:
        metrics["short_beta_mean"] = short_beta_sum / short_beta_count
    if long_beta_count:
        metrics["long_beta_mean"] = long_beta_sum / long_beta_count
    if fusion_gate_count:
        metrics["fusion_gate_mean"] = fusion_gate_sum / fusion_gate_count
    return metrics


def save_checkpoint(path: Path, model, optimizer, config, tokenizer, epoch: int, val_loss: float, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val": best_val,
            "symbols": tokenizer.symbols,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    tokenizer = CharTokenizer(config["text"]["symbols"])
    device = torch.device(config["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")

    train_loader = make_loader(config["data"]["train_manifest"], tokenizer, config, shuffle=True)
    val_loader = make_loader(config["data"]["val_manifest"], tokenizer, config, shuffle=False)
    model = build_model(config, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"].get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["training"].get("amp", False)) and device.type == "cuda")

    start_epoch = 1
    best_val = float("inf")
    resume_from = str(config["training"].get("resume_from", "") or "")
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", best_val)))
        print(f"resumed from {resume_from} at epoch {start_epoch} best_val={best_val:.5f}")

    start = time.time()
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, config, tokenizer, device, epoch)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, None, config, tokenizer, device, epoch)
        val_loss = val_metrics["loss"]
        save_checkpoint(output_dir / "latest_model.pth", model, optimizer, config, tokenizer, epoch, val_loss, best_val)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, config, tokenizer, epoch, val_loss, best_val)
        extra = ""
        if "spike_rate" in train_metrics:
            extra = f" spike_rate={train_metrics['spike_rate']:.4f}"
        if "beta_mean" in train_metrics:
            extra += f" beta_mean={train_metrics['beta_mean']:.4f}"
        if "short_beta_mean" in train_metrics and "long_beta_mean" in train_metrics:
            extra += f" short_beta={train_metrics['short_beta_mean']:.4f} long_beta={train_metrics['long_beta_mean']:.4f}"
        if "fusion_gate_mean" in train_metrics:
            extra += f" fusion_gate={train_metrics['fusion_gate_mean']:.4f}"
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.5f} val_loss={val_loss:.5f} "
            f"best_val={best_val:.5f}{extra} elapsed={time.time() - start:.1f}s"
        )


if __name__ == "__main__":
    main()
