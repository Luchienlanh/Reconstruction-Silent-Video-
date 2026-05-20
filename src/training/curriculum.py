import os
import gc
import glob
import math
import random
import warnings
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from spikingjelly.activation_based import neuron, surrogate, functional
from dataclasses import dataclass, field
from models.loss import PhaseAwareMelLoss

@dataclass
class PhaseConfig:
    """Configuration for one curriculum training phase."""
    phase_id: int
    name: str
    num_epochs: int
    lr: float = 1e-4
    max_grad_norm: float = 1.0

@dataclass
class CurriculumConfig:
    """Configuration for curriculum training across all phases."""
    phases: List[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(phase_id=1, name="F0 (Pitch)",       num_epochs=5,  lr=2e-4),
        PhaseConfig(phase_id=2, name="Voice Tone",        num_epochs=10, lr=1e-4),
        PhaseConfig(phase_id=3, name="Oscillation",       num_epochs=8,  lr=5e-5),
        PhaseConfig(phase_id=4, name="Energy",            num_epochs=5,  lr=3e-5),
    ])
    checkpoint_dir: str = "curriculum_checkpoints"
    save_every: int = 5
    log_memory_every: int = 25

def train_curriculum(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    curriculum_config: CurriculumConfig = None,
    reset_net_fn=None,
) -> Dict[str, Any]:
    """
    Multi-phase curriculum training controller.

    Args:
        encoder: visual+landmark encoder (e.g., VisualLandmarkEncoderV2)
        decoder: mel decoder (e.g., MelTemporalUpsampleDecoder wrapping TFiLMSIRENDecoder)
        dataloader: DataLoader using VNLipDatasetV2 + collate_pad_v2
        device: torch device
        curriculum_config: optional CurriculumConfig, uses defaults if None
        reset_net_fn: function to call for resetting spiking neural nets
                      (e.g., spikingjelly.activation_based.functional.reset_net)

    Returns:
        dict with keys: 'history' (list of per-epoch losses), 'phase_boundaries' (list of epoch indices)
    """
    if curriculum_config is None:
        curriculum_config = CurriculumConfig()

    os.makedirs(curriculum_config.checkpoint_dir, exist_ok=True)

    criterion = PhaseAwareMelLoss().to(device)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    phase_boundaries = []
    global_epoch = 0
    best_loss = float("inf")

    for phase_cfg in curriculum_config.phases:
        phase_id = phase_cfg.phase_id
        phase_name = phase_cfg.name
        phase_epochs = phase_cfg.num_epochs
        phase_lr = phase_cfg.lr
        max_grad_norm = phase_cfg.max_grad_norm

        print(f"\n{'='*60}")
        print(f"PHASE {phase_id}: {phase_name}")
        print(f"  Epochs: {phase_epochs} | LR: {phase_lr} | Grad Clip: {max_grad_norm}")
        print(f"{'='*60}")

        # Create optimizer fresh for each phase (allows different LR)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=phase_lr,
            weight_decay=1e-5,
        )

        phase_boundaries.append(global_epoch)

        for local_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            encoder.train()
            decoder.train()
            total_loss = 0.0
            processed_batches = 0

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Phase{phase_id} Ep{local_epoch}")):
                # Unpack batch
                paths = None
                landmark_batch = None
                if len(batch) == 5:
                    video_batch, landmark_batch, target_batch, lengths, paths = batch
                elif len(batch) == 4 and torch.is_tensor(batch[1]) and batch[1].dim() == 4:
                    video_batch, landmark_batch, target_batch, lengths = batch
                elif len(batch) == 4:
                    video_batch, target_batch, lengths, paths = batch
                else:
                    video_batch, target_batch, lengths = batch

                lengths = lengths.to(device)
                video_batch = video_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                if landmark_batch is not None:
                    landmark_batch = landmark_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if reset_net_fn is not None:
                    reset_net_fn(encoder)

                try:
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        if landmark_batch is not None:
                            z = encoder(video_batch, landmark_batch)
                        else:
                            z = encoder(video_batch)

                        # MelTemporalUpsampleDecoder expects target_len
                        if hasattr(decoder, 'infer_target_len'):
                            target_pred = decoder(z, target_len=target_batch.shape[1])
                        else:
                            target_pred = decoder(z)

                    with torch.amp.autocast("cuda", enabled=False):
                        loss = criterion(
                            target_pred.float(),
                            target_batch.float(),
                            lengths,
                            phase=phase_id,
                        )

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at phase={phase_id}, epoch={local_epoch}, "
                            f"batch={batch_idx}: {float(loss.detach().cpu())}"
                        )

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            list(encoder.parameters()) + list(decoder.parameters()),
                            max_grad_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += float(loss.detach().cpu())
                    processed_batches += 1

                    if (
                        amp_enabled
                        and curriculum_config.log_memory_every
                        and processed_batches % curriculum_config.log_memory_every == 0
                    ):
                        peak = torch.cuda.max_memory_allocated() / 1024**3
                        reserved = torch.cuda.max_memory_reserved() / 1024**3
                        print(
                            f"  -> batch {processed_batches}: "
                            f"target_T={int(lengths.max().item())}, "
                            f"peak={peak:.2f}GB, reserved={reserved:.2f}GB"
                        )
                        torch.cuda.reset_peak_memory_stats()

                except torch.cuda.OutOfMemoryError:
                    file_hint = paths[0] if paths else "<unknown>"
                    print(
                        f"OOM at phase={phase_id}, epoch={local_epoch}, batch={batch_idx}, "
                        f"target_T={int(lengths.max().item())}, file={file_hint}"
                    )
                    raise
                finally:
                    if reset_net_fn is not None:
                        reset_net_fn(encoder)
                    del video_batch, target_batch, lengths
                    if landmark_batch is not None:
                        del landmark_batch
                    if "z" in locals():
                        del z
                    if "target_pred" in locals():
                        del target_pred
                    if "loss" in locals():
                        del loss
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            if processed_batches == 0:
                raise RuntimeError(
                    f"Khong train duoc batch nao o phase={phase_id}. "
                    "Hay kiem tra dataset/max_frames/collate_fn."
                )

            avg_loss = total_loss / processed_batches
            history.append(avg_loss)

            print(
                f"Phase {phase_id} | Epoch {local_epoch:3d}/{phase_epochs} "
                f"(global {global_epoch}) | Loss: {avg_loss:.6f}"
            )

            # Save checkpoints
            state = {
                "global_epoch": global_epoch,
                "phase_id": phase_id,
                "local_epoch": local_epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": avg_loss,
                "history": history,
                "phase_boundaries": phase_boundaries,
            }

            if (
                local_epoch % curriculum_config.save_every == 0
                or local_epoch == phase_epochs
            ):
                ckpt_path = os.path.join(
                    curriculum_config.checkpoint_dir,
                    f"phase{phase_id}_epoch{local_epoch}.pth",
                )
                torch.save(state, ckpt_path)
                print(f"  -> Saved: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(curriculum_config.checkpoint_dir, "best_model.pth")
                torch.save(state, best_path)
                print(f"  -> Best model saved: {best_path}")

        # Phase boundary checkpoint
        phase_ckpt = os.path.join(
            curriculum_config.checkpoint_dir,
            f"phase{phase_id}_complete.pth",
        )
        torch.save(state, phase_ckpt)
        print(f"\n  Phase {phase_id} ({phase_name}) complete. Saved: {phase_ckpt}")

    print(f"\n{'='*60}")
    print(f"Curriculum training complete! Total epochs: {global_epoch}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"{'='*60}")

    return {"history": history, "phase_boundaries": phase_boundaries}

