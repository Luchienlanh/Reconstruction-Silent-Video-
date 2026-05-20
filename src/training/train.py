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
from models.loss import make_criterion

def train_one_epoch(
        encoder: nn.Module,
        decoder: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_mask: bool=True,
        max_grad_norm: Optional[float]=1.0,
        scaler: Optional[torch.amp.GradScaler]=None,
        log_memory_every: int=25,
    ):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    processed_batches = 0
    amp_enabled = device.type == 'cuda'
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
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
        functional.reset_net(encoder)

        try:
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                if landmark_batch is not None:
                    z = encoder(video_batch, landmark_batch)
                else:
                    z = encoder(video_batch)
                if isinstance(criterion, MelReconstructionLoss):
                    target_pred = decoder(z, target_len=target_batch.shape[1])
                else:
                    target_pred = decoder(z)

            with torch.amp.autocast('cuda', enabled=False):
                if isinstance(criterion, (CombinedAudioLoss, MelReconstructionLoss)):
                    loss = criterion(target_pred.float(), target_batch.float(), lengths if use_mask else None)
                else:
                    loss = criterion(target_pred.float(), target_batch.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at batch={batch_idx}: {float(loss.detach().cpu())}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    max_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu())
            processed_batches += 1

            if amp_enabled and log_memory_every and processed_batches % log_memory_every == 0:
                peak = torch.cuda.max_memory_allocated() / 1024**3
                reserved = torch.cuda.max_memory_reserved() / 1024**3
                print(f"  -> batch {processed_batches}: target_T={int(lengths.max().item())}, peak={peak:.2f}GB, reserved={reserved:.2f}GB")
                torch.cuda.reset_peak_memory_stats()

        except torch.cuda.OutOfMemoryError:
            file_hint = paths[0] if paths else '<unknown>'
            print(f"OOM tai batch={batch_idx}, target_T={int(lengths.max().item())}, file={file_hint}")
            raise
        finally:
            functional.reset_net(encoder)
            del video_batch, target_batch, lengths
            if landmark_batch is not None: del landmark_batch
            if 'z' in locals(): del z
            if 'target_pred' in locals(): del target_pred
            if 'loss' in locals(): del loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    if processed_batches == 0:
        raise RuntimeError("Khong train duoc batch nao. Hay kiem tra dataset/max_frames/collate_fn.")

    return total_loss / processed_batches

def train_full(encoder, decoder, train_loader, optimizer, criterion, device,
               num_epochs=50, use_mask=True, max_grad_norm=1.0,
               checkpoint_dir="checkpoints", save_best=True, scaler=None):
    """Hu?n luy?n nhi?u epoch, in log loss, l?u checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    history = []
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(encoder, decoder, train_loader,
                                   optimizer, criterion, device,
                                   use_mask=use_mask,
                                   max_grad_norm=max_grad_norm,
                                   scaler=scaler)
        history.append(avg_loss)
        print(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {avg_loss:.6f}")

        state = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
            'max_frames': globals().get('MAX_FRAMES', None),
        }

        if epoch % 10 == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save(state, checkpoint_path)
            print(f"  -> ?? l?u checkpoint: {checkpoint_path}")

        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"  -> ?? l?u model t?t nh?t: {best_path}")

    print("Ho?n t?t hu?n luy?n!")
    return history

def build_overfit_2pt_model(device):
    visual_encoder = build_encoder(globals().get("ENCODER_TYPE", "non_snn")).to(device)

    # Lay so diem landmarks tu dataset/file dau tien.
    if OVERFIT_2PT_FILES is not None:
        sample = torch.load(OVERFIT_2PT_FILES[0], map_location="cpu", weights_only=False)
        sample_lm, _ = find_landmarks_in_data(sample, require=True, path=OVERFIT_2PT_FILES[0])
        num_landmark_points = sample_lm.shape[1]
    else:
        num_landmark_points = getattr(dataset, "landmark_num_points", None)
        if num_landmark_points is None:
            num_landmark_points = infer_landmark_num_points(DATA_DIR)

    encoder = VisualLandmarkEncoder(
        visual_encoder,
        num_landmark_points=num_landmark_points,
    ).to(device)

    base_decoder = TFiLMSIRENDecoder(
        hidden_dim=512,
        out_dim=80,
        num_layers=4,
        use_conv=True,
        output_activation=None,
    ).to(device)

    decoder = MelTemporalUpsampleDecoder(
        base_decoder,
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)

    return encoder, decoder

