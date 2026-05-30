# -*- coding: utf-8 -*-

"""
Train SNN Temporal Module – Huấn luyện mạng Spiking Neural Network
từ chuỗi skeleton để nhận diện dáng đi bất ổn.

Cách dùng:
  python training/train_snn.py --dataset data/dataset.npz --epochs 80
  python training/train_snn.py --dataset data/dataset.npz --epochs 120 --lr 0.0005 --batch 64

Input dataset (.npz):
  skeletons: [N, seq_len, 36]  – chuỗi skeleton
  labels:    [N]               – nhãn risk (0=safe, 1=warning, 2=danger)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TrainSNN")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("Cần cài đặt PyTorch: pip install torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_snn_classifier(input_dim: int, hidden1: int, hidden2: int, num_classes: int):
    """Xây dựng mạng SNN + classifier head cho training."""
    from modules.temporal_snn import LIFCell

    class SNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lif1 = LIFCell(input_dim, hidden1, decay=0.85, threshold=1.0)
            self.lif2 = LIFCell(hidden1, hidden2, decay=0.85, threshold=1.0)
            self.readout = nn.Linear(hidden2, 16)
            self.classifier = nn.Sequential(
                nn.LayerNorm(16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16, num_classes),
            )
            self.hidden1 = hidden1
            self.hidden2 = hidden2

        def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
            """
            x_seq: [batch, seq_len, input_dim]
            Returns: [batch, num_classes]
            """
            batch, seq_len, _ = x_seq.shape
            device = x_seq.device

            mem1 = torch.zeros(batch, self.hidden1, device=device)
            mem2 = torch.zeros(batch, self.hidden2, device=device)

            # Chạy qua từng bước thời gian
            for t in range(seq_len):
                spk1, mem1 = self.lif1(x_seq[:, t, :], mem1)
                spk2, mem2 = self.lif2(spk1, mem2)

            # Lấy readout từ spike cuối cùng
            features = self.readout(spk2)
            logits = self.classifier(features)
            return logits

    return SNNClassifier()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SNN Temporal Module.")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output", default="weights/snn_temporal_best.pt")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        sys.exit(1)

    # Load dataset
    data = np.load(args.dataset)
    skeletons = data["skeletons"]  # [N, seq_len, 36]
    labels = data["labels"]        # [N]

    logger.info("Dataset: %d samples, seq_len=%d, input_dim=%d", *skeletons.shape)
    logger.info("Classes: %s", dict(zip(*np.unique(labels, return_counts=True))))

    num_classes = int(labels.max()) + 1

    # Train/val split
    n = len(skeletons)
    indices = np.random.permutation(n)
    val_size = int(n * args.val_split)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    x_train = torch.from_numpy(skeletons[train_idx]).float()
    y_train = torch.from_numpy(labels[train_idx]).long()
    x_val = torch.from_numpy(skeletons[val_idx]).float()
    y_val = torch.from_numpy(labels[val_idx]).long()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_snn_classifier(
        input_dim=skeletons.shape[2],
        hidden1=64, hidden2=32,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))
    logger.info("Device: %s | Epochs: %d | Batch: %d | LR: %g", device, args.epochs, args.batch, args.lr)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d | Train loss=%.4f acc=%.3f | Val loss=%.4f acc=%.3f | LR=%.6f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]["lr"],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "num_classes": num_classes,
            }, str(out_path))
            logger.info("  ✔ Best model saved → %s (val_acc=%.3f)", out_path, val_acc)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

    logger.info("Training finished. Best val accuracy: %.3f", best_val_acc)


if __name__ == "__main__":
    main()
