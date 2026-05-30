# -*- coding: utf-8 -*-

"""
Train KAN Risk Fusion – Huấn luyện mạng Kolmogorov-Arnold Network
để dung hợp đặc trưng SNN + hình học → Risk Score.

Cách dùng:
  python training/train_kan.py --dataset data/dataset.npz --snn-checkpoint weights/snn_temporal_best.pt --epochs 60
  python training/train_kan.py --dataset data/dataset.npz --snn-checkpoint weights/snn_temporal_best.pt --lr 0.0003

Quy trình:
  1. Load dataset (.npz) chứa skeleton sequences + geometric features + labels.
  2. Chạy SNN đã train (frozen) để trích xuất temporal features (16-d).
  3. Ghép temporal features (16-d) + geometric features (4-d) = 20-d.
  4. Train KAN mapping 20-d → risk score (regression) hoặc risk class.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TrainKAN")

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


def extract_snn_features(
    snn_checkpoint: str,
    skeletons: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Chạy SNN đã train (frozen) trên toàn bộ skeleton sequences
    để trích xuất temporal features [N, 16].
    """
    from modules.temporal_snn import SpikingNetwork

    ckpt = torch.load(snn_checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)

    snn = SpikingNetwork(input_dim=36, hidden1=64, hidden2=32, output_dim=16)
    snn.load_state_dict(state, strict=False)
    snn.to(device)
    snn.eval()

    all_features: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(len(skeletons)):
            seq = torch.from_numpy(skeletons[i]).float().to(device)  # [seq_len, 36]

            mem1 = torch.zeros(1, 64, device=device)
            mem2 = torch.zeros(1, 32, device=device)

            # Chạy qua từng timestep
            for t in range(seq.shape[0]):
                x = seq[t].unsqueeze(0)  # [1, 36]
                feat, mem1, mem2 = snn(x, mem1, mem2)

            # Lấy feature từ timestep cuối
            all_features.append(feat.cpu().numpy()[0])

    return np.array(all_features, dtype=np.float32)  # [N, 16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KAN Risk Fusion Module.")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--snn-checkpoint", required=True, help="Pretrained SNN weights.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output", default="weights/kan_fusion_best.pt")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--task", default="regression", choices=["regression", "classification"],
                        help="regression: predict risk 0-1 | classification: predict risk class")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    data = np.load(args.dataset)
    skeletons = data["skeletons"]   # [N, seq_len, 36]
    features = data["features"]     # [N, seq_len, 4]
    labels = data["labels"]         # [N]

    logger.info("Dataset: %d samples", len(skeletons))

    # 2. Trích xuất SNN features (frozen)
    logger.info("Extracting SNN temporal features from %s ...", args.snn_checkpoint)
    snn_features = extract_snn_features(args.snn_checkpoint, skeletons, device)
    logger.info("SNN features shape: %s", snn_features.shape)

    # 3. Lấy geometric features trung bình trên chuỗi (hoặc giá trị cuối)
    geo_features = features[:, -1, :]  # [N, 4] – lấy frame cuối cùng

    # 4. Ghép: [N, 16+4] = [N, 20]
    kan_input = np.concatenate([snn_features, geo_features], axis=1).astype(np.float32)
    logger.info("KAN input shape: %s", kan_input.shape)

    # 5. Chuẩn bị target
    if args.task == "regression":
        # Chuyển label (0,1,2) → risk score (0.0, 0.5, 1.0)
        num_classes = int(labels.max()) + 1
        targets = labels.astype(np.float32) / max(num_classes - 1, 1)
        targets = targets.reshape(-1, 1)
    else:
        targets = labels.astype(np.int64)

    # 6. Train/val split
    n = len(kan_input)
    indices = np.random.permutation(n)
    val_size = int(n * args.val_split)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    x_train = torch.from_numpy(kan_input[train_idx])
    x_val = torch.from_numpy(kan_input[val_idx])

    if args.task == "regression":
        y_train = torch.from_numpy(targets[train_idx])
        y_val = torch.from_numpy(targets[val_idx])
        criterion = nn.MSELoss()
    else:
        y_train = torch.from_numpy(targets[train_idx]).long()
        y_val = torch.from_numpy(targets[val_idx]).long()
        criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch)

    # 7. Build KAN model
    from modules.risk_fusion_kan import KANFusionNet

    if args.task == "regression":
        model = KANFusionNet(input_dim=20, hidden1=12, hidden2=6, grid1=6, grid2=4)
    else:
        # Thay Sigmoid cuối bằng output num_classes
        model = nn.Sequential(
            KANFusionNet(input_dim=20, hidden1=12, hidden2=6, grid1=6, grid2=4),
            # KANFusionNet output = 1, cần expand
        )
        # Xây lại cho classification
        from modules.risk_fusion_kan import KANLayer
        num_classes = int(labels.max()) + 1
        model = nn.Sequential(
            KANLayer(20, 12, grid_size=6),
            nn.SiLU(),
            nn.LayerNorm(12),
            KANLayer(12, 6, grid_size=4),
            nn.SiLU(),
            nn.Linear(6, num_classes),
        )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger.info("KAN Model: %d parameters | Task: %s", sum(p.numel() for p in model.parameters()), args.task)
    logger.info("Device: %s | Epochs: %d | Batch: %d | LR: %g", device, args.epochs, args.batch, args.lr)

    # 8. Training loop
    best_val_metric = float("inf") if args.task == "regression" else 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(x_train)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                if args.task == "classification":
                    val_correct += (out.argmax(1) == yb).sum().item()
        val_loss /= len(x_val)

        scheduler.step()

        if args.task == "regression":
            logger.info("Epoch %3d/%d | Train MSE=%.5f | Val MSE=%.5f", epoch, args.epochs, train_loss, val_loss)
            improved = val_loss < best_val_metric
            if improved:
                best_val_metric = val_loss
        else:
            val_acc = val_correct / len(x_val)
            logger.info("Epoch %3d/%d | Train loss=%.4f | Val loss=%.4f acc=%.3f", epoch, args.epochs, train_loss, val_loss, val_acc)
            improved = val_acc > best_val_metric
            if improved:
                best_val_metric = val_acc

        if improved:
            patience_counter = 0
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, str(out_path))
            logger.info("  ✔ Best model saved → %s", out_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Training finished. Best metric: %.5f", best_val_metric)


if __name__ == "__main__":
    main()
