# -*- coding: utf-8 -*-

"""
Train Baselines: MLP & LSTM/GRU.

Huấn luyện các mô hình baseline (cơ bản) để có cơ sở so sánh với các mô hình 
tiên tiến (KAN và SNN) như được đề cập trong báo cáo.

1. LSTM/GRU Baseline: So sánh với SNN Temporal Module (dữ liệu chuỗi)
2. MLP Baseline: So sánh với KAN Risk Fusion Module (dữ liệu bảng/vector)

Cách chạy:
  python training/train_baselines.py --dataset data/dataset.npz --model lstm --epochs 50
  python training/train_baselines.py --dataset data/dataset.npz --model mlp --epochs 50
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TrainBaselines")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("Cần cài đặt PyTorch: pip install torch")

# ======================================================================
#  1. Baseline RNN (LSTM/GRU) Model
# ======================================================================
class BaselineRNN(nn.Module):
    """Mạng LSTM/GRU dùng để so sánh với mạng SNN."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, rnn_type: str = "lstm"):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [batch, seq_len, input_dim]
        # output: [batch, seq_len, hidden_dim], h_n: [num_layers, batch, hidden_dim]
        rnn_out, _ = self.rnn(x_seq)
        
        # Chỉ lấy đặc trưng ở timestep cuối cùng
        last_out = rnn_out[:, -1, :]
        
        logits = self.fc(last_out)
        return logits


# ======================================================================
#  2. Baseline MLP Model
# ======================================================================
class BaselineMLP(nn.Module):
    """Mạng Multi-Layer Perceptron (MLP) dùng để so sánh với mạng KAN."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
#  Training & Evaluation Loop
# ======================================================================
def train_and_eval(model, train_loader, val_loader, epochs, lr, device, save_path):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 15
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += x_batch.size(0)
            
        train_acc = train_correct / train_total
        
        # Eval
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * x_batch.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += x_batch.size(0)
                
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        logger.info(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  --> Saved best model with val_acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
    return best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Models (LSTM/MLP)")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "mlp"], help="Loại mô hình baseline cần train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data
    logger.info(f"Loading data từ {args.dataset}")
    data = np.load(args.dataset)
    labels = data["labels"]
    num_classes = int(labels.max()) + 1
    
    # 2. Chuẩn bị đầu vào tùy theo model
    if args.model in ["lstm", "gru"]:
        # LSTM/GRU nhận input chuỗi thời gian: [N, seq_len, 36]
        inputs = data["skeletons"]
        logger.info(f"Training {args.model.upper()} với dữ liệu chuỗi (như SNN). Input shape: {inputs.shape}")
        model = BaselineRNN(input_dim=inputs.shape[2], hidden_dim=64, num_classes=num_classes, rnn_type=args.model).to(device)
        
    elif args.model == "mlp":
        # MLP nhận input vector (giống KAN)
        # Ta cần tính toán tương tự KAN: Lấy temporal feature từ LSTM (hoặc flatten chuỗi) + geo features
        # Để đơn giản và công bằng với KAN, ta ghép khung hình cuối của skeleton và geo_features
        skeletons = data["skeletons"][:, -1, :] # Lấy frame cuối [N, 36]
        geo_features = data["features"][:, -1, :] # [N, 4]
        inputs = np.concatenate([skeletons, geo_features], axis=1) # [N, 40]
        logger.info(f"Training MLP với vector đặc trưng. Input shape: {inputs.shape}")
        model = BaselineMLP(input_dim=inputs.shape[1], num_classes=num_classes).to(device)

    # 3. Chia tập train/val
    n = len(inputs)
    indices = np.random.permutation(n)
    val_size = int(n * 0.2)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    x_train, y_train = torch.from_numpy(inputs[train_idx]).float(), torch.from_numpy(labels[train_idx]).long()
    x_val, y_val = torch.from_numpy(inputs[val_idx]).float(), torch.from_numpy(labels[val_idx]).long()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch)

    # 4. Train
    save_path = f"weights/baseline_{args.model}_best.pt"
    logger.info(f"Bắt đầu huấn luyện mô hình {args.model.upper()}...")
    best_acc = train_and_eval(model, train_loader, val_loader, args.epochs, args.lr, device, save_path)
    logger.info(f"Hoàn tất. Độ chính xác Val tốt nhất của {args.model.upper()}: {best_acc:.4f}")

if __name__ == "__main__":
    main()
