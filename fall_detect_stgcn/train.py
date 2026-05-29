#!/usr/bin/env python
"""Train a fall-detection model on extracted UP-Fall MediaPipe pose sequences.

Supports all model types: lstm, speech_tcn, snn, pose_r2plus1d, spiking_stgcn.
Enhanced with binary fall/non-fall metrics (fall recall, non-fall recall).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fall_detect_stgcn.dataset import (
    UPFallPoseDataset,
    filter_pose_files,
    list_pose_files,
    parse_subjects,
    read_meta,
)
from fall_detect_stgcn.labels import FALL_ACTIVITIES, label_names
from fall_detect_stgcn.pose_features import INPUT_DIM
from fall_detect_stgcn.models.registry import MODEL_TYPES, build_model


def split_random_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split samples randomly with deterministic shuffling."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(round(total * train_ratio))
    val_end = train_end + int(round(total * val_ratio))
    train_end = min(max(train_end, 0), total)
    val_end = min(max(val_end, train_end), total)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def subjects_in_files(files: list[Path]) -> list[int]:
    subjects = {int(read_meta(path)["subject"]) for path in files}
    return sorted(subjects)


def split_subject_kfold(
    files: list[Path],
    k_folds: int,
    fold_index: int,
    seed: int,
    val_offset: int = 1,
) -> tuple[list[Path], list[Path], list[Path], set[int], set[int], set[int]]:
    """Subject-independent k-fold split.

    test fold = fold_index, validation fold = fold_index + val_offset, train = rest.
    """
    if k_folds < 2:
        raise ValueError("--k-folds must be at least 2.")
    subjects = subjects_in_files(files)
    if k_folds > len(subjects):
        raise ValueError(f"--k-folds={k_folds} is larger than available subjects={len(subjects)}.")
    if fold_index < 0 or fold_index >= k_folds:
        raise ValueError(f"--fold-index must be in [0, {k_folds - 1}].")

    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    folds = [set(map(int, fold)) for fold in np.array_split(shuffled, k_folds)]

    test_subjects = folds[fold_index]
    val_subjects = folds[(fold_index + val_offset) % k_folds]
    train_subjects = set(subjects) - test_subjects - val_subjects
    return (
        filter_pose_files(files, train_subjects),
        filter_pose_files(files, val_subjects),
        filter_pose_files(files, test_subjects),
        train_subjects,
        val_subjects,
        test_subjects,
    )


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    files: list[Path],
    task: str,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    augment: bool,
) -> tuple[UPFallPoseDataset, DataLoader]:
    dataset = UPFallPoseDataset(files=files, task=task, seq_len=seq_len, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = max(float(counts.sum()), 1.0)
    weights = total / (num_classes * np.maximum(counts, 1.0))
    weights[counts == 0] = 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_fall_metrics(confusion: np.ndarray, task: str) -> dict[str, float | None]:
    """Compute binary fall/non-fall metrics from the confusion matrix.

    For multiclass (11 classes): classes 0-4 are fall, 5-10 are non-fall.
    For binary (2 classes): class 0 = non-fall, class 1 = fall.

    Returns:
        Dict with binary_accuracy, fall_recall, non_fall_recall.
    """
    num_classes = confusion.shape[0]
    total = int(confusion.sum())
    if total == 0:
        return {"binary_accuracy": None, "fall_recall": None, "non_fall_recall": None}

    if task == "binary":
        # confusion is 2×2: [0]=non_fall, [1]=fall
        tp_fall = int(confusion[1, 1])
        fn_fall = int(confusion[1, 0])
        tp_nf = int(confusion[0, 0])
        fn_nf = int(confusion[0, 1])
    else:
        # Multiclass: fall = classes 0-4 (activity 1-5), non-fall = 5-10 (activity 6-11)
        fall_indices = list(range(5))
        nf_indices = list(range(5, num_classes))
        # True fall predicted as fall (any fall class)
        tp_fall = sum(int(confusion[i, j]) for i in fall_indices for j in fall_indices)
        fn_fall = sum(int(confusion[i, j]) for i in fall_indices for j in nf_indices)
        tp_nf = sum(int(confusion[i, j]) for i in nf_indices for j in nf_indices)
        fn_nf = sum(int(confusion[i, j]) for i in nf_indices for j in fall_indices)

    fall_total = tp_fall + fn_fall
    nf_total = tp_nf + fn_nf
    binary_correct = tp_fall + tp_nf
    binary_total = fall_total + nf_total

    return {
        "binary_accuracy": binary_correct / max(binary_total, 1),
        "fall_recall": tp_fall / max(fall_total, 1) if fall_total > 0 else None,
        "non_fall_recall": tp_nf / max(nf_total, 1) if nf_total > 0 else None,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    task: str,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        batch = y.numel()
        total_loss += float(loss.item()) * batch
        total += batch
        correct += int((pred == y).sum().item())
        for target, guess in zip(y.cpu().numpy(), pred.cpu().numpy()):
            confusion[int(target), int(guess)] += 1

    if total == 0:
        return {"loss": None, "accuracy": None, "confusion": confusion.tolist()}

    fall_metrics = compute_fall_metrics(confusion, task)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "binary_accuracy": fall_metrics["binary_accuracy"],
        "fall_recall": fall_metrics["fall_recall"],
        "non_fall_recall": fall_metrics["non_fall_recall"],
        "confusion": confusion.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        pred = logits.argmax(dim=1)
        batch = y.numel()
        total_loss += float(loss.item()) * batch
        total += batch
        correct += int((pred == y).sum().item())

    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    names: list[str],
    epoch: int,
    metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": args.model_type,
        "task": args.task,
        "seq_len": args.seq_len,
        "input_dim": INPUT_DIM,
        "num_classes": len(names),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "transformer_layers": args.transformer_layers,
        "feature_set": args.feature_set,
        "label_names": names,
        "split_mode": args.split_mode,
        "train_subjects": args.train_subjects,
        "val_subjects": args.val_subjects,
        "test_subjects": args.test_subjects,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "k_folds": args.k_folds,
        "fold_index": args.fold_index,
        "kfold_val_offset": args.kfold_val_offset,
        "seed": args.seed,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fall-detection model on UP-Fall pose data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pose-dir", default="Processed_UPFall_Pose")
    parser.add_argument("--out-dir", default="runs/upfall_spiking_stgcn")
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="spiking_stgcn")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--split-mode", choices=["subject", "random", "kfold"], default="subject")
    parser.add_argument("--train-subjects", default="1-13")
    parser.add_argument("--val-subjects", default="14-15")
    parser.add_argument("--test-subjects", default="16-17")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument(
        "--kfold-val-offset",
        type=int,
        default=1,
        help="Validation fold offset relative to test fold in subject k-fold mode.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument(
        "--feature-set",
        choices=["position", "pos_vel", "pos_vel_acc", "full"],
        default="full",
        help="Feature ablation for stgcn/spiking_stgcn.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug limit after split.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = get_args()
    seed_everything(args.seed)

    pose_dir = Path(args.pose_dir)
    all_files = list_pose_files(pose_dir)
    if not all_files:
        print(f"No .npz pose files found in {pose_dir}. Run extract_upfall_pose.py first.", file=sys.stderr)
        return 2

    split_subject_info = None
    if args.split_mode == "subject":
        train_files = filter_pose_files(all_files, parse_subjects(args.train_subjects))
        val_files = filter_pose_files(all_files, parse_subjects(args.val_subjects))
        test_files = filter_pose_files(all_files, parse_subjects(args.test_subjects))
    elif args.split_mode == "random":
        train_files, val_files, test_files = split_random_files(
            all_files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        (
            train_files,
            val_files,
            test_files,
            train_subjects,
            val_subjects,
            test_subjects,
        ) = split_subject_kfold(
            all_files,
            k_folds=args.k_folds,
            fold_index=args.fold_index,
            seed=args.seed,
            val_offset=args.kfold_val_offset,
        )
        split_subject_info = (train_subjects, val_subjects, test_subjects)
    if args.max_samples is not None:
        train_files = train_files[: args.max_samples]
        val_files = val_files[: args.max_samples]
        test_files = test_files[: args.max_samples]

    if not train_files:
        print("Train split is empty. Check --train-subjects and --pose-dir.", file=sys.stderr)
        return 2

    names = label_names(args.task)
    num_classes = len(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, train_loader = make_loader(
        train_files, args.task, args.seq_len, args.batch_size,
        shuffle=True, num_workers=args.num_workers, augment=True,
    )
    _, val_loader = make_loader(
        val_files, args.task, args.seq_len, args.batch_size,
        shuffle=False, num_workers=args.num_workers, augment=False,
    )
    _, test_loader = make_loader(
        test_files, args.task, args.seq_len, args.batch_size,
        shuffle=False, num_workers=args.num_workers, augment=False,
    )

    print(f"Model: {args.model_type}")
    print(f"Device: {device}")
    print(f"Task: {args.task} ({num_classes} classes)")
    print(f"Split mode: {args.split_mode}")
    if split_subject_info is not None:
        train_subjects, val_subjects, test_subjects = split_subject_info
        print(f"K-fold: {args.fold_index + 1}/{args.k_folds}")
        print(f"Train subjects: {sorted(train_subjects)}")
        print(f"Val subjects:   {sorted(val_subjects)}")
        print(f"Test subjects:  {sorted(test_subjects)}")
    print(f"Files: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    model = build_model(
        model_type=args.model_type,
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        transformer_layers=args.transformer_layers,
        feature_set=args.feature_set,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset.labels, num_classes, device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    last_path = out_dir / "last_model.pt"
    metrics_path = out_dir / "metrics.json"

    history: list[dict] = []
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes, args.task) if val_files else {}
        val_acc = val_metrics.get("accuracy")
        score = float(val_acc) if val_acc is not None else float(train_metrics["accuracy"])
        scheduler.step(score)

        # Format display values
        val_acc_str = f"{val_acc:.3f}" if val_acc is not None else "n/a"
        fall_recall = val_metrics.get("fall_recall")
        fall_recall_str = f"{fall_recall:.3f}" if fall_recall is not None else "n/a"
        nf_recall = val_metrics.get("non_fall_recall")
        nf_recall_str = f"{nf_recall:.3f}" if nf_recall is not None else "n/a"
        bin_acc = val_metrics.get("binary_accuracy")
        bin_acc_str = f"{bin_acc:.3f}" if bin_acc is not None else "n/a"

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": {k: v for k, v in val_metrics.items() if k != "confusion"},
            "val_confusion": val_metrics.get("confusion"),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"epoch {epoch:03d}  "
            f"train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['accuracy']:.3f}  "
            f"val_acc={val_acc_str}  "
            f"fall_recall={fall_recall_str}  nf_recall={nf_recall_str}  "
            f"bin_acc={bin_acc_str}"
        )

        save_checkpoint(last_path, model, args, names, epoch, record)
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, args, names, epoch, record)
            print(f"  ↑ New best (score={score:.3f})")

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    # Final test evaluation
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes, args.task) if test_files else {}

    print(f"\nBest checkpoint: {best_path.resolve()}")
    if test_metrics:
        print(f"Test accuracy:      {test_metrics['accuracy']:.3f}")
        print(f"Test binary acc:    {test_metrics.get('binary_accuracy', 'n/a')}")
        print(f"Test fall recall:   {test_metrics.get('fall_recall', 'n/a')}")
        print(f"Test NF recall:     {test_metrics.get('non_fall_recall', 'n/a')}")
        print(f"Test loss:          {test_metrics['loss']:.4f}")
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)
    else:
        print("No test files were available.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
