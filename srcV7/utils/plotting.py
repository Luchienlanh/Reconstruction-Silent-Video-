from __future__ import annotations

from pathlib import Path

import torch


def save_mel_comparison(pred: torch.Tensor, target: torch.Tensor, path: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_np = pred.detach().float().cpu()[0].T
    tgt_np = target.detach().float().cpu()[0].T
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].imshow(tgt_np, aspect="auto", origin="lower")
    axes[0].set_title("target")
    axes[1].imshow(pred_np, aspect="auto", origin="lower")
    axes[1].set_title("prediction")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

