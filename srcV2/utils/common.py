from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def batch_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: batch_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(batch_to_device(v, device) for v in value)
    return value


def safe_name(path: str | Path, max_len: int = 110) -> str:
    p = Path(path)
    base = f"{p.parent.parent.name}_{p.parent.name}" if len(p.parts) >= 2 else p.stem
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    digest = hashlib.sha1(str(p).encode("utf-8", errors="ignore")).hexdigest()[:10]
    if len(base) > max_len:
        base = base[:max_len].rstrip("._")
    return f"{base}_{digest}"


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
