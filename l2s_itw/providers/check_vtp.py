from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REQUIRED_MODULES = [
    "configargparse",
    "decord",
    "einops",
    "linear_attention_transformer",
    "local_attention",
    "numpy",
    "pandas",
    "torch",
    "transformers",
]

EXPECTED_MIN_BYTES = {
    "feature_extractor.pth": 900_000_000,
    "ft_lrs2.pth": 800_000_000,
    "ft_lrs3.pth": 800_000_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local readiness for the official VTP provider.")
    parser.add_argument("--repo-dir", default="external/vtp")
    parser.add_argument("--ckpt-path", default="pretrained_models/vtp/ft_lrs2.pth")
    parser.add_argument("--cnn-ckpt-path", default="pretrained_models/vtp/feature_extractor.pth")
    return parser.parse_args()


def exists_label(path: Path) -> str:
    if not path.exists():
        return "missing"
    min_bytes = EXPECTED_MIN_BYTES.get(path.name)
    if min_bytes is not None and path.stat().st_size < min_bytes:
        return f"partial ({path.stat().st_size} bytes)"
    return "ok"


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    ckpt_path = Path(args.ckpt_path)
    cnn_ckpt_path = Path(args.cnn_ckpt_path)

    print(f"repo_dir: {repo_dir} [{exists_label(repo_dir)}]")
    print(f"repo models.py: {repo_dir / 'models.py'} [{exists_label(repo_dir / 'models.py')}]")
    print(f"repo tokenizer cache: {repo_dir / 'checkpoints' / 'tokenizers'} [{exists_label(repo_dir / 'checkpoints' / 'tokenizers')}]")
    print(f"lip-reading ckpt: {ckpt_path} [{exists_label(ckpt_path)}]")
    print(f"feature extractor ckpt: {cnn_ckpt_path} [{exists_label(cnn_ckpt_path)}]")
    print("python modules:")
    for module in REQUIRED_MODULES:
        status = "ok" if importlib.util.find_spec(module) is not None else "missing"
        print(f"  {module}: {status}")


if __name__ == "__main__":
    main()
