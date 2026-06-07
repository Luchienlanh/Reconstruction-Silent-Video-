from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from l2s_itw.data.manifest import write_manifest


TEXTS = [
    "we need accurate lip speech",
    "the model follows the video",
    "visual text attention aligns words",
    "speech should be clear and synced",
    "this is a clean pipeline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic data for pipeline smoke tests.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train", type=int, default=12)
    parser.add_argument("--num-val", type=int, default=4)
    parser.add_argument("--visual-dim", type=int, default=512)
    parser.add_argument("--speaker-dim", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--upsample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def make_split(
    root: Path,
    split: str,
    count: int,
    visual_dim: int,
    speaker_dim: int,
    n_mels: int,
    upsample: int,
) -> list[dict[str, str]]:
    rows = []
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        sample_id = f"{split}_{index:04d}"
        frames = random.randint(18, 45)
        visual = torch.randn(frames, visual_dim)
        speaker = torch.randn(speaker_dim)
        mel = torch.randn(frames * upsample, n_mels) * 0.4

        visual_path = split_dir / f"{sample_id}.visual.pt"
        speaker_path = split_dir / f"{sample_id}.speaker.pt"
        mel_path = split_dir / f"{sample_id}.mel.pt"
        torch.save({"visual_features": visual}, visual_path)
        torch.save({"speaker_embedding": speaker}, speaker_path)
        torch.save({"mel": mel}, mel_path)

        rows.append(
            {
                "id": sample_id,
                "visual_feature_path": str(visual_path.relative_to(root)),
                "speaker_embedding_path": str(speaker_path.relative_to(root)),
                "mel_path": str(mel_path.relative_to(root)),
                "text": random.choice(TEXTS),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_rows = make_split(
        root,
        "train",
        args.num_train,
        args.visual_dim,
        args.speaker_dim,
        args.n_mels,
        args.upsample,
    )
    val_rows = make_split(
        root,
        "val",
        args.num_val,
        args.visual_dim,
        args.speaker_dim,
        args.n_mels,
        args.upsample,
    )
    write_manifest(train_rows, root / "train.jsonl")
    write_manifest(val_rows, root / "val.jsonl")
    print(f"wrote {root / 'train.jsonl'}")
    print(f"wrote {root / 'val.jsonl'}")


if __name__ == "__main__":
    main()
