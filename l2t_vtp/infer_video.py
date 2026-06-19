from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from l2t_vtp.providers import VTPProvider, VTPProviderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone VTP lip-to-text on one video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--repo-dir", default="external/vtp")
    parser.add_argument("--ckpt-path", default="pretrained_models/vtp/ft_lrs2.pth")
    parser.add_argument("--cnn-ckpt-path", default="pretrained_models/vtp/feature_extractor.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--max-decode-len", type=int, default=35)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--save-visual-features", default="", help="Optional .pt output for VTP visual features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = VTPProvider(
        VTPProviderConfig(
            repo_dir=args.repo_dir,
            ckpt_path=args.ckpt_path,
            cnn_ckpt_path=args.cnn_ckpt_path,
            device=args.device,
            beam_size=int(args.beam_size),
            max_decode_len=int(args.max_decode_len),
        )
    )
    result = provider.run(args.video)
    print(result.text)

    if args.save_visual_features:
        output = Path(args.save_visual_features)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"visual_features": result.visual_features, "text": result.text, "video_path": str(Path(args.video).resolve())}, output)

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "video_path": str(Path(args.video).resolve()),
                    "text": result.text,
                    "visual_frames": int(result.visual_features.shape[0]),
                    "visual_dim": int(result.visual_features.shape[1]) if result.visual_features.ndim == 2 else 0,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

