from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV8.data.video_dataset import VideoTextDataset, collate_video_text
from srcV8.models import AVHubertVisualFeatureExtractor
from srcV8.utils import batch_to_device, get_device, seed_everything


def cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        return amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def safe_stem(path: str) -> str:
    p = Path(path)
    name = p.stem
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    out = "".join(keep).strip("._")
    return out[:140] or "sample"


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = VideoTextDataset(args.data_dir, limit_files=args.limit_files if args.limit_files > 0 else None)
    if ds.skipped:
        print(f"[data] skipped_no_text_or_bad_cache={len(ds.skipped)}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_video_text,
    )
    extractor = AVHubertVisualFeatureExtractor(
        checkpoint=args.checkpoint,
        avhubert_dir=args.avhubert_dir,
        output_layer=args.output_layer if args.output_layer > 0 else None,
        freeze=True,
        normalize_video=not args.no_normalize_video,
        normalize_mode="none" if args.no_normalize_video else args.normalize_mode,
        crop_size=args.crop_size,
    ).to(device)
    extractor.eval()

    print(f"[device] {device}")
    print(f"[data] {args.data_dir} files={len(ds)}")
    print(f"[avhubert] dir={args.avhubert_dir}")
    print(f"[checkpoint] {args.checkpoint}")
    print(f"[preprocess] crop_size={args.crop_size} normalize={extractor.normalize_mode}")
    print(f"[output] {output_dir}")

    written = 0
    amp_enabled = device.type == "cuda" and args.amp
    for batch in tqdm(loader, desc="cache-avhubert"):
        batch = batch_to_device(batch, device)
        with cuda_autocast(amp_enabled):
            features, feature_padding = extractor(batch["video"], batch["video_mask"])
        if feature_padding is None:
            feature_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            feature_mask = ~feature_padding.bool()
        for i, src_path in enumerate(batch["paths"]):
            feat_len = int(feature_mask[i].sum().detach().cpu().item())
            feat = features[i, :feat_len].detach().float().cpu()
            out_path = output_dir / f"{safe_stem(src_path)}.pt"
            suffix = 1
            while out_path.exists() and not args.overwrite:
                out_path = output_dir / f"{safe_stem(src_path)}_{suffix}.pt"
                suffix += 1
            if out_path.exists() and args.overwrite:
                out_path.unlink()
            torch.save(
                {
                    "format": "avhubert_feature_v1",
                    "features": feat,
                    "feature_len": int(feat.shape[0]),
                    "feature_dim": int(feat.shape[1]),
                    "transcript_text": batch["transcript_texts"][i],
                    "source_cache": src_path,
                    "source_video": batch["source_videos"][i],
                    "avhubert_checkpoint": str(args.checkpoint),
                    "output_layer": args.output_layer,
                    "preprocess": {
                        "crop_size": int(args.crop_size),
                        "normalize_mode": str(extractor.normalize_mode),
                    },
                },
                out_path,
            )
            written += 1
    print(f"[done] wrote={written} feature files to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache visual-only AV-HuBERT features for srcV8 CTC training.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--output-dir", default="Processed_Data_AVHubertFeatures")
    parser.add_argument("--avhubert-dir", required=True, help="Path to cloned facebookresearch/av_hubert repo.")
    parser.add_argument("--checkpoint", required=True, help="Path to AV-HuBERT pretrained/fine-tuned checkpoint .pt.")
    parser.add_argument("--output-layer", type=int, default=0, help="1-based transformer layer. 0 means final layer.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-normalize-video", action="store_true")
    parser.add_argument(
        "--normalize-mode",
        choices=["avhubert", "per_frame", "none"],
        default="avhubert",
        help="Video normalization before AV-HuBERT. Use avhubert for official VSR/pretrained checkpoints.",
    )
    parser.add_argument("--crop-size", type=int, default=88, help="Center crop size after resizing mouth ROI to 96.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
