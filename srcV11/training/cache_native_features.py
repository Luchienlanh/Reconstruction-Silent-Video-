from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV2.data.dataset import R2INRDataset, collate_r2inr
from srcV2.models.r2plus1d_inr import R2INRMemoryEncoder
from srcV11.utils import batch_to_device, get_device, seed_everything


def split_files(data_dir: str | Path, limit_files: int = 0) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit_files and int(limit_files) > 0:
        files = files[: max(1, min(int(limit_files), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt cache files found under {data_dir}")
    return files


def safe_stem(path: str | Path) -> str:
    name = Path(str(path)).stem
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in name]
    out = "".join(keep).strip("._")
    return out[:140] or "sample"


def make_loader(args: argparse.Namespace, files: list[Path]) -> DataLoader:
    ds = R2INRDataset(
        args.data_dir,
        files=files,
        max_frames=args.max_frames,
        random_crop=False,
        seed=args.seed,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r2inr,
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device)
    files = split_files(args.data_dir, args.limit_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(args, files)
    encoder = R2INRMemoryEncoder(
        dim=args.dim,
        spatial_tokens=args.spatial_tokens,
        num_points=args.num_landmark_points,
        dropout=0.0,
    ).to(device)
    encoder.eval()

    print(f"[device] {device}")
    print(f"[data] files={len(files)} data_dir={args.data_dir}")
    print(f"[model] native R2INRMemoryEncoder dim={args.dim} spatial_tokens={args.spatial_tokens}")
    print(f"[output] {output_dir}")

    written = 0
    amp_enabled = device.type == "cuda" and args.amp
    for batch in tqdm(loader, desc="cache-native-v11"):
        batch = batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            encoded = encoder(batch)
            features = encoded["frame_memory"].detach().float()
        masks = batch["video_mask"].detach().bool()
        for i, src_path in enumerate(batch["paths"]):
            item = torch.load(src_path, map_location="cpu", weights_only=False)
            feat_len = int(masks[i].sum().cpu().item())
            feat = features[i, :feat_len].cpu()
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
                    "feature_source": "srcV11_native_r2inr_encoder",
                    "features": feat,
                    "feature_len": int(feat.shape[0]),
                    "feature_dim": int(feat.shape[1]),
                    "transcript_text": str(item.get("transcript_text", "")),
                    "source_cache": src_path,
                    "source_video": str(item.get("source_video", "")),
                    "native_config": {
                        "dim": int(args.dim),
                        "spatial_tokens": int(args.spatial_tokens),
                        "num_landmark_points": int(args.num_landmark_points),
                    },
                },
                out_path,
            )
            written += 1
    print(f"[done] wrote={written} feature files to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache srcV11 native video/landmark features without AV-HuBERT/fairseq.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2_10k")
    parser.add_argument("--output-dir", default="Processed_Data_NativeFeatures_LRS2_10k")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-tokens", type=int, default=4)
    parser.add_argument("--num-landmark-points", type=int, default=40)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
