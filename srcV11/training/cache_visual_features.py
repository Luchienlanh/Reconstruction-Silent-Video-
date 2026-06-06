from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from srcV11.data import normalize_english
from srcV11.utils import seed_everything


def torch_load_cpu(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def load_r2inr_cache(path: str | Path) -> dict[str, Any]:
    item = torch_load_cpu(path)
    if item.get("format") != "r2inr_v1":
        raise ValueError(f"{path} is not an r2inr_v1 cache file.")
    if "video" not in item:
        raise ValueError(f"{path} has no video tensor.")
    return item


def video_to_tchw(video: torch.Tensor) -> torch.Tensor:
    video = video.float()
    if video.ndim == 3:
        return video.unsqueeze(1)
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with 3 or 4 dims, got {tuple(video.shape)}")
    if video.shape[0] in {1, 3}:
        return video.permute(1, 0, 2, 3).contiguous()
    if video.shape[1] in {1, 3}:
        return video.contiguous()
    raise ValueError(f"Cannot infer video layout from shape {tuple(video.shape)}")


def match_length(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    length = int(length)
    if x.shape[0] == length:
        return x
    if x.shape[0] > length:
        return x[:length]
    pad_shape = (length - x.shape[0], *x.shape[1:])
    pad = torch.full(pad_shape, float(value), dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def make_visual_features(
    item: dict[str, Any],
    mouth_size: int = 32,
    include_landmarks: bool = True,
    include_valid_flag: bool = True,
    pixel_mean: float = 0.5,
    pixel_std: float = 0.5,
) -> torch.Tensor:
    frames = video_to_tchw(item["video"])
    video_len = int(item.get("video_len", frames.shape[0]))
    frames = match_length(frames, video_len)
    small = F.interpolate(
        frames,
        size=(int(mouth_size), int(mouth_size)),
        mode="bilinear",
        align_corners=False,
    )
    pixels = (small - float(pixel_mean)) / max(float(pixel_std), 1e-6)
    parts = [pixels.flatten(1)]

    if include_landmarks and "landmarks" in item:
        landmarks = item["landmarks"].float()
        landmarks = match_length(landmarks, video_len)
        parts.append(torch.nan_to_num(landmarks.flatten(1), nan=0.0, posinf=0.0, neginf=0.0))

    if include_valid_flag:
        valid = item.get("mouth_valid_mask")
        if valid is None:
            valid = torch.ones(video_len, dtype=torch.float32)
        else:
            valid = match_length(valid.float().reshape(-1), video_len)
        parts.append(valid.float().unsqueeze(-1))

    features = torch.cat(parts, dim=-1).contiguous()
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def process_file(path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    item = load_r2inr_cache(path)
    transcript = str(item.get("transcript_text", "")).strip()
    normalized = normalize_english(transcript)
    if not normalized and not args.allow_empty_text:
        return {"file": str(path), "status": "skipped", "reason": "empty_transcript"}

    features = make_visual_features(
        item,
        mouth_size=args.mouth_size,
        include_landmarks=not args.no_landmarks,
        include_valid_flag=not args.no_valid_flag,
        pixel_mean=args.pixel_mean,
        pixel_std=args.pixel_std,
    )
    out_path = output_dir / f"{safe_stem(path)}.pt"
    suffix = 1
    while out_path.exists() and not args.overwrite:
        out_path = output_dir / f"{safe_stem(path)}_{suffix}.pt"
        suffix += 1
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    payload = {
        "format": "avhubert_feature_v1",
        "feature_source": "srcV11_visual_frame_v1",
        "features": features.cpu(),
        "feature_len": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "transcript_text": transcript,
        "source_cache": str(path),
        "source_video": str(item.get("source_video", "")),
        "source_text": str(item.get("source_text", "")),
        "split": str(item.get("split", "")),
        "transcript_meta": item.get("transcript_meta", {}),
        "visual_feature_config": {
            "mouth_size": int(args.mouth_size),
            "include_landmarks": not args.no_landmarks,
            "include_valid_flag": not args.no_valid_flag,
            "pixel_mean": float(args.pixel_mean),
            "pixel_std": float(args.pixel_std),
        },
    }
    torch.save(payload, out_path)
    return {
        "file": str(out_path),
        "source_cache": str(path),
        "feature_len": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "transcript_chars": len(normalized),
        "status": "ok",
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    files = split_files(args.data_dir, args.limit_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] r2inr_files={len(files)} data_dir={args.data_dir}")
    print(f"[features] mouth_size={args.mouth_size} landmarks={not args.no_landmarks} valid_flag={not args.no_valid_flag}")
    print(f"[output] {output_dir}")

    items = []
    for path in tqdm(files, desc="cache-visual-features"):
        try:
            items.append(process_file(path, output_dir, args))
        except Exception as exc:
            items.append({"file": str(path), "status": "failed", "reason": repr(exc)})
            print(f"[fail] {path}: {exc}")

    ok = [item for item in items if item.get("status") == "ok"]
    skipped = [item for item in items if item.get("status") == "skipped"]
    failed = [item for item in items if item.get("status") == "failed"]
    summary = {
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "total": len(items),
        "ok": len(ok),
        "skipped": len(skipped),
        "failed": len(failed),
        "config": vars(args),
        "items": items,
    }
    (output_dir / "visual_feature_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] ok={len(ok)} skipped={len(skipped)} failed={len(failed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache deterministic visual frame features from r2inr_v1 mouth crops for srcV11 CTC.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2_10k")
    parser.add_argument("--output-dir", default="Processed_Data_VisualFeatures_LRS2_10k")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--mouth-size", type=int, default=32)
    parser.add_argument("--pixel-mean", type=float, default=0.5)
    parser.add_argument("--pixel-std", type=float, default=0.5)
    parser.add_argument("--no-landmarks", action="store_true")
    parser.add_argument("--no-valid-flag", action="store_true")
    parser.add_argument("--allow-empty-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
