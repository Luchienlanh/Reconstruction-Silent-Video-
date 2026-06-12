from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from l2s_itw.data.manifest import read_manifest, write_manifest
from l2s_itw.utils import ensure_dir


PATH_KEYS = [
    "visual_feature_path",
    "speaker_embedding_path",
    "video_path",
    "source_video_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize cached log-mels to the HiFi-GAN-compatible per-sample max scale.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mel-key", default="mel_path")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_row_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else manifest_dir / path


def rebase_path_fields(row: dict, source_dir: Path) -> dict:
    updated = dict(row)
    for key in PATH_KEYS:
        if key not in updated or not updated[key]:
            continue
        path = Path(str(updated[key]))
        if not path.is_absolute():
            updated[key] = str((source_dir / path).resolve())
    return updated


def normalize_log_mel(mel: torch.Tensor) -> torch.Tensor:
    linear = torch.exp(mel.float())
    linear = linear / linear.amax().clamp_min(1e-5)
    return torch.log(linear.clamp_min(1e-5))


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    manifest_dir = input_manifest.parent
    rows = read_manifest(input_manifest, limit=args.limit)
    output_dir = ensure_dir(args.output_dir)
    mel_dir = ensure_dir(output_dir / "mels")
    output_rows = []

    for index, row in enumerate(tqdm(rows, desc="normalize mels")):
        if args.mel_key not in row:
            raise KeyError(f"Missing '{args.mel_key}' in manifest row {index}")
        sample_id = str(row.get("id") or f"sample_{index:06d}")
        source_mel_path = resolve_row_path(str(row[args.mel_key]), manifest_dir)
        target_mel_path = mel_dir / f"{sample_id}.mel.pt"

        if args.overwrite or not target_mel_path.exists():
            obj = torch.load(source_mel_path, map_location="cpu")
            mel = obj["mel"] if isinstance(obj, dict) and "mel" in obj else obj
            normalized = normalize_log_mel(mel)
            if isinstance(obj, dict):
                out_obj = dict(obj)
                out_obj["mel"] = normalized
                out_obj["normalize_mel"] = True
                out_obj["normalized_from"] = str(source_mel_path.resolve())
            else:
                out_obj = {"mel": normalized, "normalize_mel": True, "normalized_from": str(source_mel_path.resolve())}
            torch.save(out_obj, target_mel_path)

        new_row = rebase_path_fields(row, manifest_dir)
        new_row[args.mel_key] = str(target_mel_path.relative_to(output_dir))
        output_rows.append(new_row)

    write_manifest(output_rows, output_dir / "manifest.jsonl")
    print(f"wrote {output_dir / 'manifest.jsonl'}")
    print(f"rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
