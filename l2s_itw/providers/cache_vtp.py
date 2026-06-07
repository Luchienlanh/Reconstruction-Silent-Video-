from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from l2s_itw.data.manifest import read_manifest
from l2s_itw.providers.vtp import VTPProvider, VTPProviderConfig
from l2s_itw.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache full VTP text + visual features into l2s_itw manifests.")
    parser.add_argument("--input-manifest", required=True, help="JSONL rows with a video_path field.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", required=True, help="Path to cloned prajwalkr/vtp repo.")
    parser.add_argument("--ckpt-path", required=True, help="VTP fine-tuned lip-reading checkpoint.")
    parser.add_argument("--cnn-ckpt-path", required=True, help="VTP feature extractor checkpoint.")
    parser.add_argument("--video-key", default="video_path")
    parser.add_argument("--builder", default="vtp24x24")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-size", type=int, default=30)
    parser.add_argument("--max-decode-len", type=int, default=35)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_id(row: dict, index: int, video_path: Path) -> str:
    if row.get("id"):
        return str(row["id"])
    return f"{video_path.stem}_{index:06d}"


def resolve_row_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else manifest_dir / path


def rebase_optional_path(row: dict, key: str, source_dir: Path) -> None:
    if key not in row or not row[key]:
        return
    path = Path(str(row[key]))
    row[key] = str(path if path.is_absolute() else (source_dir / path).resolve())


def main() -> None:
    args = parse_args()
    input_manifest = Path(args.input_manifest)
    manifest_dir = input_manifest.parent
    rows = read_manifest(input_manifest, limit=args.limit)

    output_dir = ensure_dir(args.output_dir)
    feature_dir = ensure_dir(output_dir / "visual_features")
    text_dir = ensure_dir(output_dir / "texts")
    output_manifest = output_dir / "manifest.jsonl"

    provider = VTPProvider(
        VTPProviderConfig(
            repo_dir=args.repo_dir,
            ckpt_path=args.ckpt_path,
            cnn_ckpt_path=args.cnn_ckpt_path,
            builder=args.builder,
            device=args.device,
            beam_size=args.beam_size,
            max_decode_len=args.max_decode_len,
            chunk_size=args.chunk_size,
        )
    )

    out_rows = []
    for index, row in enumerate(tqdm(rows, desc="cache VTP")):
        result_text = ""
        if args.video_key not in row:
            raise KeyError(f"Missing '{args.video_key}' in manifest row {index}")
        video_path = resolve_row_path(str(row[args.video_key]), manifest_dir)
        sample_id = make_id(row, index, video_path)
        feature_path = feature_dir / f"{sample_id}.visual.pt"
        text_path = text_dir / f"{sample_id}.txt"

        if args.overwrite or not feature_path.exists() or not text_path.exists():
            result = provider.run(video_path)
            torch.save(
                {
                    "visual_features": result.visual_features,
                    "provider": "vtp",
                    "video_path": str(video_path),
                    "text": result.text,
                },
                feature_path,
            )
            text_path.write_text(result.text + "\n", encoding="utf-8")
            result_text = result.text
        else:
            cached = torch.load(feature_path, map_location="cpu")
            result_text = str(cached.get("text") or text_path.read_text(encoding="utf-8").strip())

        new_row = dict(row)
        rebase_optional_path(new_row, "mel_path", manifest_dir)
        rebase_optional_path(new_row, "speaker_embedding_path", manifest_dir)
        new_row["id"] = sample_id
        new_row["source_video_path"] = str(video_path)
        new_row["visual_feature_path"] = str(feature_path.relative_to(output_dir))
        new_row["text"] = result_text
        out_rows.append(new_row)

    with output_manifest.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {output_manifest}")


if __name__ == "__main__":
    main()
