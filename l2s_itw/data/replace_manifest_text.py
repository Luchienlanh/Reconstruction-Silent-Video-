from __future__ import annotations

import argparse
from pathlib import Path

from l2s_itw.data.manifest import read_manifest, write_manifest


PATH_KEYS = [
    "visual_feature_path",
    "mel_path",
    "speaker_embedding_path",
    "video_path",
    "source_video_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace text in one manifest with text from another manifest matched by sample id."
    )
    parser.add_argument("--base-manifest", required=True, help="Manifest whose paths/features should be preserved.")
    parser.add_argument("--text-manifest", required=True, help="Manifest containing the replacement text.")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--keep-original-as", default="vtp_text", help="Field name for the original base text.")
    return parser.parse_args()


def rebase_path_fields(row: dict, source_dir: Path) -> dict:
    updated = dict(row)
    for key in PATH_KEYS:
        if key not in updated or not updated[key]:
            continue
        path = Path(str(updated[key]))
        if not path.is_absolute():
            updated[key] = str((source_dir / path).resolve())
    return updated


def main() -> None:
    args = parse_args()
    base_manifest = Path(args.base_manifest)
    base_rows = read_manifest(base_manifest)
    text_rows = read_manifest(args.text_manifest)
    text_by_id = {str(row[args.id_key]): str(row.get(args.text_key, "")) for row in text_rows}

    replaced = 0
    missing = []
    output_rows = []
    for row in base_rows:
        sample_id = str(row[args.id_key])
        new_row = rebase_path_fields(row, base_manifest.parent)
        if sample_id in text_by_id:
            if args.keep_original_as:
                new_row[args.keep_original_as] = str(new_row.get(args.text_key, ""))
            new_row[args.text_key] = text_by_id[sample_id]
            replaced += 1
        else:
            missing.append(sample_id)
        output_rows.append(new_row)

    write_manifest(output_rows, Path(args.output_manifest))
    print(f"wrote {args.output_manifest}")
    print(f"replaced_text: {replaced}/{len(base_rows)}")
    if missing:
        print(f"missing_text: {len(missing)}")
        print("first_missing:", ", ".join(missing[:5]))


if __name__ == "__main__":
    main()
