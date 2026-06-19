from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_manifest(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_lrs2_text(path: str | Path) -> str:
    text_path = Path(path)
    lines = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("text:"):
            return stripped.split(":", 1)[1].strip()
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def sample_seconds(row: dict[str, Any]) -> float | None:
    mel_frames = row.get("mel_frames")
    hop_length = row.get("hop_length")
    sample_rate = row.get("sample_rate")
    if mel_frames and hop_length and sample_rate:
        return float(mel_frames) * float(hop_length) / float(sample_rate)
    return None

