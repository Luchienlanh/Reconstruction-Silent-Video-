from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from l2t_arch.text import CharTokenizer


def read_manifest(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


class ArchDataset(Dataset):
    def __init__(self, manifest: str | Path, tokenizer: CharTokenizer, limit: int = 0) -> None:
        self.rows = read_manifest(manifest, limit=limit)
        self.tokenizer = tokenizer
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        obj = torch.load(row["visual_feature_path"], map_location="cpu")
        visual = obj["visual_features"] if isinstance(obj, dict) and "visual_features" in obj else obj
        visual = visual.float()
        vtp_text = str(row.get("vtp_text", ""))
        gt_text = str(row.get("gt_text", ""))
        return {
            "id": row.get("id", str(index)),
            "visual": visual,
            "vtp_tokens": torch.tensor(self.tokenizer.encode(vtp_text, add_bos=True, add_eos=True), dtype=torch.long),
            "gt_in": torch.tensor(self.tokenizer.encode(gt_text, add_bos=True), dtype=torch.long),
            "gt_out": torch.tensor(self.tokenizer.encode(gt_text, add_eos=True), dtype=torch.long),
            "ctc_target": torch.tensor(self.tokenizer.encode(gt_text), dtype=torch.long),
            "vtp_text": vtp_text,
            "gt_text": gt_text,
        }


def _pad_1d(items: list[torch.Tensor], value: int) -> torch.Tensor:
    return pad_sequence(items, batch_first=True, padding_value=value)


def collate_arch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    visuals = [item["visual"] for item in batch]
    ctc_targets = [item["ctc_target"] for item in batch]
    return {
        "ids": [item["id"] for item in batch],
        "visuals": pad_sequence(visuals, batch_first=True),
        "visual_lengths": torch.tensor([x.shape[0] for x in visuals], dtype=torch.long),
        "vtp_tokens": _pad_1d([item["vtp_tokens"] for item in batch], 0),
        "vtp_lengths": torch.tensor([item["vtp_tokens"].numel() for item in batch], dtype=torch.long),
        "gt_in": _pad_1d([item["gt_in"] for item in batch], 0),
        "gt_out": _pad_1d([item["gt_out"] for item in batch], 0),
        "gt_lengths": torch.tensor([item["gt_out"].numel() for item in batch], dtype=torch.long),
        "ctc_targets": torch.cat(ctc_targets) if ctc_targets else torch.empty(0, dtype=torch.long),
        "ctc_target_lengths": torch.tensor([x.numel() for x in ctc_targets], dtype=torch.long),
        "vtp_texts": [item["vtp_text"] for item in batch],
        "gt_texts": [item["gt_text"] for item in batch],
    }

