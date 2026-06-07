from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from l2s_itw.config import apply_overrides, load_config
from l2s_itw.data import read_manifest
from l2s_itw.models import VisualTTS
from l2s_itw.text import CharTokenizer
from l2s_itw.utils import ensure_dir, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize mel with the clean Visual TTS model.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--sample", required=True, help="JSONL manifest used as input.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted.key=value.")
    return parser.parse_args()


def load_tensor(path_value: str, base_dir: Path, dict_key: str) -> torch.Tensor:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        obj = obj.get(dict_key, obj.get("tensor"))
    if not torch.is_tensor(obj):
        raise TypeError(f"Expected tensor in {path}")
    return obj.float()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config: dict[str, Any] = checkpoint.get("config", load_config(args.config))
    config = apply_overrides(config, args.set)

    seed_everything(int(config.get("seed", 1234)))
    device = resolve_device(str(config["training"].get("device", "cuda")))
    tokenizer = CharTokenizer(
        symbols=checkpoint.get("tokenizer_symbols", config["text"]["symbols"]),
        lowercase=bool(checkpoint.get("tokenizer_lowercase", config["text"].get("lowercase", True))),
    )

    model = VisualTTS(config, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    manifest_path = Path(args.sample)
    rows = read_manifest(manifest_path)
    row = rows[int(args.sample_index)]
    base_dir = manifest_path.parent

    visual = load_tensor(row[config["data"]["visual_feature_key"]], base_dir, "visual_features")
    speaker_key = config["data"]["speaker_embedding_key"]
    if speaker_key in row and row[speaker_key]:
        speaker = load_tensor(row[speaker_key], base_dir, "speaker_embedding").flatten()
    else:
        speaker = torch.zeros(int(config["model"]["speaker_dim"]), dtype=torch.float32)

    text = str(row.get(config["data"]["text_key"], ""))
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    with torch.no_grad():
        output = model(
            visuals=visual.unsqueeze(0).to(device),
            visual_lengths=torch.tensor([visual.shape[0]], dtype=torch.long, device=device),
            tokens=tokens.unsqueeze(0).to(device),
            text_lengths=torch.tensor([tokens.shape[0]], dtype=torch.long, device=device),
            speakers=speaker.unsqueeze(0).to(device),
            return_attention=True,
        )

    mel = output["mel"][0, : int(output["mel_lengths"][0].item())].detach().cpu()
    output_dir = ensure_dir(args.output_dir)
    sample_id = row.get("id", f"sample_{args.sample_index:04d}")
    torch.save({"mel": mel, "text": text, "id": sample_id}, output_dir / f"{sample_id}.mel.pt")
    np.save(output_dir / f"{sample_id}.mel.npy", mel.numpy())
    if "attention" in output:
        torch.save(output["attention"].detach().cpu(), output_dir / f"{sample_id}.attention.pt")
    print(f"wrote mel: {output_dir / f'{sample_id}.mel.pt'}")


if __name__ == "__main__":
    main()
