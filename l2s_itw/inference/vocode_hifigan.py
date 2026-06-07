from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml

from l2s_itw.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a mel-spectrogram .pt/.npy file to wav with SpeechBrain HiFi-GAN.")
    parser.add_argument("--mel", help="Path to mel .pt or .npy. .pt can contain a 'mel' tensor.")
    parser.add_argument("--manifest", help="JSONL manifest. Used to load the row's mel_path when --mel is omitted.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--hifigan-dir", default="pretrained_models/tts-hifigan-libritts-16kHz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def mel_path_from_manifest(path: str | Path, sample_index: int) -> Path:
    with Path(path).open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == sample_index:
                row = json.loads(line)
                if "mel_path" not in row:
                    raise KeyError(f"Manifest row {sample_index} has no mel_path")
                return Path(row["mel_path"])
    raise IndexError(f"sample-index {sample_index} is out of range for {path}")


def load_mel(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        import numpy as np

        mel = torch.from_numpy(np.load(path))
    else:
        obj = torch.load(path, map_location="cpu")
        mel = obj["mel"] if isinstance(obj, dict) else obj
    if mel.ndim != 2:
        raise ValueError(f"Expected mel [T, 80] or [80, T], got {tuple(mel.shape)}")
    if mel.shape[0] == 80 and mel.shape[1] != 80:
        mel = mel.transpose(0, 1)
    return mel.float()


def load_generator(hifigan_dir: str | Path, device: torch.device) -> torch.nn.Module:
    hifigan_dir = Path(hifigan_dir)
    with (hifigan_dir / "hyperparams.yaml").open("r", encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)
    generator = hparams["generator"].to(device)
    state_dict = torch.load(hifigan_dir / "generator.ckpt", map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def main() -> None:
    args = parse_args()
    if not args.mel and not args.manifest:
        raise ValueError("Provide either --mel or --manifest.")
    mel_path = Path(args.mel) if args.mel else mel_path_from_manifest(args.manifest, args.sample_index)
    device = resolve_device(args.device)
    mel = load_mel(mel_path).to(device)
    generator = load_generator(args.hifigan_dir, device)

    with torch.no_grad():
        wav = generator(mel.transpose(0, 1).unsqueeze(0)).squeeze().detach().cpu()

    wav = wav.clamp(-1.0, 1.0).numpy()
    output = Path(args.output)
    ensure_dir(output.parent)
    sf.write(output, wav, int(args.sample_rate))
    print(f"wrote wav: {output}")


if __name__ == "__main__":
    main()
