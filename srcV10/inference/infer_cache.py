from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from srcV10.data import V10R2INRDataset, collate_v10, split_cache_files
from srcV10.models import build_model_from_config
from srcV10.utils import batch_to_device, get_device


def logmel_to_audio(
    logmel: np.ndarray,
    out_wav: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
) -> None:
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("librosa is required for Griffin-Lim wav export in srcV10 inference.") from exc

    mel = np.exp(logmel.T).astype(np.float32)
    wav = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,
        n_iter=n_iter,
    )
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1e-6:
        wav = 0.95 * wav / peak
    wav = wav.astype(np.float32)
    try:
        import soundfile as sf

        sf.write(str(out_wav), wav, sample_rate)
    except Exception:
        from scipy.io import wavfile

        wavfile.write(str(out_wav), sample_rate, np.clip(wav, -1.0, 1.0))


def save_plot(pred: torch.Tensor, target: torch.Tensor, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].imshow(target.T.numpy(), aspect="auto", origin="lower")
    axes[0].set_title("Target log-mel")
    axes[1].imshow(pred.T.numpy(), aspect="auto", origin="lower")
    axes[1].set_title("Predicted log-mel")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    if args.av_feature_dir:
        config["av_feature_dir"] = args.av_feature_dir
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    if args.cache_file:
        files = [Path(args.cache_file)]
    else:
        train_files, val_files = split_cache_files(
            args.data_dir,
            args.val_ratio,
            int(config.get("seed", 42)),
            args.limit_files or None,
        )
        files = val_files if args.split == "val" and val_files else train_files
    sample = files[max(0, min(args.sample_index, len(files) - 1))]
    av_feature_dir = args.av_feature_dir or config.get("av_feature_dir", "")
    ds = V10R2INRDataset(
        args.data_dir,
        files=[sample],
        max_frames=args.max_frames,
        random_crop=False,
        av_feature_dir=av_feature_dir if config.get("use_avhubert_features", False) else None,
        require_av_features=False,
    )
    batch = batch_to_device(next(iter(DataLoader(ds, batch_size=1, collate_fn=collate_v10))), device)
    batch["return_aux"] = True
    if args.sample_flow:
        batch["sample_flow"] = True
        batch["flow_steps"] = args.flow_steps
    out = model(batch)
    pred = out["mel"] if isinstance(out, dict) else out
    length = int(batch["mel_lengths"][0].item())
    pred_mel = pred[0, :length].detach().float().cpu()
    target = batch["mel"][0, :length].detach().float().cpu()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sample.stem
    np.save(output_dir / f"{stem}_pred_mel.npy", pred_mel.numpy().astype(np.float32))
    torch.save(
        {
            "pred_mel": pred_mel,
            "target_mel": target,
            "cache_file": str(sample),
            "av_feature_path": (batch.get("av_feature_paths") or [""])[0],
        },
        output_dir / f"{stem}_debug.pt",
    )
    save_plot(pred_mel, target, output_dir / f"{stem}_compare.png")
    item = torch.load(sample, map_location="cpu", weights_only=False)
    sample_rate = int(item.get("sample_rate", args.sample_rate))
    hop_length = int(item.get("hop_length", args.hop_length))
    logmel_to_audio(
        pred_mel.numpy().astype(np.float32),
        output_dir / f"{stem}_griffinlim.wav",
        sample_rate=sample_rate,
        n_fft=args.n_fft,
        hop_length=hop_length,
        win_length=args.win_length,
        n_iter=args.griffinlim_iters,
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "cache_file": str(sample),
        "av_feature_path": (batch.get("av_feature_paths") or [""])[0],
        "mel_frames": int(length),
        "pred_std": float(pred_mel.std(unbiased=False).item()),
        "target_std": float(target.std(unbiased=False).item()),
    }
    (output_dir / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer one srcV10 R2INR cache sample.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2_10k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="infer_srcV10")
    parser.add_argument("--av-feature-dir", default="")
    parser.add_argument("--cache-file", default="")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--sample-flow", action="store_true")
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--griffinlim-iters", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
