from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from srcV4.data.window_dataset import extract_window, load_cache, window_starts
from srcV4.models import V4SpeechModel
from srcV4.training.train_windows import parse_layers
from srcV4.utils import batch_to_device, get_device


def _default_config() -> dict:
    return {
        "dim": 512,
        "num_landmark_points": 40,
        "fusion_type": "landmark_first",
        "encoder_width": 32,
        "resnet_layers": (1, 1, 1, 1),
        "visual_temporal_layers": 1,
        "landmark_temporal_layers": 1,
        "decoder_layers": 6,
        "dropout": 0.15,
        "use_snn": False,
        "snn_layers": 2,
        "snn_tau": 2.0,
        "siren_layers": 2,
        "siren_omega": 20.0,
        "visual_encoder_type": "r2plus1d",
    }


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> V4SpeechModel:
    cfg = _default_config()
    cfg.update(ckpt.get("config") or {})
    
    layers = cfg.get("resnet_layers", (1, 1, 1, 1))
    if isinstance(layers, str):
        layers = parse_layers(layers)
        
    mel_mean = ckpt.get("mel_mean")
    if not torch.is_tensor(mel_mean):
        mel_mean = torch.tensor([-4.0])
        
    model = V4SpeechModel(
        dim=int(cfg.get("dim", 512)),
        num_landmark_points=int(cfg.get("num_landmark_points", 40)),
        fusion_type=str(cfg.get("fusion_type", "landmark_first")),
        encoder_width=int(cfg.get("encoder_width", 32)),
        resnet_layers=tuple(layers),
        visual_temporal_layers=int(cfg.get("visual_temporal_layers", 1)),
        landmark_temporal_layers=int(cfg.get("landmark_temporal_layers", 1)),
        decoder_layers=int(cfg.get("decoder_layers", 6)),
        dropout=float(cfg.get("dropout", 0.15)),
        output_bias_init=float(mel_mean.float().mean().item()),
        use_snn=bool(cfg.get("use_snn", False)),
        snn_layers=int(cfg.get("snn_layers", 2)),
        snn_tau=float(cfg.get("snn_tau", 2.0)),
        siren_layers=int(cfg.get("siren_layers", 2)),
        siren_omega=float(cfg.get("siren_omega", 20.0)),
        visual_encoder_type=str(cfg.get("visual_encoder_type", "r2plus1d")),
    ).to(device)
    
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[checkpoint] missing sample:", missing[:8])
    if unexpected:
        print("[checkpoint] unexpected sample:", unexpected[:8])
        
    return model


def single_window_batch(win: dict) -> dict:
    video = win["video"].unsqueeze(0)
    landmarks = win["landmarks"].unsqueeze(0)
    mel_len = int(win["mel_len"])
    video_len = int(win["video_len"])
    return {
        "video": video,
        "landmarks": landmarks,
        "video_times": win["video_times"].unsqueeze(0),
        "mel_times": win["mel_times"].unsqueeze(0),
        "mouth_valid_mask": win["mouth_valid_mask"].unsqueeze(0),
        "video_mask": torch.ones(1, video_len, dtype=torch.bool),
        "mel_mask": torch.ones(1, mel_len, dtype=torch.bool),
        "mel": win["mel"].unsqueeze(0),
    }


def blend_weights(length: int, device: torch.device) -> torch.Tensor:
    if length <= 2:
        return torch.ones(length, 1, device=device)
    w = torch.hann_window(length, periodic=False, device=device).view(length, 1)
    return w.clamp_min(0.05)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(ckpt, device)
    model.eval()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.pt"))
    if args.sample_path:
        path = Path(args.sample_path)
    else:
        if not files:
            raise RuntimeError(f"No .pt files found under {data_dir}")
        path = files[int(args.sample_index)]
        
    item = load_cache(path)
    starts = window_starts(int(item["video_len"]), args.window_frames, args.hop_frames)
    full_len = int(item["mel_len"])
    out = torch.zeros(full_len, 80, device=device)
    weight = torch.zeros(full_len, 1, device=device)

    for start in starts:
        win = extract_window(path, item, start, args.window_frames)
        mel_idx = win["mel_indices"].to(device)
        if mel_idx.numel() <= 0:
            continue
        batch = batch_to_device(single_window_batch(win), device)
        pred = model(batch, target_len=int(mel_idx.numel()))[0].float()
        
        if pred.shape[0] != mel_idx.numel():
            pred = F.interpolate(
                pred.transpose(0, 1).unsqueeze(0),
                size=int(mel_idx.numel()),
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
            
        w = blend_weights(pred.shape[0], device)
        out[mel_idx] += pred * w
        weight[mel_idx] += w

    pred_mel = out / weight.clamp_min(1e-6)
    missing = weight.squeeze(-1) <= 0
    if bool(missing.any()):
        pred_mel[missing] = ckpt["mel_mean"].to(device).view(1, -1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pred_mel": pred_mel.cpu(),
        "target_mel": item["mel"].float().cpu(),
        "path": str(path),
        "checkpoint": str(args.checkpoint),
        "window_frames": int(args.window_frames),
        "hop_frames": int(args.hop_frames),
    }
    out_path = output_dir / f"{path.stem}_pred_mel.pt"
    torch.save(payload, out_path)
    print(f"[saved] {out_path}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].imshow(item["mel"].float().T.numpy(), aspect="auto", origin="lower")
        axes[0].set_title("target")
        axes[1].imshow(pred_mel.cpu().T.numpy(), aspect="auto", origin="lower")
        axes[1].set_title("srcV4 overlap-add")
        fig.tight_layout()
        png_path = output_dir / f"{path.stem}_pred_vs_target.png"
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        print(f"[saved] {png_path}")
    except Exception as exc:
        print(f"[plot] skipped: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run srcV4 window model with overlap-add.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="inference_srcV4_ola")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-path", default="")
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--hop-frames", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
