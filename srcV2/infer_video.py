from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from srcV2.data.build_cache import LipLandmarkExtractor, decode_video_with_mouth_crops
from srcV2.data.dataset import _load_cache, _mouth_motion_features, extract_window, window_starts
from srcV2.models import R2INRModel
from srcV2.training.common import load_checkpoint, model_inputs
from srcV2.utils.common import batch_to_device, get_device


def load_config(ckpt: dict, args: argparse.Namespace) -> SimpleNamespace:
    cfg = ckpt.get("config") or {}
    motion_weight = ckpt.get("model_state_dict", {}).get("encoder.motion.0.weight")
    motion_dim = int(motion_weight.shape[1]) if torch.is_tensor(motion_weight) else int(cfg.get("motion_dim", 19))
    return SimpleNamespace(
        dim=int(cfg.get("dim", 512)),
        spatial_tokens=int(cfg.get("spatial_tokens", 4)),
        num_landmark_points=int(cfg.get("num_landmark_points", 40)),
        dropout=float(cfg.get("dropout", 0.0)),
        motion_dim=motion_dim,
        multi_gpu=False,
    )


def load_model(checkpoint: Path, device: torch.device, args: argparse.Namespace) -> tuple[R2INRModel, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = load_config(ckpt, args)
    model = R2INRModel(
        dim=cfg.dim,
        spatial_tokens=cfg.spatial_tokens,
        num_points=cfg.num_landmark_points,
        dropout=cfg.dropout,
        motion_dim=cfg.motion_dim,
    ).to(device)
    load_checkpoint(checkpoint, model, device)
    model.eval()
    return model, ckpt


def video_to_item(video_path: Path, args: argparse.Namespace) -> dict:
    extractor = LipLandmarkExtractor(enabled=not args.no_mediapipe, model_path=args.face_landmarker_model)
    try:
        video, landmarks, valid_mask, crop_boxes, video_times, fps = decode_video_with_mouth_crops(
            video_path,
            frame_size=args.frame_size,
            margin=args.margin,
            extractor=extractor,
            force_fps=args.force_fps,
        )
    finally:
        extractor.close()
    duration = float(video_times[-1].item()) + 0.5 / max(float(fps), 1e-6)
    mel_len = max(1, int(round(duration * args.sample_rate / args.hop_length)))
    mel_times = (torch.arange(mel_len, dtype=torch.float32) + 0.5) * (args.hop_length / args.sample_rate)
    return {
        "format": "r2inr_v1",
        "video": video.contiguous(),
        "landmarks": landmarks.contiguous(),
        "mel": torch.zeros(mel_len, args.n_mels),
        "video_len": int(video.shape[1]),
        "mel_len": int(mel_len),
        "fps": float(fps),
        "sample_rate": int(args.sample_rate),
        "hop_length": int(args.hop_length),
        "video_times": video_times.contiguous(),
        "mel_times": mel_times.contiguous(),
        "mouth_valid_mask": valid_mask.contiguous(),
        "crop_boxes": crop_boxes.contiguous(),
        "source_video": str(video_path),
    }


def single_batch_from_window(win: dict) -> dict:
    video_len = int(win["video_len"])
    mel_len = int(win["mel_len"])
    return {
        "video": win["video"].unsqueeze(0),
        "landmarks": win["landmarks"].unsqueeze(0),
        "mel": win["mel"].unsqueeze(0),
        "video_times": win["video_times"].unsqueeze(0),
        "mel_times": win["mel_times"].unsqueeze(0),
        "mouth_valid_mask": win["mouth_valid_mask"].unsqueeze(0),
        "mouth_motion": win.get("mouth_motion", _mouth_motion_features(win["landmarks"])).unsqueeze(0),
        "video_mask": torch.ones(1, video_len, dtype=torch.bool),
        "mel_mask": torch.ones(1, mel_len, dtype=torch.bool),
        "video_lengths": torch.tensor([video_len], dtype=torch.long),
        "mel_lengths": torch.tensor([mel_len], dtype=torch.long),
    }


def blend_weights(length: int, device: torch.device) -> torch.Tensor:
    if length <= 2:
        return torch.ones(length, 1, device=device)
    return torch.hann_window(length, periodic=False, device=device).view(length, 1).clamp_min(0.05)


@torch.no_grad()
def predict_mel(model: R2INRModel, item: dict, device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    starts = window_starts(int(item["video_len"]), args.window_frames, args.hop_frames)
    full_len = int(item["mel_len"])
    out = torch.zeros(full_len, args.n_mels, device=device)
    weight = torch.zeros(full_len, 1, device=device)
    source = Path(item.get("source_video") or "input")
    for start in starts:
        win = extract_window(source, item, start, args.window_frames)
        mel_idx = win["mel_indices"].to(device)
        if mel_idx.numel() <= 0:
            continue
        batch = batch_to_device(single_batch_from_window(win), device)
        pred = model(model_inputs(batch)).float()[0]
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
    missing = weight.squeeze(-1) <= 0
    pred_mel = out / weight.clamp_min(1e-6)
    if bool(missing.any()):
        pred_mel[missing] = pred_mel[~missing].mean(dim=0, keepdim=True) if bool((~missing).any()) else 0.0
    return pred_mel.cpu()


def save_plot(pred_mel: torch.Tensor, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(pred_mel.float().T.numpy(), aspect="auto", origin="lower")
    ax.set_title("srcV2 predicted mel")
    ax.set_xlabel("Mel frame")
    ax.set_ylabel("Mel bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def safe_stem(value: str) -> str:
    text = str(value).replace("\\", "/")
    stem = Path(text).stem or "sample"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem[:160] or "sample"


def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    checkpoint = Path(args.checkpoint)
    model, ckpt = load_model(checkpoint, device, args)
    if args.sample_cache:
        item = _load_cache(args.sample_cache)
    elif args.video:
        item = video_to_item(Path(args.video), args)
    else:
        raise ValueError("Provide either --video or --sample-cache.")

    pred_mel = predict_mel(model, item, device, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(item.get("source_video") or args.sample_cache or args.video or "sample")
    out_pt = output_dir / f"{stem}_srcV2_pred_mel.pt"
    out_png = output_dir / f"{stem}_srcV2_pred_mel.png"
    torch.save(
        {
            "pred_mel": pred_mel,
            "source": item.get("source_video", args.sample_cache or args.video),
            "checkpoint": str(checkpoint),
            "config": ckpt.get("config", {}),
        },
        out_pt,
    )
    save_plot(pred_mel, out_png)
    print(f"[device] {device}")
    print(f"[checkpoint] {checkpoint}")
    print(f"[frames] video={int(item['video_len'])} mel={tuple(pred_mel.shape)}")
    print(f"[saved] {out_pt}")
    print(f"[saved] {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer srcV2 mel from a silent video or r2inr cache sample.")
    parser.add_argument("--checkpoint", default="../trainning_output/best_model.pth")
    parser.add_argument("--video", default="")
    parser.add_argument("--sample-cache", default="")
    parser.add_argument("--output-dir", default="../test_output_srcV2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--hop-frames", type=int, default=10)
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--margin", type=float, default=1.8)
    parser.add_argument("--force-fps", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--face-landmarker-model", default="face_landmarker_v2_with_blendshapes.task")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
