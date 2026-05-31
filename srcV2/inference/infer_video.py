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

from srcV2.data.build_cache import LipLandmarkExtractor, decode_video_with_mouth_crops
from srcV2.models import ContentUnitLipToSpeechModel
from srcV2.utils.audio import _torch_mel_filterbank
from srcV2.utils.common import get_device


def build_content_unit_model(config: dict, device: torch.device) -> ContentUnitLipToSpeechModel:
    model = ContentUnitLipToSpeechModel(
        dim=int(config.get("dim", 256)),
        spatial_tokens=int(config.get("spatial_tokens", 2)),
        num_points=int(config.get("num_landmark_points", 40)),
        dropout=float(config.get("dropout", 0.05)),
        encoder_layers=int(config.get("encoder_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 2)),
        heads=int(config.get("heads", 4)),
        num_units=int(config.get("num_units", 50)),
        unit_temperature=float(config.get("unit_temperature", 1.0)),
        detach_unit_condition=bool(config.get("detach_unit_condition", True)),
        detach_content_hidden=bool(config.get("detach_content_hidden", True)),
        unit_teacher_prob=0.0,
    ).to(device)
    return model


def logmel_to_audio(
    logmel: np.ndarray,
    out_wav: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
) -> None:
    wav: np.ndarray | None = None
    try:
        import librosa

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
    except Exception:
        mel_t = torch.from_numpy(np.exp(logmel).astype(np.float32)).transpose(0, 1)
        fb = _torch_mel_filterbank(sample_rate, n_fft, logmel.shape[1], mel_t.dtype, mel_t.device)
        spec_mag = torch.linalg.pinv(fb.float()).matmul(mel_t.float()).clamp_min(1e-8)
        angles = torch.rand_like(spec_mag) * (2.0 * np.pi)
        window = torch.hann_window(win_length, dtype=spec_mag.dtype)
        length = max(1, int(logmel.shape[0] * hop_length))
        complex_spec = torch.polar(spec_mag, angles)
        for _ in range(max(1, int(n_iter))):
            wav_t = torch.istft(
                complex_spec,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                length=length,
            )
            rebuilt = torch.stft(
                wav_t,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            phase = rebuilt.angle()
            if phase.shape[-1] != spec_mag.shape[-1]:
                phase = phase[:, : spec_mag.shape[-1]]
                if phase.shape[-1] < spec_mag.shape[-1]:
                    phase = torch.nn.functional.pad(phase, (0, spec_mag.shape[-1] - phase.shape[-1]))
            complex_spec = torch.polar(spec_mag, phase)
        wav = wav_t.detach().cpu().numpy()
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


def save_mel_image(logmel: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(logmel.T, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("mel frame")
    ax.set_ylabel("mel bin")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args) -> None:
    device = get_device(args.device)
    checkpoint = Path(args.checkpoint)
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(ckpt.get("config", {}))
    model_type = config.get("model_type", "content_unit")
    if model_type != "content_unit":
        raise ValueError(f"This inference script expects a content_unit checkpoint, got {model_type!r}")

    model = build_content_unit_model(config, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[checkpoint] loaded={checkpoint} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    extractor = LipLandmarkExtractor(enabled=not args.no_mediapipe, model_path=args.face_landmarker_model)
    try:
        video, landmarks, valid_mask, boxes, video_times, fps = decode_video_with_mouth_crops(
            video_path,
            frame_size=args.frame_size,
            margin=args.margin,
            extractor=extractor,
            force_fps=args.force_fps,
        )
    finally:
        extractor.close()

    duration = float(video_times[-1].item() + 0.5 / max(float(fps), 1e-6))
    mel_len = max(1, int(round(duration * args.sample_rate / args.hop_length)))
    batch = {
        "video": video.unsqueeze(0).to(device),
        "landmarks": landmarks.unsqueeze(0).to(device),
        "video_mask": torch.ones(1, video.shape[1], dtype=torch.bool, device=device),
        "mel_mask": torch.ones(1, mel_len, dtype=torch.bool, device=device),
        "return_aux": True,
    }

    with torch.inference_mode():
        outputs = model(batch)
    pred_mel = outputs["mel"][0, :mel_len].detach().float().cpu()
    unit_logits = outputs["unit_logits"][0, :mel_len].detach().float().cpu()
    unit_ids = unit_logits.argmax(dim=-1).numpy().astype(np.int64)
    pred_np = pred_mel.numpy().astype(np.float32)

    stem = video_path.stem.replace(" ", "_").replace(":", "-")
    mel_path = output_dir / f"{stem}_pred_mel.npy"
    unit_path = output_dir / f"{stem}_units.txt"
    png_path = output_dir / f"{stem}_pred_mel.png"
    wav_path = output_dir / f"{stem}_griffinlim.wav"
    meta_path = output_dir / f"{stem}_meta.json"

    np.save(mel_path, pred_np)
    unit_path.write_text(" ".join(str(int(x)) for x in unit_ids.tolist()) + "\n", encoding="utf-8")
    save_mel_image(pred_np, png_path, title=f"Predicted log-mel: {video_path.name}")
    logmel_to_audio(
        pred_np,
        wav_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_iter=args.griffinlim_iters,
    )

    meta = {
        "video": str(video_path),
        "checkpoint": str(checkpoint),
        "frames": int(video.shape[1]),
        "fps": float(fps),
        "duration": duration,
        "mel_frames": int(mel_len),
        "valid_ratio": float(valid_mask.float().mean().item()),
        "pred_mel_mean": float(pred_mel.mean().item()),
        "pred_mel_std": float(pred_mel.std().item()),
        "unit_unique": int(np.unique(unit_ids).size),
        "outputs": {
            "mel_npy": str(mel_path),
            "units_txt": str(unit_path),
            "mel_png": str(png_path),
            "wav": str(wav_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run ContentUnit lip-to-speech inference on one silent video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="inference_outputs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--margin", type=float, default=1.8)
    parser.add_argument("--force-fps", type=float, default=0.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--griffinlim-iters", type=int, default=64)
    parser.add_argument("--no-mediapipe", action="store_true")
    parser.add_argument("--face-landmarker-model", default="face_landmarker_v2_with_blendshapes.task")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
