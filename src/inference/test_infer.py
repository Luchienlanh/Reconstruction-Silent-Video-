"""
Quick inference/smoke test for the modular lip-to-mel pipeline.

Examples:
    python src/inference/test_infer.py --smoke-only
    python src/inference/test_infer.py
    python src/inference/test_infer.py --checkpoint ephrat_baseline_checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VNLipDatasetV2, collate_pad_v2  # noqa: E402
from models.decoders.siren import TFiLMSIRENDecoder  # noqa: E402
from models.decoders.upsample import MelTemporalUpsampleDecoder  # noqa: E402
from models.encoders.factory import VisualLandmarkEncoderV2, build_encoder  # noqa: E402


DEFAULT_PLOT_PATH = PROJECT_ROOT / "inference_debug_pred_vs_gt.png"


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def auto_checkpoint() -> Optional[Path]:
    candidates = [
        PROJECT_ROOT / "checkpoints_modular" / "best_model.pth",
        PROJECT_ROOT / "ephrat_baseline_checkpoints" / "best_model.pth",
        PROJECT_ROOT / "ephrat_baseline_checkpoints" / "overfit_1utt.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def build_models(
    device: torch.device,
    encoder_type: str,
    num_landmark_points: int,
):
    visual_encoder = build_encoder(encoder_type).to(device)
    encoder = VisualLandmarkEncoderV2(
        visual_encoder,
        num_landmark_points=num_landmark_points,
        z_dim=512,
    ).to(device)

    base_decoder = TFiLMSIRENDecoder(
        hidden_dim=256,
        out_dim=80,
        num_layers=4,
        use_conv=True,
        output_activation=None,
    ).to(device)
    decoder = MelTemporalUpsampleDecoder(
        base_decoder,
        sample_rate=16000,
        fps=25,
        hop_length=256,
    ).to(device)
    return encoder, decoder


def compatible_state_dict(model: torch.nn.Module, state_dict: dict) -> tuple[dict, int, int]:
    model_state = model.state_dict()
    kept = {}
    skipped = 0
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            kept[key] = value
        else:
            skipped += 1
    return kept, len(model_state) - len(kept), skipped


def save_mel_plot(pred_mel: torch.Tensor, target_mel: torch.Tensor, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = pred_mel.detach().cpu()[0].float().transpose(0, 1).numpy()
    target = target_mel.detach().cpu()[0].float().transpose(0, 1).numpy()
    diff = pred - target
    vmin = min(float(pred.min()), float(target.min()))
    vmax = max(float(pred.max()), float(target.max()))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    gt_img = axes[0].imshow(target, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    axes[1].imshow(pred, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    diff_img = axes[2].imshow(diff, origin="lower", aspect="auto", interpolation="nearest", cmap="coolwarm")

    axes[0].set_title("Ground truth mel")
    axes[1].set_title("Predicted mel")
    axes[2].set_title("Prediction - ground truth")
    axes[2].set_xlabel("Mel frame")
    for ax in axes:
        ax.set_ylabel("Mel bin")

    fig.colorbar(gt_img, ax=axes[:2], fraction=0.02, pad=0.02)
    fig.colorbar(diff_img, ax=axes[2], fraction=0.02, pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_checkpoint(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    checkpoint_path: Optional[Path],
    device: torch.device,
    strict: bool,
) -> None:
    if checkpoint_path is None:
        print("[checkpoint] No checkpoint found/provided; running with random weights.")
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[checkpoint] Loaded: {safe_text(checkpoint_path)}")
    print(f"[checkpoint] Keys: {sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict with encoder_state_dict/decoder_state_dict.")

    if "encoder_state_dict" not in ckpt or "decoder_state_dict" not in ckpt:
        message = "Checkpoint has no encoder_state_dict/decoder_state_dict; skipping weight load."
        if strict:
            raise KeyError(message)
        print(f"[checkpoint] {message}")
        return

    if strict:
        enc_result = encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
        dec_result = decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
        print(f"[checkpoint] Encoder missing={len(enc_result.missing_keys)} unexpected={len(enc_result.unexpected_keys)}")
        print(f"[checkpoint] Decoder missing={len(dec_result.missing_keys)} unexpected={len(dec_result.unexpected_keys)}")
    else:
        enc_state, enc_missing, enc_skipped = compatible_state_dict(encoder, ckpt["encoder_state_dict"])
        dec_state, dec_missing, dec_skipped = compatible_state_dict(decoder, ckpt["decoder_state_dict"])
        encoder.load_state_dict(enc_state, strict=False)
        decoder.load_state_dict(dec_state, strict=False)
        print(f"[checkpoint] Encoder loaded={len(enc_state)} missing_or_shape_diff={enc_missing} skipped={enc_skipped}")
        print(f"[checkpoint] Decoder loaded={len(dec_state)} missing_or_shape_diff={dec_missing} skipped={dec_skipped}")
    if ckpt.get("loss") is not None:
        print(f"[checkpoint] Loss: {ckpt['loss']}")


def run_smoke(args: argparse.Namespace, device: torch.device) -> None:
    encoder, decoder = build_models(
        device=device,
        encoder_type=args.encoder_type,
        num_landmark_points=args.num_landmark_points,
    )
    load_checkpoint(encoder, decoder, resolve_path(args.checkpoint) or auto_checkpoint(), device, args.strict)
    encoder.eval()
    decoder.eval()

    video = torch.randn(1, 1, args.max_frames, 112, 112, device=device)
    landmarks = torch.randn(1, args.max_frames, args.num_landmark_points, 6, device=device)

    with torch.no_grad():
        z = encoder(video, landmarks)
        mel = decoder(z)

    print("[smoke] video:", tuple(video.shape))
    print("[smoke] landmarks:", tuple(landmarks.shape))
    print("[smoke] latent:", tuple(z.shape), "finite:", torch.isfinite(z).all().item())
    print("[smoke] mel:", tuple(mel.shape), "finite:", torch.isfinite(mel).all().item())
    print("[smoke] mel stats: min={:.4f} max={:.4f} mean={:.4f}".format(
        mel.min().item(),
        mel.max().item(),
        mel.mean().item(),
    ))


def run_dataset_infer(args: argparse.Namespace, device: torch.device) -> None:
    data_dir = resolve_path(args.data_dir)
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    dataset = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=False,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(resolve_path(args.dataset_output_dir) or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
    )

    sample_index = args.index
    if args.sample:
        sample_path = resolve_path(args.sample)
        if sample_path is None:
            raise ValueError("--sample cannot be empty")
        dataset.files = [sample_path.name]
        dataset.data_dir = str(sample_path.parent)
        sample_index = 0

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"--index {sample_index} out of range for dataset size {len(dataset)}")

    batch = collate_pad_v2([dataset[sample_index]])
    video, landmarks, target, lengths, paths = batch
    video = video.to(device)
    landmarks = landmarks.to(device)
    target = target.to(device)
    lengths = lengths.to(device)

    encoder, decoder = build_models(
        device=device,
        encoder_type=args.encoder_type,
        num_landmark_points=dataset.landmark_num_points,
    )
    load_checkpoint(encoder, decoder, resolve_path(args.checkpoint) or auto_checkpoint(), device, args.strict)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        z = encoder(video, landmarks)
        mel = decoder(z, target_len=target.shape[1])

    l1 = torch.nn.functional.l1_loss(mel.float(), target.float()).item()
    print("[data] file:", safe_text(paths[0]))
    print("[data] video:", tuple(video.shape))
    print("[data] landmarks:", tuple(landmarks.shape))
    print("[data] target mel:", tuple(target.shape), "length:", int(lengths[0].item()))
    print("[infer] latent:", tuple(z.shape), "finite:", torch.isfinite(z).all().item())
    print("[infer] pred mel:", tuple(mel.shape), "finite:", torch.isfinite(mel).all().item())
    print("[infer] pred stats: min={:.4f} max={:.4f} mean={:.4f}".format(
        mel.min().item(),
        mel.max().item(),
        mel.mean().item(),
    ))
    print(f"[infer] L1 vs target: {l1:.6f}")

    plot_path = resolve_path(args.plot) if args.plot else DEFAULT_PLOT_PATH
    save_mel_plot(mel, target, plot_path)
    print(f"[plot] Saved pred vs ground truth: {safe_text(plot_path)}")

    if args.output:
        out_path = resolve_path(args.output)
        if out_path is None:
            raise ValueError("--output cannot be empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "path": paths[0],
                "pred_mel": mel.detach().cpu(),
                "target_mel": target.detach().cpu(),
                "lengths": lengths.detach().cpu(),
            },
            out_path,
        )
        print(f"[output] Saved: {safe_text(out_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick model smoke test or one-sample inference test.")
    parser.add_argument("--data-dir", default="Processed_Data_Mel_HiFiGAN")
    parser.add_argument("--dataset-output-dir", default="Dataset_Output")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint. Auto-detects if omitted.")
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "snn", "nonsnn", "cnn_transformer"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample", default=None, help="Specific .pt sample path to test.")
    parser.add_argument("--output", default=None, help="Optional .pt output path for predicted mel.")
    parser.add_argument("--plot", default=str(DEFAULT_PLOT_PATH), help="PNG path for pred/ground-truth mel plot.")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--smoke-only", action="store_true", help="Run dummy tensor test instead of loading dataset.")
    parser.add_argument("--num-landmark-points", type=int, default=468, help="Only used by --smoke-only.")
    parser.add_argument("--disable-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[device] {device}")
    if args.smoke_only:
        run_smoke(args, device)
    else:
        run_dataset_infer(args, device)


if __name__ == "__main__":
    main()
