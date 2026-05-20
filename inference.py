"""
Inference and plotting script for models trained by train_simple_mel.py.

This file lives outside src/. It loads a simple-mel checkpoint, predicts mel
for one or more .pt samples, and saves pred-vs-ground-truth plots.

Examples:
    python inference.py --checkpoint simple_mel_checkpoints/best_model.pth --index 0
    python inference.py --checkpoint simple_mel_checkpoints/best_model.pth --data-dir FullFrame_test --index 10
    python inference.py --checkpoint simple_mel_checkpoints/best_model.pth --num-samples 8 --output-dir inference_outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VNLipDatasetV2, collate_pad_v2  # noqa: E402
from models.decoders.siren import TFiLMSIRENDecoder  # noqa: E402
from models.decoders.upsample import MelTemporalUpsampleDecoder  # noqa: E402
from models.encoders.factory import VisualLandmarkEncoderV2, build_encoder  # noqa: E402


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def default_data_dir() -> str:
    full_frame = PROJECT_ROOT / "Processed_Data_Mel_HiFiGAN_FullFrame"
    if full_frame.is_dir():
        return "Processed_Data_Mel_HiFiGAN_FullFrame"
    return "Processed_Data_Mel_HiFiGAN"


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {safe_text(path)}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "encoder_state_dict" not in ckpt or "decoder_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain encoder_state_dict and decoder_state_dict.")
    return ckpt


def config_value(args: argparse.Namespace, ckpt_config: dict, name: str, default=None):
    value = getattr(args, name)
    if value is not None:
        return value
    return ckpt_config.get(name, default)


def build_models(device: torch.device, encoder_type: str, num_landmark_points: int):
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


def reset_snn_if_needed(module: torch.nn.Module, encoder_type: str) -> None:
    if encoder_type != "snn":
        return
    from spikingjelly.activation_based import functional

    functional.reset_net(module)


def save_mel_plot(pred_mel: torch.Tensor, target_mel: torch.Tensor, output_path: Path, title: str) -> None:
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

    fig.suptitle(title)
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


def create_dataset(args: argparse.Namespace, ckpt_config: dict):
    data_dir_arg = args.data_dir or ckpt_config.get("data_dir") or default_data_dir()
    data_dir = resolve_path(data_dir_arg)
    dataset_output_dir = resolve_path(args.dataset_output_dir or ckpt_config.get("dataset_output_dir") or "Dataset_Output")
    max_frames = config_value(args, ckpt_config, "max_frames", 125)
    if max_frames is not None and int(max_frames) <= 0:
        max_frames = None

    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")

    force_full_frame = config_value(args, ckpt_config, "force_full_frame", False)
    disable_fallback = config_value(args, ckpt_config, "disable_fallback", False)

    dataset = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=max_frames,
        random_crop=False,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(dataset_output_dir or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not bool(disable_fallback),
        force_full_frame=bool(force_full_frame),
    )

    if args.sample:
        sample_path = resolve_path(args.sample)
        if sample_path is None or not sample_path.is_file():
            raise FileNotFoundError(f"Sample not found: {safe_text(sample_path)}")
        dataset.files = [sample_path.name]
        dataset.data_dir = str(sample_path.parent)

    return dataset, data_dir, max_frames


def infer_one(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    dataset: VNLipDatasetV2,
    index: int,
    device: torch.device,
    encoder_type: str,
):
    batch = collate_pad_v2([dataset[index]])
    video, landmarks, target, lengths, paths = batch
    video = video.to(device)
    landmarks = landmarks.to(device)
    target = target.to(device)
    lengths = lengths.to(device)

    encoder.eval()
    decoder.eval()
    reset_snn_if_needed(encoder, encoder_type)
    with torch.no_grad():
        z = encoder(video, landmarks)
        pred = decoder(z, target_len=target.shape[1])
    reset_snn_if_needed(encoder, encoder_type)

    l1 = torch.nn.functional.l1_loss(pred.float(), target.float()).item()
    return {
        "path": paths[0],
        "video": video,
        "landmarks": landmarks,
        "target": target,
        "lengths": lengths,
        "pred": pred,
        "l1": l1,
    }


def run(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = resolve_path(args.checkpoint)
    if checkpoint_path is None:
        raise ValueError("--checkpoint is required")
    ckpt = load_checkpoint(checkpoint_path, device)
    ckpt_config = ckpt.get("config", {}) or {}

    dataset, data_dir, max_frames = create_dataset(args, ckpt_config)
    encoder_type = args.encoder_type or ckpt_config.get("encoder_type", "non_snn")
    encoder, decoder = build_models(device, encoder_type, dataset.landmark_num_points)
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)

    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")
    print(f"[checkpoint] {safe_text(checkpoint_path)}")
    print(f"[data] {safe_text(data_dir)}")
    print(f"[model] encoder={encoder_type} landmarks={dataset.landmark_num_points} max_frames={max_frames}")

    start = 0 if args.sample else args.index
    indices = [start] if args.sample else list(range(start, min(start + args.num_samples, len(dataset))))
    if not indices:
        raise IndexError("No samples selected for inference.")

    rows = []
    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"--index {index} out of range for dataset size {len(dataset)}")

        result = infer_one(encoder, decoder, dataset, index, device, encoder_type)
        sample_name = Path(result["path"]).stem
        plot_path = output_dir / f"pred_vs_gt_index_{index:05d}.png"
        title = f"index={index} | L1={result['l1']:.6f} | {sample_name}"
        save_mel_plot(result["pred"], result["target"], plot_path, title)

        if args.save_pt:
            pt_path = output_dir / f"prediction_index_{index:05d}.pt"
            torch.save(
                {
                    "path": result["path"],
                    "pred_mel": result["pred"].detach().cpu(),
                    "target_mel": result["target"].detach().cpu(),
                    "lengths": result["lengths"].detach().cpu(),
                    "l1": result["l1"],
                },
                pt_path,
            )
        else:
            pt_path = None

        row = {
            "index": index,
            "path": result["path"],
            "l1": result["l1"],
            "plot": str(plot_path),
            "prediction_pt": str(pt_path) if pt_path is not None else None,
            "pred_shape": tuple(result["pred"].shape),
            "target_shape": tuple(result["target"].shape),
        }
        rows.append(row)
        print(
            "[sample {:05d}] L1={:.6f} pred={} target={} plot={}".format(
                index,
                result["l1"],
                tuple(result["pred"].shape),
                tuple(result["target"].shape),
                safe_text(plot_path),
            )
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "data_dir": str(data_dir),
                "encoder_type": encoder_type,
                "max_frames": max_frames,
                "results": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    avg_l1 = sum(row["l1"] for row in rows) / len(rows)
    print(f"[done] samples={len(rows)} avg_l1={avg_l1:.6f}")
    print(f"[output] {safe_text(output_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict mel and plot pred-vs-ground-truth after simple mel training.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth or last_model.pth from train_simple_mel.py.")
    parser.add_argument("--data-dir", default=None, help="Dataset directory. Defaults to checkpoint config, then FullFrame dataset.")
    parser.add_argument("--dataset-output-dir", default=None)
    parser.add_argument("--output-dir", default="inference_outputs")
    parser.add_argument("--sample", default=None, help="Specific .pt sample path. Overrides --index/--num-samples.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None, help="0 or negative means full sample. Defaults to checkpoint config.")
    parser.add_argument("--encoder-type", default=None, choices=["non_snn", "nonsnn", "cnn_transformer", "snn"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force-full-frame", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable-fallback", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-pt", action="store_true", help="Also save pred_mel/target_mel tensors.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
