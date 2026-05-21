"""
Unified inference, plotting, and audio synthesis script for modular models.

This script can run in four modes:
1. Normal Inference: Predict mel-spectrogram, plot, and optionally synthesize audio via HiFi-GAN.
2. Smoke Test Mode: Fast verification with random tensors (no dataset or GPU required).
3. Standalone HiFi-GAN Mode: Convert pre-saved predicted `.pt` mel-spectrogram files into `.wav` directly.

Examples:
    # Run a quick offline smoke test:
    python src/inference/infer.py --smoke-only

    # Run inference on a dataset sample with plots and audio synthesis:
    python src/inference/infer.py --checkpoint checkpoints_modular/best_model.pth --index 0 --save-wav

    # Decode a pre-saved .pt prediction to wav:
    python src/inference/infer.py --decode-pt-only inference_outputs/prediction_index_00000.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# Resolve directories relative to this file (located at src/inference/infer.py)
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import VNLipDatasetV2, collate_pad_v2  # noqa: E402
from models.decoders.dual import DualDecoder, DualWrapDecoder  # noqa: E402
from models.decoders.finer import TFiLMFINERDecoder  # noqa: E402
from models.decoders.siren import TFiLMSIRENDecoder  # noqa: E402
from models.decoders.upsample import MelTemporalUpsampleDecoder  # noqa: E402
from models.decoders.wire import TFiLMWIREDecoder  # noqa: E402
from models.decoders.wrap import TFiLMWrapFISINDecoder, TFiLMWrapFIWIDecoder  # noqa: E402
from models.encoders.factory import VisualLandmarkEncoderV2, build_encoder  # noqa: E402

DEFAULT_HIFIGAN_SOURCE = "speechbrain/tts-hifigan-libritts-16kHz"
DEFAULT_HIFIGAN_SAVEDIR = PROJECT_ROOT / "pretrained_models" / "tts-hifigan-libritts-16kHz"


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


def build_base_decoder(decoder_type: str):
    decoder_type = decoder_type.lower()
    common = dict(hidden_dim=256, out_dim=80, num_layers=4, use_conv=True)
    if decoder_type == "siren":
        return TFiLMSIRENDecoder(**common, output_activation=None)
    if decoder_type == "wire":
        return TFiLMWIREDecoder(**common)
    if decoder_type == "finer":
        return TFiLMFINERDecoder(**common)
    if decoder_type == "dual":
        return DualDecoder(**common)
    if decoder_type in {"dual_wrap", "dualwrap"}:
        return DualWrapDecoder(**common)
    if decoder_type in {"wrap_siren", "wrap_fisin", "wrap"}:
        return TFiLMWrapFISINDecoder(**common)
    if decoder_type in {"wrap_wire", "wrap_fiwi"}:
        return TFiLMWrapFIWIDecoder(**common)
    raise ValueError(f"Unknown decoder_type: {decoder_type}")


def build_models(device: torch.device, encoder_type: str, decoder_type: str, num_landmark_points: int):
    visual_encoder = build_encoder(encoder_type).to(device)
    encoder = VisualLandmarkEncoderV2(
        visual_encoder,
        num_landmark_points=num_landmark_points,
        z_dim=512,
    ).to(device)

    base_decoder = build_base_decoder(decoder_type).to(device)
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
    try:
        from spikingjelly.activation_based import functional
        functional.reset_net(module)
    except ImportError:
        pass


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


def load_hifigan(args: argparse.Namespace):
    from speechbrain.inference.vocoders import HIFIGAN
    from speechbrain.utils.fetching import LocalStrategy

    savedir = resolve_path(args.hifigan_savedir)
    savedir_text = str(savedir) if savedir is not None else str(DEFAULT_HIFIGAN_SAVEDIR)
    return HIFIGAN.from_hparams(
        source=args.hifigan_source,
        savedir=savedir_text,
        local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        run_opts={"device": "cpu"},
    )


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    import torchaudio
    wav = waveform.detach().cpu().float()
    if wav.dim() == 3:
        wav = wav[0]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav, sample_rate)


def save_wavs_from_mels(
    hifi_gan,
    pred_mel: torch.Tensor,
    target_mel: torch.Tensor,
    lengths: torch.Tensor,
    output_dir: Path,
    tag: str,
    sample_rate: int,
    hop_length: int,
) -> tuple[Path, Path]:
    pred_specs = pred_mel.detach().cpu().float().transpose(1, 2)
    target_specs = target_mel.detach().cpu().float().transpose(1, 2)
    mel_lens = lengths.detach().cpu().long()

    with torch.no_grad():
        pred_wav = hifi_gan.decode_batch(pred_specs, mel_lens=mel_lens, hop_len=hop_length)
        gt_wav = hifi_gan.decode_batch(target_specs, mel_lens=mel_lens, hop_len=hop_length)

    pred_path = output_dir / f"pred_{tag}.wav"
    gt_path = output_dir / f"ground_truth_{tag}.wav"
    save_waveform(pred_path, pred_wav, sample_rate)
    save_waveform(gt_path, gt_wav, sample_rate)
    return pred_path, gt_path


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


def run_smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    print("[smoke] Running quick dummy tensor smoke test...")
    encoder, decoder = build_models(
        device=device,
        encoder_type=args.encoder_type or "non_snn",
        decoder_type=args.decoder_type or "siren",
        num_landmark_points=args.num_landmark_points,
    )
    encoder.eval()
    decoder.eval()

    dummy_video = torch.randn(1, 1, 30, 112, 112, device=device)
    dummy_landmarks = torch.randn(1, 30, args.num_landmark_points, 6, device=device)

    with torch.no_grad():
        z = encoder(dummy_video, dummy_landmarks)
        mel = decoder(z, target_len=150)

    print("[smoke] Dummy video shape:", tuple(dummy_video.shape))
    print("[smoke] Dummy landmarks shape:", tuple(dummy_landmarks.shape))
    print("[smoke] Output latent shape:", tuple(z.shape), "finite:", torch.isfinite(z).all().item())
    print("[smoke] Output mel shape:", tuple(mel.shape), "finite:", torch.isfinite(mel).all().item())
    print("[smoke] Output statistics: min={:.4f} max={:.4f} mean={:.4f}".format(
        mel.min().item(),
        mel.max().item(),
        mel.mean().item(),
    ))
    print("[smoke] Smoke test completed successfully!")


def run_standalone_decoder(args: argparse.Namespace) -> None:
    pt_path = resolve_path(args.decode_pt_only)
    if pt_path is None or not pt_path.is_file():
        raise FileNotFoundError(f"Saved prediction file not found: {safe_text(args.decode_pt_only)}")

    print(f"[standalone] Loading saved mel prediction from: {safe_text(pt_path)}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict) or "pred_mel" not in data or "target_mel" not in data:
        raise KeyError("Saved .pt file must contain keys 'pred_mel' and 'target_mel'.")

    pred_mel = data["pred_mel"]
    target_mel = data["target_mel"]
    lengths = data.get("lengths", torch.tensor([pred_mel.shape[1]]))

    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[standalone] Loading HiFi-GAN vocoder on CPU...")
    hifi_gan = load_hifigan(args)

    tag = f"standalone_{pt_path.stem}"
    pred_path, gt_path = save_wavs_from_mels(
        hifi_gan=hifi_gan,
        pred_mel=pred_mel,
        target_mel=target_mel,
        lengths=lengths,
        output_dir=output_dir,
        tag=tag,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
    )
    print(f"[standalone] Decoded successfully!")
    print(f"[standalone] predicted wav -> {safe_text(pred_path)}")
    print(f"[standalone] ground_truth wav -> {safe_text(gt_path)}")


def run(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Standalone Mel to WAV conversion mode
    if args.decode_pt_only:
        run_standalone_decoder(args)
        return

    # 2. Smoke Test Mode
    if args.smoke_only:
        run_smoke_test(args, device)
        return

    # 3. Normal Inference Pipeline
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless running with --smoke-only or --decode-pt-only")

    checkpoint_path = resolve_path(args.checkpoint)
    if checkpoint_path is None:
        raise ValueError("--checkpoint cannot be empty")
    ckpt = load_checkpoint(checkpoint_path, device)
    ckpt_config = ckpt.get("config", {}) or {}

    dataset, data_dir, max_frames = create_dataset(args, ckpt_config)
    encoder_type = args.encoder_type or ckpt_config.get("encoder_type", "non_snn")
    decoder_type = args.decoder_type or ckpt_config.get("decoder_type", "siren")
    encoder, decoder = build_models(device, encoder_type, decoder_type, dataset.landmark_num_points)

    # Checkpoint weight loading (compatible mode by default)
    if args.strict:
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    else:
        # Auto shape alignment to prevent loading crashes if changing topologies
        enc_state = encoder.state_dict()
        dec_state = decoder.state_dict()
        enc_loaded = {k: v for k, v in ckpt["encoder_state_dict"].items() if k in enc_state and enc_state[k].shape == v.shape}
        dec_loaded = {k: v for k, v in ckpt["decoder_state_dict"].items() if k in dec_state and dec_state[k].shape == v.shape}
        encoder.load_state_dict(enc_loaded, strict=False)
        decoder.load_state_dict(dec_loaded, strict=False)
        print(f"[checkpoint] Encoder loaded: {len(enc_loaded)}/{len(enc_state)} layers")
        print(f"[checkpoint] Decoder loaded: {len(dec_loaded)}/{len(dec_state)} layers")

    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")
    print(f"[checkpoint] {safe_text(checkpoint_path)}")
    print(f"[data] {safe_text(data_dir)}")
    print(f"[model] encoder={encoder_type} decoder={decoder_type} landmarks={dataset.landmark_num_points} max_frames={max_frames}")

    start = 0 if args.sample else args.index
    indices = [start] if args.sample else list(range(start, min(start + args.num_samples, len(dataset))))
    if not indices:
        raise IndexError("No samples selected for inference.")

    hifi_gan = None
    if args.save_wav:
        print("[hifigan] Loading vocoder on CPU...")
        hifi_gan = load_hifigan(args)

    rows = []
    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"--index {index} out of range for dataset size {len(dataset)}")

        result = infer_one(encoder, decoder, dataset, index, device, encoder_type)
        sample_name = Path(result["path"]).name
        plot_path = output_dir / f"pred_vs_gt_index_{index:05d}.png"
        title = f"index={index} | L1={result['l1']:.6f} | {sample_name}"
        save_mel_plot(result["pred"], result["target"], plot_path, title)

        if args.save_wav and hifi_gan is not None:
            tag = f"index_{index:05d}"
            pred_wav_path, gt_wav_path = save_wavs_from_mels(
                hifi_gan=hifi_gan,
                pred_mel=result["pred"],
                target_mel=result["target"],
                lengths=result["lengths"],
                output_dir=output_dir,
                tag=tag,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
            )
        else:
            pred_wav_path, gt_wav_path = None, None

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
            "pred_wav": str(pred_wav_path) if pred_wav_path is not None else None,
            "gt_wav": str(gt_wav_path) if gt_wav_path is not None else None,
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
                "decoder_type": decoder_type,
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
    parser = argparse.ArgumentParser(description="Predict mel, plot, and synthesize audio with HiFi-GAN.")
    parser.add_argument("--checkpoint", default=None, help="Path to best_model.pth or last_model.pth.")
    parser.add_argument("--data-dir", default=None, help="Dataset directory.")
    parser.add_argument("--dataset-output-dir", default=None)
    parser.add_argument("--output-dir", default="inference_outputs")
    parser.add_argument("--sample", default=None, help="Specific .pt sample path.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None, help="0 or negative means full sample.")
    parser.add_argument("--encoder-type", default=None, choices=["non_snn", "nonsnn", "cnn_transformer", "snn"])
    parser.add_argument(
        "--decoder-type",
        default=None,
        choices=["siren", "wire", "finer", "dual", "dual_wrap", "wrap_siren", "wrap_fisin", "wrap", "wrap_wire", "wrap_fiwi"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force-full-frame", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable-fallback", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-pt", action="store_true", help="Save predicted mel tensors as .pt files.")
    parser.add_argument("--save-wav", action="store_true", help="Pass mels through HiFi-GAN to synthesize reconstructed and ground-truth wav files.")
    parser.add_argument("--smoke-only", action="store_true", help="Run a quick dummy tensor smoke test.")
    parser.add_argument("--num-landmark-points", type=int, default=468, help="Only used by --smoke-only.")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--decode-pt-only", default=None, help="Standalone HiFi-GAN decoding: path to a saved prediction .pt file.")
    parser.add_argument("--hifigan-source", default=DEFAULT_HIFIGAN_SOURCE)
    parser.add_argument("--hifigan-savedir", default=str(DEFAULT_HIFIGAN_SAVEDIR))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
