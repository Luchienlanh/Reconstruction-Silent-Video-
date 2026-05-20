"""
Overfit sanity test for the modular silent-video-to-mel pipeline.

This file intentionally lives outside src/. It trains on one .pt sample and
plots predicted mel vs ground truth so you can verify whether the model can
learn the basic mapping before using curriculum training.

Examples:
    python overfit_one_sample_test.py --epochs 300 --index 0
    python overfit_one_sample_test.py --data-dir Processed_Data_Mel_HiFiGAN_FullFrame --epochs 500
    python overfit_one_sample_test.py --force-full-frame --epochs 300
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
from models.loss import MelReconstructionLoss  # noqa: E402


DEFAULT_HIFIGAN_SOURCE = "speechbrain/tts-hifigan-libritts-16kHz"
DEFAULT_HIFIGAN_SAVEDIR = "pretrained_models/tts-hifigan-libritts-16kHz"


def safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def resolve_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


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


def load_hifigan(args: argparse.Namespace):
    from speechbrain.inference.vocoders import HIFIGAN
    from speechbrain.utils.fetching import LocalStrategy

    savedir = resolve_path(args.hifigan_savedir)
    savedir_text = str(savedir) if savedir is not None else args.hifigan_savedir
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
) -> None:
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
    print(f"[wav] {safe_text(pred_path)}")
    print(f"[wav] {safe_text(gt_path)}")


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


def load_one_sample(args: argparse.Namespace, device: torch.device):
    data_dir = resolve_path(args.data_dir)
    dataset_output_dir = resolve_path(args.dataset_output_dir)
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {safe_text(data_dir)}")

    dataset = VNLipDatasetV2(
        data_dir=str(data_dir),
        max_frames=args.max_frames,
        random_crop=False,
        return_path=True,
        target_type="mel_hifigan",
        use_landmarks=True,
        dataset_output_dir=str(dataset_output_dir or PROJECT_ROOT / "Dataset_Output"),
        enable_fallback=not args.disable_fallback,
        force_full_frame=args.force_full_frame,
    )

    sample_index = args.index
    if args.sample:
        sample_path = resolve_path(args.sample)
        if sample_path is None or not sample_path.is_file():
            raise FileNotFoundError(f"Sample not found: {safe_text(sample_path)}")
        dataset.files = [sample_path.name]
        dataset.data_dir = str(sample_path.parent)
        sample_index = 0

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"--index {sample_index} out of range for dataset size {len(dataset)}")

    batch = collate_pad_v2([dataset[sample_index]])
    video, landmarks, target, lengths, paths = batch
    return (
        video.to(device),
        landmarks.to(device),
        target.to(device),
        lengths.to(device),
        paths[0],
        dataset.landmark_num_points,
    )


def make_loss(args: argparse.Namespace, device: torch.device):
    if args.loss == "l1":
        return None
    return MelReconstructionLoss(
        lambda_mel=1.0,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
    ).to(device)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    criterion,
) -> torch.Tensor:
    if criterion is None:
        return torch.nn.functional.l1_loss(pred.float(), target.float())
    return criterion(pred.float(), target.float(), lengths)


def save_checkpoint(
    output_dir: Path,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
    sample_path: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "sample_path": sample_path,
        },
        output_dir / "best_model.pth",
    )


def run(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = resolve_path(args.output_dir) or PROJECT_ROOT / "overfit_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")
    video, landmarks, target, lengths, sample_path, num_landmark_points = load_one_sample(args, device)
    print(f"[sample] {safe_text(sample_path)}")
    print(f"[data] video={tuple(video.shape)} landmarks={tuple(landmarks.shape)} target={tuple(target.shape)}")
    print("[target] min={:.4f} max={:.4f} mean={:.4f}".format(
        target.min().item(),
        target.max().item(),
        target.mean().item(),
    ))

    encoder, decoder = build_models(device, args.encoder_type, num_landmark_points)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = make_loss(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    hifi_gan = None

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        reset_snn_if_needed(encoder, args.encoder_type)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
            z = encoder(video, landmarks)
            pred = decoder(z, target_len=target.shape[1])

        loss = compute_loss(pred, target, lengths, criterion)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.detach().cpu().item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                args.max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        reset_snn_if_needed(encoder, args.encoder_type)

        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            save_checkpoint(output_dir, encoder, decoder, optimizer, epoch, best_loss, args, sample_path)

        should_log = epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs
        should_plot = epoch == 1 or epoch % args.plot_every == 0 or epoch == args.epochs

        if should_log:
            print(
                "[epoch {:04d}] loss={:.6f} best={:.6f} pred_min={:.4f} pred_max={:.4f} pred_mean={:.4f}".format(
                    epoch,
                    loss_value,
                    best_loss,
                    pred.detach().min().item(),
                    pred.detach().max().item(),
                    pred.detach().mean().item(),
                )
            )

        if should_plot:
            encoder.eval()
            decoder.eval()
            reset_snn_if_needed(encoder, args.encoder_type)
            with torch.no_grad():
                eval_pred = decoder(encoder(video, landmarks), target_len=target.shape[1])
            reset_snn_if_needed(encoder, args.encoder_type)
            plot_path = output_dir / f"pred_vs_gt_epoch_{epoch:04d}.png"
            save_mel_plot(eval_pred, target, plot_path, title=f"Epoch {epoch} | loss={loss_value:.6f}")
            print(f"[plot] {safe_text(plot_path)}")
            if args.save_wav and args.wav_every > 0 and epoch % args.wav_every == 0:
                if hifi_gan is None:
                    print("[hifigan] Loading vocoder on CPU...")
                    hifi_gan = load_hifigan(args)
                save_wavs_from_mels(
                    hifi_gan=hifi_gan,
                    pred_mel=eval_pred,
                    target_mel=target,
                    lengths=lengths,
                    output_dir=output_dir,
                    tag=f"epoch_{epoch:04d}",
                    sample_rate=args.sample_rate,
                    hop_length=args.hop_length,
                )

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"loss": history, "best_loss": best_loss, "config": vars(args)}, f, indent=2)

    if args.save_wav:
        encoder.eval()
        decoder.eval()
        reset_snn_if_needed(encoder, args.encoder_type)
        with torch.no_grad():
            final_pred = decoder(encoder(video, landmarks), target_len=target.shape[1])
        reset_snn_if_needed(encoder, args.encoder_type)
        if hifi_gan is None:
            print("[hifigan] Loading vocoder on CPU...")
            hifi_gan = load_hifigan(args)
        save_wavs_from_mels(
            hifi_gan=hifi_gan,
            pred_mel=final_pred,
            target_mel=target,
            lengths=lengths,
            output_dir=output_dir,
            tag="final",
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
        )

    print(f"[done] best_loss={best_loss:.6f}")
    print(f"[output] {safe_text(output_dir)}")
    print(f"[checkpoint] {safe_text(output_dir / 'best_model.pth')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit one sample to debug whether the pipeline can learn.")
    parser.add_argument("--data-dir", default="Processed_Data_Mel_HiFiGAN")
    parser.add_argument("--dataset-output-dir", default="Dataset_Output")
    parser.add_argument("--output-dir", default="overfit_debug")
    parser.add_argument("--sample", default=None, help="Specific .pt file. If omitted, uses --index.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--encoder-type", default="non_snn", choices=["non_snn", "nonsnn", "cnn_transformer", "snn"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss", default="mel", choices=["l1", "mel"])
    parser.add_argument("--lambda-delta", type=float, default=0.0)
    parser.add_argument("--lambda-delta2", type=float, default=0.0)
    parser.add_argument("--lambda-energy", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--plot-every", type=int, default=50)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-full-frame", action="store_true")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--no-save-wav", dest="save_wav", action="store_false", help="Disable final HiFi-GAN wav export.")
    parser.add_argument("--wav-every", type=int, default=0, help="Also save wav every N epochs. 0 means final only.")
    parser.add_argument("--hifigan-source", default=DEFAULT_HIFIGAN_SOURCE)
    parser.add_argument("--hifigan-savedir", default=DEFAULT_HIFIGAN_SAVEDIR)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.set_defaults(save_wav=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
