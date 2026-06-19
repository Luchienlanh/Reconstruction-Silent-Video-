from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from l2t_arch.decode import decode_ctc_batch
from l2t_arch.models import build_model
from l2t_arch.text import CharTokenizer
from l2t_vtp.providers import VTPProvider, VTPProviderConfig


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _checkpoint_path() -> Path:
    explicit = os.environ.get("L2T_CHECKPOINT")
    if explicit:
        return Path(explicit).expanduser()

    candidates = [
        Path("checkpoints_l2t_arch_full_dual_path/best_model.pth"),
        Path("outputtrainning-20260614T072932Z-3-001/outputtrainning/checkpoints_l2t_arch_full_dual_path/best_model.pth"),
        Path("outputtrainning-20260614T072932Z-3-001/outputtrainning/results_and_models/checkpoints_l2t_arch_full_dual_path/best_model.pth"),
        Path("checkpoints_l2t_arch_dual_plif_monotonic/best_model.pth"),
        Path("checkpoints_l2t_arch_dual_path_causal/best_model.pth"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _device() -> torch.device:
    requested = os.environ.get("L2T_DEVICE", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _to_float(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _visual_sequence(features: torch.Tensor) -> torch.Tensor:
    features = features.detach().float()
    if features.ndim == 3:
        if features.shape[0] == 1:
            features = features.squeeze(0)
        else:
            features = features.flatten(0, 1)
    if features.ndim != 2:
        raise ValueError(f"Expected VTP visual features with shape [frames, dim], got {tuple(features.shape)}")
    return features


class DemoRuntime:
    def __init__(self) -> None:
        self.device = _device()
        self.checkpoint_path = _checkpoint_path()
        self.vtp_repo_dir = _env_path("VTP_REPO_DIR", "external/vtp")
        self.vtp_ckpt_path = _env_path("VTP_CKPT_PATH", "pretrained_models/vtp/ft_lrs2.pth")
        self.vtp_cnn_ckpt_path = _env_path("VTP_CNN_CKPT_PATH", "pretrained_models/vtp/feature_extractor.pth")
        self.max_decode_len = int(os.environ.get("L2T_MAX_DECODE_LEN", "80"))
        self.vtp_max_decode_len = int(os.environ.get("VTP_MAX_DECODE_LEN", "80"))
        self.vtp_beam_size = int(os.environ.get("VTP_BEAM_SIZE", "30"))
        self.model: torch.nn.Module | None = None
        self.tokenizer: CharTokenizer | None = None
        self.config: dict[str, Any] | None = None
        self.provider: VTPProvider | None = None

    def load(self) -> None:
        if self.model is not None and self.provider is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"L2T checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.tokenizer = CharTokenizer(checkpoint.get("symbols", self.config["text"]["symbols"]))
        self.model = build_model(self.config, self.tokenizer.vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.provider = VTPProvider(
            VTPProviderConfig(
                repo_dir=str(self.vtp_repo_dir),
                ckpt_path=str(self.vtp_ckpt_path),
                cnn_ckpt_path=str(self.vtp_cnn_ckpt_path),
                device=str(self.device),
                beam_size=self.vtp_beam_size,
                max_decode_len=self.vtp_max_decode_len,
                fp16=self.device.type == "cuda",
            )
        )

    @property
    def model_type(self) -> str:
        if not self.config:
            return ""
        return str(self.config["model"]["type"])

    @torch.no_grad()
    def transcribe(self, video_path: Path) -> dict[str, Any]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.provider is not None

        started = time.perf_counter()
        vtp_started = time.perf_counter()
        result = self.provider.run(video_path)
        vtp_elapsed = time.perf_counter() - vtp_started

        features = _visual_sequence(result.visual_features)
        visual = features.unsqueeze(0).to(self.device)
        visual_lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=self.device)
        vtp_tokens = torch.tensor(
            [self.tokenizer.encode(result.text, add_bos=True, add_eos=True)],
            dtype=torch.long,
            device=self.device,
        )

        decode_started = time.perf_counter()
        kind = self.model_type
        stats: dict[str, float] = {}
        if kind == "visual_ctc":
            output = self.model(visuals=visual, visual_lengths=visual_lengths, vtp_tokens=vtp_tokens)
            transcript = decode_ctc_batch(output["ctc_logits"], self.tokenizer)[0]
        else:
            generated = torch.full((1, 1), self.tokenizer.bos_id, dtype=torch.long, device=self.device)
            output = {}
            for _ in range(self.max_decode_len):
                output = self.model(
                    visuals=visual,
                    visual_lengths=visual_lengths,
                    vtp_tokens=vtp_tokens,
                    gt_in=generated,
                )
                next_id = output["logits"][:, -1].argmax(dim=-1)
                generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
                if bool(next_id.eq(self.tokenizer.eos_id).all()):
                    break
            transcript = self.tokenizer.decode(generated.detach().cpu().tolist()[0])

        for key in ("spike_rate", "beta_mean", "short_beta_mean", "long_beta_mean", "fusion_gate_mean"):
            if key in output:
                value = _to_float(output[key])
                if value is not None:
                    stats[key] = value

        decode_elapsed = time.perf_counter() - decode_started
        elapsed = time.perf_counter() - started
        return {
            "transcript": transcript,
            "vtp_text": result.text,
            "model_type": kind,
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "visual_frames": int(features.shape[0]),
            "visual_dim": int(features.shape[1]),
            "timing": {
                "total_seconds": elapsed,
                "vtp_seconds": vtp_elapsed,
                "decode_seconds": decode_elapsed,
            },
            "stats": stats,
        }


runtime = DemoRuntime()
app = FastAPI(title="l2t_arch Lip-to-Text Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("L2T_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "checkpoint": str(runtime.checkpoint_path),
        "checkpoint_exists": runtime.checkpoint_path.exists(),
        "vtp_repo_dir": str(runtime.vtp_repo_dir),
        "vtp_repo_exists": runtime.vtp_repo_dir.exists(),
        "device": str(runtime.device),
    }


@app.post("/api/transcribe")
async def transcribe_video(video: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Upload a video file: mp4, mov, avi, mkv, or webm.")

    try:
        with tempfile.TemporaryDirectory(prefix="l2t_demo_") as tmp_dir:
            video_path = Path(tmp_dir) / f"input{suffix}"
            with video_path.open("wb") as f:
                while chunk := await video.read(1024 * 1024):
                    f.write(chunk)
            result = runtime.transcribe(video_path)
            result["filename"] = video.filename
            return JSONResponse(result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


def main() -> None:
    import uvicorn

    uvicorn.run(
        "l2t_arch.demo_api:app",
        host=os.environ.get("L2T_HOST", "0.0.0.0"),
        port=int(os.environ.get("L2T_PORT", "8000")),
        reload=os.environ.get("L2T_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
