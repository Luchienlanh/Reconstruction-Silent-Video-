from __future__ import annotations

import sys
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVHubertVisualFeatureExtractor(nn.Module):
    """Thin adapter around the official facebookresearch/av_hubert fairseq model.

    Expected video input is B x 1 x T x H x W. The official visual frontend uses
    96x96 mouth crops, so this adapter resizes if needed.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        avhubert_dir: str | Path,
        output_layer: int | None = None,
        freeze: bool = True,
        normalize_video: bool = True,
    ):
        super().__init__()
        self.checkpoint = str(checkpoint)
        self.avhubert_dir = str(avhubert_dir)
        self.output_layer = output_layer
        self.normalize_video = bool(normalize_video)
        self.model = self._load_model(self.checkpoint, self.avhubert_dir)
        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    @staticmethod
    def _load_model(checkpoint: str, avhubert_dir: str):
        avhubert_root = Path(avhubert_dir).resolve()
        if not avhubert_root.exists():
            raise FileNotFoundError(f"AV-HuBERT repo dir does not exist: {avhubert_root}")
        ckpt = Path(checkpoint).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"AV-HuBERT checkpoint does not exist: {ckpt}")

        fairseq_root = avhubert_root / "fairseq"
        fairseq_pkg = fairseq_root / "fairseq"
        avhubert_pkg = avhubert_root / "avhubert"
        if not avhubert_pkg.is_dir():
            raise FileNotFoundError(
                "AV-HuBERT package dir is missing. Expected: "
                f"{avhubert_pkg}"
            )
        if not fairseq_pkg.is_dir():
            raise FileNotFoundError(
                "AV-HuBERT fairseq submodule is missing. Run: "
                f"git -C {avhubert_root} submodule update --init --recursive"
            )

        # Official AV-HuBERT uses old absolute imports inside avhubert/*.py
        # such as `from hubert_pretraining import ...`, so both the repo root
        # and the avhubert package directory need to be importable.
        for candidate in (avhubert_root, avhubert_pkg, fairseq_root):
            text = str(candidate)
            while text in sys.path:
                sys.path.remove(text)
        sys.path.insert(0, str(avhubert_root))
        sys.path.insert(0, str(avhubert_pkg))
        sys.path.insert(0, str(fairseq_root))
        for name in list(sys.modules):
            if (
                name == "fairseq"
                or name.startswith("fairseq.")
                or name in {"hubert", "hubert_pretraining"}
                or name.startswith("avhubert.")
            ):
                del sys.modules[name]
        importlib.invalidate_caches()

        try:
            import fairseq  # type: ignore
            fairseq_file = getattr(fairseq, "__file__", None)
            if fairseq_file is None:
                raise ImportError(
                    "Imported fairseq as a namespace package, not the real package. "
                    f"Expected package under {fairseq_pkg}."
                )
            import hubert_pretraining  # noqa: F401
            import hubert  # noqa: F401
            import avhubert.hubert  # noqa: F401
            import avhubert.hubert_pretraining  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Could not import AV-HuBERT/fairseq. Use Python 3.8 for the official "
                "AV-HuBERT/fairseq stack, initialize the fairseq submodule, and install "
                "the AV-HuBERT dependencies. Make sure PYTHONPATH includes the repo root, "
                "the fairseq submodule, and av_hubert/avhubert. On Python 3.11+ this old "
                "fairseq often fails."
            ) from exc

        models, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(ckpt)])
        model = models[0]
        if hasattr(model, "remove_pretraining_modules"):
            model.remove_pretraining_modules()
        return model

    @staticmethod
    def _resize_video(video: torch.Tensor, size: int = 96) -> torch.Tensor:
        if video.shape[-2:] == (size, size):
            return video
        b, c, t, h, w = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
        return flat.reshape(b, t, c, size, size).permute(0, 2, 1, 3, 4).contiguous()

    def _prepare_video(self, video: torch.Tensor) -> torch.Tensor:
        video = self._resize_video(video.float(), 96)
        if self.normalize_video:
            # AV-HuBERT recipes use normalized mouth ROI. Our cache is already
            # usually in 0..1; this standardization avoids dataset-level scale drift.
            mean = video.mean(dim=(-1, -2), keepdim=True)
            std = video.std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(1e-4)
            video = (video - mean) / std
        return video

    @torch.no_grad()
    def forward(self, video: torch.Tensor, video_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        video = self._prepare_video(video)
        padding_mask = None if video_mask is None else ~video_mask.bool()
        source = {"video": video, "audio": None}
        if hasattr(self.model, "extract_finetune"):
            features, feature_padding = self.model.extract_finetune(
                source,
                padding_mask=padding_mask,
                mask=False,
                ret_conv=False,
                output_layer=self.output_layer,
            )
        else:
            features, feature_padding = self.model.extract_features(
                source,
                padding_mask=padding_mask,
                mask=False,
                ret_conv=False,
                output_layer=self.output_layer,
            )
        return features.float(), feature_padding
