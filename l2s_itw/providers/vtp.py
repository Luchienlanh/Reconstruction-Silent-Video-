from __future__ import annotations

import contextlib
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class VTPProviderConfig:
    repo_dir: str
    ckpt_path: str
    cnn_ckpt_path: str
    builder: str = "vtp24x24"
    device: str = "cuda"
    feat_dim: int = 512
    num_blocks: int = 6
    hidden_units: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.1
    img_size: int = 96
    frame_size: int = 160
    normalize: bool = True
    beam_size: int = 30
    beam_len_alpha: float = 1.0
    max_decode_len: int = 35
    chunk_size: int = 100
    lm_alpha: float = 0.0
    use_flip_decode: bool = True
    fp16: bool = True


@dataclass
class VTPResult:
    text: str
    visual_features: torch.Tensor


@contextlib.contextmanager
def _temporary_vtp_import_context(config: VTPProviderConfig):
    repo_dir = str(Path(config.repo_dir).resolve())
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    old_cwd = os.getcwd()
    old_modules = {
        name: sys.modules.get(name)
        for name in ["config", "dataloader", "models", "utils", "search", "modules"]
    }

    sys.path.insert(0, repo_dir)
    sys.argv = [
        "vtp_provider",
        "--builder",
        config.builder,
        "--ckpt_path",
        str(Path(config.ckpt_path).resolve()),
        "--cnn_ckpt_path",
        str(Path(config.cnn_ckpt_path).resolve()),
        "--device",
        config.device,
        "--feat_dim",
        str(config.feat_dim),
        "--num_blocks",
        str(config.num_blocks),
        "--hidden_units",
        str(config.hidden_units),
        "--num_heads",
        str(config.num_heads),
        "--dropout_rate",
        str(config.dropout_rate),
        "--img_size",
        str(config.img_size),
        "--frame_size",
        str(config.frame_size),
        "--normalize",
        str(config.normalize),
        "--beam_size",
        str(config.beam_size),
        "--beam_len_alpha",
        str(config.beam_len_alpha),
        "--max_decode_len",
        str(config.max_decode_len),
        "--chunk_size",
        str(config.chunk_size),
        "--lm_alpha",
        str(config.lm_alpha),
        "--fp16",
        str(config.fp16),
    ]

    for name in old_modules:
        sys.modules.pop(name, None)

    try:
        os.chdir(repo_dir)
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path = old_path
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class VTPProvider:
    """Adapter around the official `prajwalkr/vtp` implementation.

    The official VTP code exposes text inference and feature extraction as
    separate scripts. This adapter keeps our pipeline clean by returning both
    outputs from one loaded model:

    - decoded lip-reading text
    - `model.face_encoder(...)` frame embeddings with shape `[T, 512]`
    """

    def __init__(self, config: VTPProviderConfig) -> None:
        self.config = config
        self.repo_dir = Path(config.repo_dir)
        self.ckpt_path = Path(config.ckpt_path)
        self.cnn_ckpt_path = Path(config.cnn_ckpt_path)
        self._validate_paths()

        with _temporary_vtp_import_context(config):
            self._config_mod = importlib.import_module("config")
            self._dataloader_mod = importlib.import_module("dataloader")
            self._models_mod = importlib.import_module("models")
            self._utils_mod = importlib.import_module("utils")
            self._search_mod = importlib.import_module("search")
            self.args = self._config_mod.load_args()
            self.video_loader = self._dataloader_mod.VideoDataset(self.args)
            self.augmentor = self._dataloader_mod.AugmentationPipeline(self.args)
            self.model = self._build_model()

    def _validate_paths(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"VTP repo_dir does not exist: {self.repo_dir}")
        if not (self.repo_dir / "models.py").exists():
            raise FileNotFoundError(f"VTP repo_dir does not look like prajwalkr/vtp: {self.repo_dir}")
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"VTP lip-reading checkpoint does not exist: {self.ckpt_path}")
        if not self.cnn_ckpt_path.exists():
            raise FileNotFoundError(f"VTP visual backbone checkpoint does not exist: {self.cnn_ckpt_path}")

    def _build_model(self) -> torch.nn.Module:
        builders = self._models_mod.builders
        model = builders[self.args.builder](
            self.video_loader.vocab_size + 1,
            self.args.feat_dim,
            N=self.args.num_blocks,
            d_model=self.args.hidden_units,
            h=self.args.num_heads,
            dropout=self.args.dropout_rate,
        )
        model = model.to(self.args.device).eval()
        model = self._utils_mod.load(
            model,
            self.args.ckpt_path,
            face_encoder_ckpt=self.args.cnn_ckpt_path,
            device=self.args.device,
        )[0]
        return model.eval()

    @contextlib.contextmanager
    def _runtime_import_context(self):
        old_path = sys.path[:]
        old_cwd = os.getcwd()
        module_names = ["config", "dataloader", "models", "utils", "search", "modules"]
        old_modules = {name: sys.modules.get(name) for name in module_names}

        sys.path.insert(0, str(self.repo_dir.resolve()))
        sys.modules["config"] = self._config_mod
        sys.modules["dataloader"] = self._dataloader_mod
        sys.modules["models"] = self._models_mod
        sys.modules["utils"] = self._utils_mod
        sys.modules["search"] = self._search_mod
        if "modules" in old_modules and old_modules["modules"] is not None:
            sys.modules["modules"] = old_modules["modules"]

        try:
            os.chdir(str(self.repo_dir.resolve()))
            yield
        finally:
            os.chdir(old_cwd)
            sys.path = old_path
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    @property
    def visual_dim(self) -> int:
        return int(self.args.feat_dim)

    def run(self, video_path: str | Path) -> VTPResult:
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        with self._runtime_import_context():
            frames = torch.FloatTensor(self.video_loader.read_video(str(video_path))).unsqueeze(0)
            frames = self.augmentor(frames).detach()

            chunk_frames = int(self.args.chunk_size) * 25
            texts: list[str] = []
            features: list[torch.Tensor] = []

            for start in range(0, frames.size(2), chunk_frames):
                src = frames[:, :, start : start + chunk_frames].to(self.args.device)
                src_mask = torch.ones((1, 1, src.size(2)), device=self.args.device)

                with torch.no_grad():
                    with self._autocast():
                        features.append(self.model.face_encoder(src, src_mask)[0].detach().float().cpu())
                        texts.append(self._decode_chunk(src, src_mask))

            text = " ".join(t for t in texts if t).strip()
            visual_features = torch.cat(features, dim=0) if features else torch.empty(0, self.visual_dim)
            return VTPResult(text=text, visual_features=visual_features)

    def _decode_chunk(self, src: torch.Tensor, src_mask: torch.Tensor) -> str:
        beam_outs, beam_scores = self._forward_pass(src, src_mask)
        beam_outs = beam_outs[0]
        beam_scores = list(beam_scores[0])

        if self.config.use_flip_decode:
            flipped = self.augmentor.horizontal_flip(src)
            flipped_outs, flipped_scores = self._forward_pass(flipped, src_mask)
            beam_outs = beam_outs + flipped_outs[0]
            beam_scores = beam_scores + list(flipped_scores[0])

        best_idx = max(range(len(beam_scores)), key=lambda idx: beam_scores[idx])
        return self.video_loader.to_tokens(beam_outs[best_idx].detach().cpu().numpy().tolist())

    def _forward_pass(self, src: torch.Tensor, src_mask: torch.Tensor):
        encoder_output, encoded_mask = self.model.encode(src, src_mask)
        return self._search_mod.beam_search(
            decoder=self.model,
            bos_index=self._config_mod.start_symbol,
            eos_index=self._config_mod.end_symbol,
            max_output_length=self.args.max_decode_len,
            pad_index=0,
            encoder_output=encoder_output,
            src_mask=encoded_mask,
            size=self.args.beam_size,
            alpha=self.args.beam_len_alpha,
            n_best=self.args.beam_size,
        )

    @contextlib.contextmanager
    def _autocast(self):
        enabled = bool(self.config.fp16) and str(self.args.device).startswith("cuda")
        if hasattr(torch, "amp"):
            with torch.amp.autocast("cuda", enabled=enabled):
                yield
        else:
            with torch.cuda.amp.autocast(enabled=enabled):
                yield


def config_from_dict(raw: dict) -> VTPProviderConfig:
    return VTPProviderConfig(**{key: value for key, value in raw.items() if key in VTPProviderConfig.__annotations__})
