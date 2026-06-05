from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# HuBERT is a PyTorch model here. Keep TensorFlow/protobuf out of the import path.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm

from srcV2.utils.audio import load_waveform
from srcV4.utils import seed_everything


class SSLFeatureExtractor:
    def __init__(self, model_name: str, device: torch.device, sample_rate: int = 16000, layer: int = -1):
        from transformers import AutoFeatureExtractor, AutoModel

        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.sample_rate = int(sample_rate)
        self.layer = int(layer)

    @torch.no_grad()
    def __call__(self, audio_path: str | Path) -> np.ndarray:
        wav, sr = load_waveform(audio_path, self.sample_rate)
        if int(sr) != self.sample_rate:
            raise RuntimeError(f"Expected sample_rate={self.sample_rate}, got {sr}: {audio_path}")
        samples = wav.squeeze(0).cpu().numpy()
        inputs = self.processor(samples, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[self.layer] if out.hidden_states is not None else out.last_hidden_state
        return hidden.squeeze(0).float().cpu().numpy()


def cache_files(data_dir: str | Path, limit: int | None = None) -> list[Path]:
    files = sorted(Path(data_dir).glob("*.pt"))
    if limit is not None:
        files = files[: max(1, min(int(limit), len(files)))]
    if not files:
        raise RuntimeError(f"No .pt files found under {data_dir}")
    return files


def choose_sample_indices(n_items: int, max_items: int, rng: np.random.Generator) -> np.ndarray:
    if n_items <= max_items:
        return np.arange(n_items)
    return np.sort(rng.choice(n_items, size=max_items, replace=False))


def units_to_mel_length(units: np.ndarray, mel_len: int) -> torch.Tensor:
    if units.shape[0] == int(mel_len):
        return torch.from_numpy(units.astype(np.int64))
    x = torch.from_numpy(units.astype(np.float32)).view(1, 1, -1)
    return F.interpolate(x, size=int(mel_len), mode="nearest").view(-1).long()


def extract_audio(video_path: str | Path, wav_path: str | Path, sample_rate: int, ffmpeg_bin: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        str(wav_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()}")


class AudioResolver:
    def __init__(self, sample_rate: int, ffmpeg_bin: str):
        self.sample_rate = int(sample_rate)
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.tmp = tempfile.TemporaryDirectory(prefix="srcV4_units_audio_")
        self.tmp_dir = Path(self.tmp.name)
        self.counter = 0

    def close(self) -> None:
        self.tmp.cleanup()

    def resolve(self, item: dict, cache_path: Path) -> Path:
        source_audio = item.get("source_audio")
        if source_audio and Path(source_audio).is_file():
            return Path(source_audio)
        source_video = item.get("source_video")
        if not source_video or not Path(source_video).is_file():
            raise RuntimeError(f"Missing source_audio/source_video in {cache_path}")
        self.counter += 1
        wav_path = self.tmp_dir / f"audio_{self.counter:08d}.wav"
        extract_audio(source_video, wav_path, self.sample_rate, self.ffmpeg_bin)
        return wav_path


def copy_cache(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return torch.load(dst, map_location="cpu", weights_only=False)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    files = cache_files(args.data_dir, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = SSLFeatureExtractor(args.model_name, device=device, sample_rate=args.sample_rate, layer=args.layer)
    resolver = AudioResolver(args.sample_rate, args.ffmpeg_bin)

    print(f"[device] {device}")
    print(f"[data] files={len(files)}")
    print(f"[ssl] model={args.model_name} layer={args.layer}")
    print(f"[kmeans] num_units={args.num_units} sample_per_file={args.sample_per_file}")

    feature_chunks: list[np.ndarray] = []
    feature_dim = 0
    try:
        for path in tqdm(files, desc="extract-fit"):
            item = torch.load(path, map_location="cpu", weights_only=False)
            audio_path = resolver.resolve(item, path)
            feats = extractor(audio_path)
            feature_dim = int(feats.shape[-1])
            idx = choose_sample_indices(feats.shape[0], args.sample_per_file, rng)
            feature_chunks.append(feats[idx].astype(np.float32))
        train_feats = np.concatenate(feature_chunks, axis=0)
        if train_feats.shape[0] < args.num_units:
            raise RuntimeError(f"Not enough feature frames ({train_feats.shape[0]}) for num_units={args.num_units}")

        kmeans = MiniBatchKMeans(
            n_clusters=args.num_units,
            batch_size=args.kmeans_batch_size,
            n_init=args.kmeans_n_init,
            random_state=args.seed,
            verbose=0,
        )
        kmeans.fit(train_feats)

        items = []
        for src in tqdm(files, desc="write-units"):
            dst = output_dir / src.name
            item = copy_cache(src, dst)
            audio_path = resolver.resolve(item, src)
            feats = extractor(audio_path)
            units = kmeans.predict(feats.astype(np.float32))
            unit_t = units_to_mel_length(units, int(item["mel_len"]))
            item["speech_units"] = unit_t.contiguous()
            item["num_speech_units"] = int(args.num_units)
            item["speech_unit_source"] = {
                "model_name": args.model_name,
                "layer": int(args.layer),
                "num_units": int(args.num_units),
                "ssl_frames": int(feats.shape[0]),
                "feature_dim": int(feature_dim or feats.shape[-1]),
                "aligned_to": "mel_len",
            }
            torch.save(item, dst)
            items.append(
                {
                    "file": str(dst),
                    "mel_len": int(item["mel_len"]),
                    "unit_len": int(unit_t.numel()),
                    "unique_units": int(unit_t.unique().numel()),
                }
            )
    finally:
        resolver.close()

    manifest = {
        "source_data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "total": len(items),
        "model_name": args.model_name,
        "layer": int(args.layer),
        "num_units": int(args.num_units),
        "sample_rate": int(args.sample_rate),
        "items": items,
    }
    (output_dir / "speech_units_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote={len(items)} out={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add HuBERT/k-means speech units to srcV4 r2inr cache files.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2_10k")
    parser.add_argument("--output-dir", default="Processed_Data_R2INR_LRS2_10k_units50")
    parser.add_argument("--model-name", default="facebook/hubert-base-ls960")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--num-units", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--sample-per-file", type=int, default=400)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--kmeans-n-init", type=int, default=3)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
