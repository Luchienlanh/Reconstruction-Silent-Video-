from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from srcV8.data.video_dataset import load_text_cache
from srcV8.models import AVHubertVisualFeatureExtractor
from srcV11.data import cer, load_feature_cache, normalize_english, split_feature_files, wer
from srcV11.utils import get_device


def configure_task(task: Any, cfg: Any, modalities: list[str]) -> Any:
    if hasattr(cfg, "task"):
        try:
            cfg.task.modalities = modalities
        except Exception:
            pass
    for name in ("tokenizer", "bpe"):
        if hasattr(task, f"build_{name}") and hasattr(cfg, name):
            try:
                getattr(task, f"build_{name}")(getattr(cfg, name))
            except Exception:
                pass
    return task


def tokens_to_words(tokens: torch.Tensor, dictionary: Any, generator: Any | None = None) -> str:
    ignore = set()
    for owner in (dictionary, generator):
        if owner is None:
            continue
        for name in ("pad", "eos"):
            value = getattr(owner, name, None)
            try:
                ignore.add(int(value() if callable(value) else value))
            except Exception:
                pass
    symbols = getattr(generator, "symbols_to_strip_from_output", None)
    if symbols is not None:
        ignore.update(int(x) for x in symbols)
    try:
        text = dictionary.string(tokens.int().cpu(), extra_symbols_to_ignore=ignore)
    except TypeError:
        text = dictionary.string(tokens.int().cpu())
    # AV-HuBERT VSR recipes use a character-like dictionary with "|" as word
    # separator. This mirrors the official infer_s2s.py cleanup.
    return " ".join("".join(text.split()).replace("|", " ").split())


def load_video_item(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    if args.feature_file:
        feature_path = Path(args.feature_file)
        feature = load_feature_cache(feature_path)
        source_cache = str(feature.get("source_cache", ""))
        if not source_cache:
            raise ValueError(f"{feature_path} does not contain source_cache.")
        cache_path = Path(source_cache)
        ref_text = str(feature.get("transcript_text", ""))
        return load_text_cache(cache_path), cache_path, ref_text

    if args.cache_file:
        cache_path = Path(args.cache_file)
        item = load_text_cache(cache_path)
        return item, cache_path, str(item.get("transcript_text", ""))

    files = sorted(Path(args.data_dir).glob("*.pt"))
    if not files:
        raise RuntimeError(f"No .pt cache files found under {args.data_dir}")
    sample = files[max(0, min(int(args.sample_index), len(files) - 1))]
    item = load_text_cache(sample)
    return item, sample, str(item.get("transcript_text", ""))


def make_source(extractor: AVHubertVisualFeatureExtractor, item: dict[str, Any], device: torch.device) -> tuple[dict, torch.Tensor]:
    video_len = int(item.get("video_len", item["video"].shape[1]))
    video = item["video"].float()[:, :video_len].unsqueeze(0).to(device)
    valid_mask = torch.ones(1, video_len, dtype=torch.bool, device=device)
    video = extractor._prepare_video(video)
    padding_mask = ~valid_mask
    return {"video": video, "audio": None}, padding_mask


@torch.no_grad()
def decode_s2s(model: torch.nn.Module, task: Any, source: dict, padding_mask: torch.Tensor, args: argparse.Namespace) -> tuple[str, dict]:
    from fairseq.dataclass.configs import GenerationConfig

    generation = GenerationConfig(
        beam=int(args.beam),
        lenpen=float(args.lenpen),
        max_len_a=float(args.max_len_a),
        max_len_b=int(args.max_len_b),
    )
    generator = task.build_generator([model], generation)
    sample = {
        "id": torch.LongTensor([0]).to(padding_mask.device),
        "net_input": {"source": source, "padding_mask": padding_mask},
    }
    hypos = task.inference_step(generator, [model], sample)
    tokens = hypos[0][0]["tokens"].int().cpu()
    text = tokens_to_words(tokens, task.target_dictionary, generator)
    meta = {"decode_mode": "seq2seq", "tokens": tokens.tolist(), "score": float(hypos[0][0].get("score", 0.0))}
    return text, meta


@torch.no_grad()
def decode_ctc(model: torch.nn.Module, task: Any, source: dict, padding_mask: torch.Tensor) -> tuple[str, dict]:
    output = model(source=source, padding_mask=padding_mask)
    logits = model.get_logits(output) if hasattr(model, "get_logits") else output["encoder_out"]
    if logits.dim() != 3:
        raise RuntimeError(f"Expected CTC logits to be 3D, got {tuple(logits.shape)}")
    if logits.shape[0] == 1:
        ids = logits[0].argmax(dim=-1).detach().cpu().tolist()
    else:
        ids = logits[:, 0].argmax(dim=-1).detach().cpu().tolist()
    collapsed = []
    prev = None
    for idx in ids:
        idx = int(idx)
        if idx != 0 and idx != prev:
            collapsed.append(idx)
        prev = idx
    tokens = torch.tensor(collapsed, dtype=torch.long)
    text = tokens_to_words(tokens, task.target_dictionary)
    return text, {"decode_mode": "ctc", "tokens": collapsed}


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    item, sample_path, ref_raw = load_video_item(args)
    ref_text = normalize_english(ref_raw)

    extractor = AVHubertVisualFeatureExtractor(
        checkpoint=args.checkpoint,
        avhubert_dir=args.avhubert_dir,
        freeze=True,
        normalize_mode=args.normalize_mode,
        crop_size=args.crop_size,
    ).to(device)
    model = extractor.model.to(device).eval()
    task = configure_task(extractor.fairseq_task, extractor.fairseq_cfg, ["video"])
    source, padding_mask = make_source(extractor, item, device)

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        hyp_text, decode_meta = decode_s2s(model, task, source, padding_mask, args)
    else:
        hyp_text, decode_meta = decode_ctc(model, task, source, padding_mask)

    meta = {
        "checkpoint": str(args.checkpoint),
        "sample": str(sample_path),
        "source_video": item.get("source_video", ""),
        "video_len": int(item.get("video_len", item["video"].shape[1])),
        "preprocess": {"crop_size": int(args.crop_size), "normalize_mode": str(args.normalize_mode)},
        "reference_text": ref_text,
        "predicted_text": normalize_english(hyp_text),
        "raw_predicted_text": hyp_text,
        "cer": cer(ref_text, hyp_text) if ref_text else None,
        "wer": wer(ref_text, hyp_text) if ref_text else None,
        **decode_meta,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sample_path.stem
    (output_dir / f"{stem}_official_vsr.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode one R2INR/LRS2 sample with the official AV-HuBERT VSR head.")
    parser.add_argument("--avhubert-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-file", default="", help="Cached avhubert_feature_v1 file; source_cache will be decoded.")
    parser.add_argument("--cache-file", default="", help="Original R2INR text cache .pt file.")
    parser.add_argument("--data-dir", default="Processed_Data_R2INR_LRS2_10k")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", default="infer_avhubert_official")
    parser.add_argument("--beam", type=int, default=20)
    parser.add_argument("--lenpen", type=float, default=1.0)
    parser.add_argument("--max-len-a", type=float, default=1.0)
    parser.add_argument("--max-len-b", type=int, default=20)
    parser.add_argument("--normalize-mode", choices=["avhubert", "per_frame", "none"], default="avhubert")
    parser.add_argument("--crop-size", type=int, default=88)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
