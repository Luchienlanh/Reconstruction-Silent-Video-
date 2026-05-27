from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from srcV6.data import FullClipCTCDataset, cer, collate_ctc, greedy_decode, wer
from srcV6.models import build_model_from_config
from srcV6.utils import batch_to_device, get_device


def safe_print(text: str) -> None:
    import sys

    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(str(text).encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def load_model(checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    vocab = {str(k): int(v) for k, v in ckpt["vocab"].items()}
    model = build_model_from_config(ckpt.get("config") or {}, len(vocab)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[checkpoint] missing sample:", missing[:8])
    if unexpected:
        print("[checkpoint] unexpected sample:", unexpected[:8])
    model.eval()
    return model, vocab, ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer transcript from srcV6 CTC checkpoint.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-path", default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model, vocab, _ckpt = load_model(args.checkpoint, device)
    if args.sample_path:
        files = [Path(args.sample_path)]
    else:
        files = sorted(Path(args.data_dir).glob("*.pt"))
        if not files:
            raise RuntimeError(f"No .pt files found under {args.data_dir}")
        files = [files[int(args.sample_index)]]
    dataset = FullClipCTCDataset(
        args.data_dir,
        vocab=vocab,
        files=files,
        frame_stride=args.frame_stride,
        min_input_target_ratio=0.0,
        text_unit=str((_ckpt.get("config") or {}).get("text_unit", "char")),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_ctc)
    batch = batch_to_device(next(iter(loader)), device)
    logits = model(batch)
    pred_ids = logits.float().argmax(dim=-1)[0, : int(batch["video_lengths"][0].item())].detach().cpu().tolist()
    hyp = greedy_decode(pred_ids, vocab)
    ref = batch["transcript_texts"][0]
    safe_print(f"[path] {batch['paths'][0]}")
    safe_print(f"[ref] {ref}")
    safe_print(f"[hyp] {hyp}")
    safe_print(f"[cer] {cer(ref, hyp):.4f}")
    safe_print(f"[wer] {wer(ref, hyp):.4f}")


if __name__ == "__main__":
    run(parse_args())
