from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from l2t_arch.data import ArchDataset, collate_arch
from l2t_arch.decode import decode_ctc_batch, decode_seq2seq_batch
from l2t_arch.metrics import wer_cer
from l2t_arch.models import build_model
from l2t_arch.text import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen-VTP architecture checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-decode-len", type=int, default=120)
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    tokenizer = CharTokenizer(ckpt.get("symbols", config["text"]["symbols"]))
    dataset = ArchDataset(args.manifest, tokenizer, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_arch)
    model = build_model(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    predictions = []
    totals = {"word_edits": 0, "words": 0, "char_edits": 0, "chars": 0, "exact": 0}
    spike_sum = 0.0
    spike_count = 0
    beta_sum = 0.0
    beta_count = 0
    short_beta_sum = 0.0
    short_beta_count = 0
    long_beta_sum = 0.0
    long_beta_count = 0
    fusion_gate_sum = 0.0
    fusion_gate_count = 0
    kind = str(config["model"]["type"])

    with torch.no_grad():
        for raw_batch in tqdm(loader, desc="evaluate"):
            batch = move_batch(raw_batch, device)
            if kind == "visual_ctc":
                output = model(**batch)
                hyp_texts = decode_ctc_batch(output["ctc_logits"], tokenizer)
            else:
                output = model(**batch)
                hyp_texts = decode_seq2seq_batch(model, raw_batch, tokenizer, max_len=int(args.max_decode_len))
            if "spike_rate" in output:
                spike_sum += float(output["spike_rate"].detach().item())
                spike_count += 1
            if "beta_mean" in output:
                beta_sum += float(output["beta_mean"].detach().item())
                beta_count += 1
            if "short_beta_mean" in output:
                short_beta_sum += float(output["short_beta_mean"].detach().item())
                short_beta_count += 1
            if "long_beta_mean" in output:
                long_beta_sum += float(output["long_beta_mean"].detach().item())
                long_beta_count += 1
            if "fusion_gate_mean" in output:
                fusion_gate_sum += float(output["fusion_gate_mean"].detach().item())
                fusion_gate_count += 1

            for sample_id, gt, pred, vtp in zip(raw_batch["ids"], raw_batch["gt_texts"], hyp_texts, raw_batch["vtp_texts"]):
                metrics = wer_cer(gt, pred)
                for key in totals:
                    totals[key] += int(metrics[key])
                predictions.append({"id": sample_id, "gt": gt, "pred": pred, "vtp_text": vtp, **metrics})

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "samples": len(predictions),
        "wer": totals["word_edits"] / max(totals["words"], 1),
        "cer": totals["char_edits"] / max(totals["chars"], 1),
        "exact_match": totals["exact"] / max(len(predictions), 1),
    }
    if spike_count:
        summary["spike_rate"] = spike_sum / spike_count
    if beta_count:
        summary["beta_mean"] = beta_sum / beta_count
    if short_beta_count:
        summary["short_beta_mean"] = short_beta_sum / short_beta_count
    if long_beta_count:
        summary["long_beta_mean"] = long_beta_sum / long_beta_count
    if fusion_gate_count:
        summary["fusion_gate_mean"] = fusion_gate_sum / fusion_gate_count

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
