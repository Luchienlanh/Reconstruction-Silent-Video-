from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from srcV9.data.landmark_dataset import load_text_cache, split_cache_files
from srcV9.data.text import cer, normalize_text_nodiac, wer
from srcV9.utils import seed_everything, write_json


def _delta(x: torch.Tensor) -> torch.Tensor:
    d = x[1:] - x[:-1]
    return torch.cat([torch.zeros_like(d[:1]), d], dim=0)


def normalize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
    landmarks = torch.nan_to_num(landmarks.float(), nan=0.0, posinf=0.0, neginf=0.0)
    xy = landmarks[..., :2]
    center = xy.mean(dim=1, keepdim=True)
    xy_c = xy - center
    scale = xy_c.pow(2).sum(dim=-1).sqrt().amax(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
    xy_n = xy_c / scale
    if landmarks.shape[-1] >= 6:
        d1 = landmarks[..., 2:4] / scale
        d2 = landmarks[..., 4:6] / scale
    else:
        d1 = _delta(xy_n)
        d2 = _delta(d1)
    return torch.cat([xy_n, d1, d2], dim=-1)


def sequence_features(landmarks: torch.Tensor) -> torch.Tensor:
    x = normalize_landmarks(landmarks)
    xy = x[..., :2]
    d1 = x[..., 2:4]
    d2 = x[..., 4:6]
    xs = xy[..., 0]
    ys = xy[..., 1]
    width = (xs.amax(dim=1) - xs.amin(dim=1)).unsqueeze(-1)
    height = (ys.amax(dim=1) - ys.amin(dim=1)).unsqueeze(-1)
    area = width * height
    aspect = height / width.clamp_min(1e-4)
    speed = d1.pow(2).sum(dim=-1).sqrt()
    accel = d2.pow(2).sum(dim=-1).sqrt()
    radial = xy.pow(2).sum(dim=-1).sqrt()
    geom = torch.cat(
        [
            width,
            height,
            area,
            aspect,
            speed.mean(dim=1, keepdim=True),
            speed.std(dim=1, unbiased=False, keepdim=True),
            speed.amax(dim=1, keepdim=True),
            accel.mean(dim=1, keepdim=True),
            radial.mean(dim=1, keepdim=True),
            radial.std(dim=1, unbiased=False, keepdim=True),
        ],
        dim=-1,
    )
    flat_motion = torch.cat([xy.flatten(1), d1.flatten(1), d2.flatten(1), geom], dim=-1)
    return flat_motion.float()


def resample_sequence(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[0] == target_len:
        return x
    return F.interpolate(x.T.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0).T


def make_embedding(landmarks: torch.Tensor, seq_len: int = 64) -> torch.Tensor:
    seq = sequence_features(landmarks)
    seq = resample_sequence(seq, seq_len)
    mean = seq.mean(dim=0)
    std = seq.std(dim=0, unbiased=False)
    delta = (seq[1:] - seq[:-1]).abs().mean(dim=0) if seq.shape[0] > 1 else torch.zeros_like(mean)
    emb = torch.cat([seq.flatten(), mean, std, delta], dim=0)
    return F.normalize(emb.float(), dim=0)


def load_items(files: list[Path], seq_len: int) -> list[dict]:
    items = []
    for path in tqdm(files, desc="embed"):
        item = load_text_cache(path)
        text = normalize_text_nodiac(str(item.get("transcript_text", "")))
        emb = make_embedding(item["landmarks"].float(), seq_len=seq_len)
        items.append(
            {
                "path": str(path),
                "source_video": item.get("source_video", ""),
                "text": text,
                "embedding": emb,
            }
        )
    return items


def run(args) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files = split_cache_files(
        args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_files=args.limit_files if args.limit_files > 0 else None,
    )
    if args.query_split == "train":
        query_files = train_files
    elif args.query_split == "all":
        query_files = sorted(train_files + val_files)
    else:
        query_files = val_files
    if not query_files:
        query_files = train_files

    db_items = load_items(train_files, seq_len=args.seq_len)
    query_items = load_items(query_files, seq_len=args.seq_len)
    db_matrix = torch.stack([x["embedding"] for x in db_items], dim=0)

    results = []
    total_cer = 0.0
    total_wer = 0.0
    top1_same = 0
    for query in tqdm(query_items, desc="retrieve"):
        sims = torch.mv(db_matrix, query["embedding"])
        if args.exclude_self:
            for i, db in enumerate(db_items):
                if db["path"] == query["path"]:
                    sims[i] = -1e9
        topk = torch.topk(sims, k=min(args.topk, len(db_items)))
        matches = []
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            db = db_items[int(idx)]
            matches.append({"score": float(score), "path": db["path"], "text": db["text"]})
        hyp = matches[0]["text"] if matches else ""
        c = cer(query["text"], hyp)
        w = wer(query["text"], hyp)
        total_cer += c
        total_wer += w
        top1_same += int(hyp == query["text"])
        row = {
            "query_path": query["path"],
            "query_text": query["text"],
            "hyp_text": hyp,
            "cer": c,
            "wer": w,
            "matches": matches,
        }
        results.append(row)

    n = max(1, len(results))
    summary = {
        "data_dir": str(args.data_dir),
        "train_files": len(train_files),
        "query_files": len(query_items),
        "seq_len": args.seq_len,
        "topk": args.topk,
        "mean_cer": total_cer / n,
        "mean_wer": total_wer / n,
        "top1_exact_text_rate": top1_same / n,
        "results": results,
        "config": vars(args),
    }
    write_json(output_dir / "retrieval_results.json", summary)
    print(f"[data] db={len(db_items)} query={len(query_items)} split={args.query_split}")
    print(f"[retrieval] mean_cer={summary['mean_cer']:.4f} mean_wer={summary['mean_wer']:.4f} exact={summary['top1_exact_text_rate']:.4f}")
    for row in results[: args.print_samples]:
        print(f"ref: {row['query_text'][:160]}")
        print(f"hyp: {row['hyp_text'][:160]}")
        print(f"cer={row['cer']:.3f} wer={row['wer']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Landmark-motion retrieval baseline.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--output-dir", default="retrieval_srcV9_landmark")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--query-split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--exclude-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-samples", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

