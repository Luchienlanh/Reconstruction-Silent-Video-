from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srcV9.data.landmark_dataset import load_text_cache, split_cache_files
from srcV9.data.text import build_vocab, tokenize_text_unit


def collect_tokens(files: list[Path], text_unit: str) -> tuple[list[list[str]], Counter[str]]:
    all_tokens = []
    counter: Counter[str] = Counter()
    for path in files:
        item = load_text_cache(path)
        tokens = tokenize_text_unit(str(item.get("transcript_text", "")), text_unit=text_unit)
        all_tokens.append(tokens)
        counter.update(tokens)
    return all_tokens, counter


def run(args: argparse.Namespace) -> None:
    train_files, val_files = split_cache_files(
        args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_files=args.limit_files if args.limit_files > 0 else None,
    )
    train_tokens, train_counter = collect_tokens(train_files, args.text_unit)
    val_tokens, val_counter = collect_tokens(val_files, args.text_unit)
    vocab = build_vocab(
        [" ".join(tokens) if args.text_unit == "syllable_nodiac" else "".join(tokens) for tokens in train_tokens],
        min_freq=args.min_token_freq,
        text_unit=args.text_unit,
    )
    known = set(vocab)
    special = {"<blank>", "<pad>", "<unk>"}
    known = known - special
    val_total = sum(len(tokens) for tokens in val_tokens)
    val_oov = sum(1 for tokens in val_tokens for tok in tokens if tok not in known)
    train_total = sum(len(tokens) for tokens in train_tokens)
    unique_train = len(train_counter)
    unique_val = len(val_counter)
    unique_oov = sorted(tok for tok in val_counter if tok not in known)
    print(f"[data] train_files={len(train_files)} val_files={len(val_files)}")
    print(f"[unit] {args.text_unit} min_token_freq={args.min_token_freq}")
    print(f"[train] tokens={train_total} unique={unique_train} vocab={len(vocab)}")
    print(f"[val] tokens={val_total} unique={unique_val}")
    print(f"[oov] val_oov_tokens={val_oov}/{max(1, val_total)} rate={val_oov / max(1, val_total):.4f}")
    print(f"[oov] unique_oov={len(unique_oov)}")
    if unique_oov:
        print("[oov examples]", " ".join(unique_oov[: args.print_oov]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check train/val token coverage for srcV9 CTC labels.")
    parser.add_argument("--data-dir", default="Processed_Data_TextV1")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--text-unit", choices=["char", "char_nodiac", "syllable_nodiac"], default="syllable_nodiac")
    parser.add_argument("--min-token-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-oov", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

