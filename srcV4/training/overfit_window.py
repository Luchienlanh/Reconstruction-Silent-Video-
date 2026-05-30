from __future__ import annotations

from srcV4.training.train_windows import parse_args, run


def main() -> None:
    args = parse_args()
    if args.output_dir == "checkpoints_srcV4_lrs2":
        args.output_dir = "overfit_srcV4_window"
    args.limit_files = 1 if args.limit_files <= 0 else args.limit_files
    args.max_windows_per_file = 1 if args.max_windows_per_file <= 0 else args.max_windows_per_file
    args.val_ratio = 0.0
    args.batch_size = 1
    args.drop_last = False
    args.epochs = 50 if args.epochs == 80 else args.epochs
    run(args)


if __name__ == "__main__":
    main()
