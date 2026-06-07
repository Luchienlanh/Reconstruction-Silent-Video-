from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URLS = {
    "public": "https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data",
    "extended": "https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/extended_train_data",
}

FILES = {
    "feature_extractor": "feature_extractor.pth",
    "ft_lrs2": "ft_lrs2.pth",
    "ft_lrs3": "ft_lrs3.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume-capable downloader for official VTP checkpoints.")
    parser.add_argument("--variant", choices=BASE_URLS.keys(), default="extended")
    parser.add_argument("--target", choices=FILES.keys(), action="append")
    parser.add_argument("--output-dir", default="pretrained_models/vtp")
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=3.0)
    return parser.parse_args()


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_range(url: str, path: Path, start: int, end: int) -> int:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if start > 0 and status != 206:
            raise RuntimeError(f"Server did not honor Range request: status={status}")
        data = response.read()
    expected = end - start + 1
    if len(data) > expected:
        raise RuntimeError(f"Server returned too many bytes: {len(data)} > {expected}")
    with path.open("ab") as f:
        f.write(data)
    return len(data)


def download_one(url: str, path: Path, chunk_bytes: int, retries: int, sleep_seconds: float) -> None:
    size = remote_size(url)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = path.stat().st_size if path.exists() else 0
    if existing > size:
        raise RuntimeError(
            f"Local file is larger than remote file: {path}. "
            "Move it aside or truncate it before resuming."
        )
    if existing == size:
        print(f"already complete: {path} ({size} bytes)")
        return

    print(f"downloading {url}")
    print(f"destination {path}")
    print(f"resume from {existing} / {size} bytes")

    failures = 0
    while existing < size:
        end = min(existing + chunk_bytes - 1, size - 1)
        try:
            written = download_range(url, path, existing, end)
            if written <= 0:
                raise RuntimeError("downloaded empty chunk")
            existing += written
            failures = 0
            percent = existing * 100.0 / size
            print(f"{path.name}: {existing}/{size} bytes ({percent:.2f}%)")
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            failures += 1
            if failures > retries:
                raise RuntimeError(f"Too many failures while downloading {url}") from exc
            print(f"retry {failures}/{retries} after error: {exc}")
            time.sleep(sleep_seconds)

    final_size = path.stat().st_size
    if final_size != size:
        raise RuntimeError(f"Download size mismatch for {path}: {final_size} != {size}")
    print(f"complete: {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    chunk_bytes = int(args.chunk_mb) * 1024 * 1024
    base_url = BASE_URLS[args.variant]
    targets = args.target or ["feature_extractor", "ft_lrs2"]

    for target in targets:
        filename = FILES[target]
        download_one(
            url=f"{base_url}/{filename}",
            path=output_dir / filename,
            chunk_bytes=chunk_bytes,
            retries=int(args.retries),
            sleep_seconds=float(args.sleep),
        )


if __name__ == "__main__":
    main()
