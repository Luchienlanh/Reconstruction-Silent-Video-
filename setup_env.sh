#!/usr/bin/env bash
set -euo pipefail

# Full environment setup for the silent-video reconstruction project.
#
# Usage:
#   bash setup_env.sh
#
# Optional:
#   VENV_DIR=.venv TORCH_FLAVOR=cu124 bash setup_env.sh
#   VENV_DIR=.venv TORCH_FLAVOR=cpu bash setup_env.sh
#
# TORCH_FLAVOR options:
#   cu124  - CUDA 12.4 PyTorch wheels, default
#   cpu    - CPU-only PyTorch wheels

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-"${PROJECT_ROOT}/.venv"}"
TORCH_FLAVOR="${TORCH_FLAVOR:-cu124}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${PROJECT_ROOT}/src/requirements.txt"

echo "[setup] Project root: ${PROJECT_ROOT}"
echo "[setup] Virtualenv: ${VENV_DIR}"
echo "[setup] Torch flavor: ${TORCH_FLAVOR}"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[error] Requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "[apt] Installing system packages..."
  if [[ "${EUID}" -eq 0 ]]; then
    APT_PREFIX=()
  else
    APT_PREFIX=(sudo)
  fi

  "${APT_PREFIX[@]}" apt-get update
  "${APT_PREFIX[@]}" apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1
else
  echo "[apt] apt-get not found; skipping system package install."
  echo "[apt] Make sure python3, python3-venv, pip, ffmpeg, libsndfile, and OpenCV runtime libs are installed."
fi

echo "[venv] Creating virtual environment if needed..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[pip] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

case "${TORCH_FLAVOR}" in
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  cu124)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    ;;
  *)
    echo "[error] Unsupported TORCH_FLAVOR='${TORCH_FLAVOR}'. Use 'cu124' or 'cpu'." >&2
    exit 1
    ;;
esac

echo "[pip] Installing PyTorch stack from ${TORCH_INDEX_URL}..."
python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url "${TORCH_INDEX_URL}"

echo "[pip] Installing project requirements..."
python -m pip install -r "${REQ_FILE}"

echo "[check] Import smoke test..."
python - <<'PY'
import torch
import torchaudio
import cv2
import librosa
import speechbrain
import spikingjelly
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchaudio", torchaudio.__version__)
print("opencv", cv2.__version__)
print("librosa", librosa.__version__)
print("speechbrain", getattr(speechbrain, "__version__", "ok"))
print("spikingjelly", getattr(spikingjelly, "__version__", "ok"))
PY

cat <<EOF

[done] Environment setup complete.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Quick smoke tests:
  python overfit_one_sample_test.py --epochs 1 --max-frames 4 --index 0 --no-save-wav
  python train_simple_mel.py --epochs 1 --limit 2 --max-frames 4 --batch-size 1 --val-ratio 0.5

EOF
