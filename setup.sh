#!/usr/bin/env bash
set -euo pipefail

# Server bootstrap for the l2t_arch / frozen-VTP lip-to-text pipeline.
#
# Typical usage after cloning the repo:
#   bash setup.sh
#
# Optional:
#   bash setup.sh --download-vtp
#   DATA_ROOT=/workspace TORCH_FLAVOR=cu124 bash setup.sh --download-vtp
#   TORCH_FLAVOR=cpu bash setup.sh --skip-apt
#
# This script intentionally does not download LRS2/LRS3 datasets. Dataset
# credentials and large downloads should be handled explicitly after setup.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data}"
DATASET_ROOT="${DATASET_ROOT:-${DATA_ROOT}/datasets}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/cache/l2t_arch}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/logs/l2t_arch}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_FLAVOR="${TORCH_FLAVOR:-cu121}"
DOWNLOAD_VTP=0
SKIP_APT=0

usage() {
  cat <<'EOF'
Usage:
  bash setup.sh [options]

Options:
  --download-vtp      Download feature_extractor, ft_lrs2, and ft_lrs3 checkpoints.
  --skip-apt          Skip apt-get package installation.
  --help              Show this help message.

Environment variables:
  DATA_ROOT           Default: /data
  DATASET_ROOT        Default: $DATA_ROOT/datasets
  CACHE_ROOT          Default: $DATA_ROOT/cache/l2t_arch
  LOG_ROOT            Default: $DATA_ROOT/logs/l2t_arch
  VENV_DIR            Default: $PROJECT_ROOT/.venv
  PYTHON_BIN          Default: python3
  TORCH_FLAVOR        cu121, cu124, or cpu. Default: cu121
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-vtp)
      DOWNLOAD_VTP=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

if command -v sudo >/dev/null 2>&1 && [[ "${EUID}" -ne 0 ]]; then
  SUDO=(sudo)
else
  SUDO=()
fi

log "Project root: ${PROJECT_ROOT}"
log "Data root: ${DATA_ROOT}"
log "Cache root: ${CACHE_ROOT}"
log "Virtualenv: ${VENV_DIR}"
log "Torch flavor: ${TORCH_FLAVOR}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  log "Creating DATA_ROOT with elevated permissions if needed"
  "${SUDO[@]}" mkdir -p "${DATA_ROOT}"
fi
if [[ ! -w "${DATA_ROOT}" ]]; then
  if [[ "${EUID}" -eq 0 ]]; then
    echo "[error] DATA_ROOT exists but is not writable: ${DATA_ROOT}" >&2
    exit 1
  fi
  log "Trying to make current user owner of DATA_ROOT"
  "${SUDO[@]}" chown -R "$(id -u):$(id -g)" "${DATA_ROOT}"
fi

if [[ "${SKIP_APT}" -eq 0 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing system packages"
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends \
      git git-lfs curl wget aria2 unzip p7zip-full tar tmux htop tree \
      ffmpeg build-essential pkg-config \
      python3 python3-venv python3-pip python3-dev \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libsndfile1
    git lfs install || true
  else
    log "apt-get not found; skipping system package installation"
  fi
else
  log "Skipping apt-get installation"
fi

log "Creating workspace directories"
mkdir -p \
  "${DATA_ROOT}" \
  "${DATASET_ROOT}/lrs2/downloads" \
  "${DATASET_ROOT}/lrs2/extracted" \
  "${DATASET_ROOT}/lrs3/downloads/part1" \
  "${DATASET_ROOT}/lrs3/downloads/part2" \
  "${DATASET_ROOT}/lrs3/extracted" \
  "${CACHE_ROOT}/manifests" \
  "${CACHE_ROOT}/datasets" \
  "${CACHE_ROOT}/checkpoints" \
  "${CACHE_ROOT}/reports" \
  "${CACHE_ROOT}/pretrained_models/vtp" \
  "${LOG_ROOT}"

log "Creating virtual environment"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Upgrading pip tooling"
python -m pip install --upgrade pip wheel setuptools

case "${TORCH_FLAVOR}" in
  cu121)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  cu124)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    ;;
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "[error] Unsupported TORCH_FLAVOR='${TORCH_FLAVOR}'. Use cu121, cu124, or cpu." >&2
    exit 1
    ;;
esac

log "Installing PyTorch from ${TORCH_INDEX_URL}"
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  log "Installing repo requirements.txt"
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
else
  log "requirements.txt not found; skipping"
fi

log "Installing VTP/l2t_arch dependencies"
python -m pip install \
  opencv-python-headless \
  decord \
  pandas \
  transformers \
  configargparse \
  einops \
  linear-attention-transformer \
  local-attention \
  gdown \
  soundfile \
  librosa \
  tqdm

log "Cloning external VTP repo if needed"
mkdir -p "${PROJECT_ROOT}/external"
if [[ ! -d "${PROJECT_ROOT}/external/vtp/.git" ]]; then
  if [[ -d "${PROJECT_ROOT}/external/vtp" ]]; then
    echo "[warn] ${PROJECT_ROOT}/external/vtp exists but is not a git repo; leaving it untouched."
  else
    git clone https://github.com/prajwalkr/vtp.git "${PROJECT_ROOT}/external/vtp"
  fi
else
  git -C "${PROJECT_ROOT}/external/vtp" status --short >/dev/null
  echo "[ok] external/vtp already exists"
fi

if [[ "${DOWNLOAD_VTP}" -eq 1 ]]; then
  log "Downloading official VTP checkpoints"
  python -m l2s_itw.providers.download_vtp_checkpoints \
    --variant extended \
    --target feature_extractor \
    --target ft_lrs2 \
    --target ft_lrs3 \
    --output-dir "${CACHE_ROOT}/pretrained_models/vtp"

  log "Checking VTP install with ft_lrs2 checkpoint"
  python -m l2s_itw.providers.check_vtp \
    --repo-dir "${PROJECT_ROOT}/external/vtp" \
    --ckpt-path "${CACHE_ROOT}/pretrained_models/vtp/ft_lrs2.pth" \
    --cnn-ckpt-path "${CACHE_ROOT}/pretrained_models/vtp/feature_extractor.pth"
else
  log "Skipping VTP checkpoint download. Run again with --download-vtp when ready."
fi

log "Writing reusable environment file"
cat > "${PROJECT_ROOT}/server_env.sh" <<EOF
export DATA_ROOT="${DATA_ROOT}"
export DATASET_ROOT="${DATASET_ROOT}"
export CACHE_ROOT="${CACHE_ROOT}"
export LOG_ROOT="${LOG_ROOT}"
export PROJECT_ROOT="${PROJECT_ROOT}"
export LRS2_ROOT="${DATASET_ROOT}/lrs2"
export LRS3_ROOT="${DATASET_ROOT}/lrs3"
source "${VENV_DIR}/bin/activate"
EOF

log "Running import smoke test"
python - <<'PY'
import importlib
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

modules = [
    "decord",
    "pandas",
    "transformers",
    "configargparse",
    "einops",
    "local_attention",
    "linear_attention_transformer",
    "gdown",
    "l2t_arch",
    "l2s_itw",
]
for name in modules:
    importlib.import_module(name)
    print(name, "ok")
PY

cat <<EOF

[done] Basic server setup is complete.

Use this in future shells:
  cd "${PROJECT_ROOT}"
  source server_env.sh

Next manual steps:
  1. Download LRS2 with your Oxford credentials.
  2. Download LRS3 from the two Google Drive folders.
  3. Extract datasets and set LRS2_DATA_DIR / LRS3_DATA_DIR.
  4. Build raw manifests, VTP cache, l2t_arch datasets, then train/evaluate.

The full step-by-step commands are in:
  Huong_dan_server_LRS2_LRS3_l2t_arch.docx

EOF
