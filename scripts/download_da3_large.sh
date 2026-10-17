#!/bin/bash
# Download Depth-Anything-3 LARGE teacher weights for DA3 spatial alignment.
# Destination matches train YAML teacher_model_path:
#   ./playground/Pretrained_models/DA3-LARGE  (relative to third_party/starVLA/)
#
# Usage:
#   bash scripts/download_da3_large.sh
#   OUT_DIR=/abs/path/DA3-LARGE bash scripts/download_da3_large.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_da3_large.sh   # CN mirror
#
# Requires: huggingface_hub (`pip install -U huggingface_hub`) with the
# `hf` CLI (preferred) or a Python import fallback. Auth only needed if the
# repo is gated for your account (DA3-LARGE is public as of writing).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
STARVLA_ROOT="${REPO_ROOT}/third_party/starVLA"

REPO_ID="${HF_REPO_ID:-depth-anything/DA3-LARGE}"
OUT_DIR="${OUT_DIR:-${STARVLA_ROOT}/playground/Pretrained_models/DA3-LARGE}"

mkdir -p "$(dirname "${OUT_DIR}")"

echo "[download_da3_large] repo=${REPO_ID}"
echo "[download_da3_large] out=${OUT_DIR}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
    echo "[download_da3_large] HF_ENDPOINT=${HF_ENDPOINT}"
fi

if command -v hf >/dev/null 2>&1; then
    hf download "${REPO_ID}" --local-dir "${OUT_DIR}"
elif python -c "import huggingface_hub" >/dev/null 2>&1; then
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${REPO_ID}", local_dir="${OUT_DIR}")
PY
else
    echo "[download_da3_large] ERROR: need hf CLI or python package huggingface_hub" >&2
    echo "[download_da3_large]   pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

# Sanity: DepthAnything3.from_pretrained expects a config + weights present.
if [[ ! -f "${OUT_DIR}/config.json" ]]; then
    echo "[download_da3_large] WARNING: ${OUT_DIR}/config.json missing; check the download." >&2
fi

echo "[download_da3_large] done -> ${OUT_DIR}"
echo "[download_da3_large] point framework.spatial_alignment.teacher_model_path at this dir"
echo "[download_da3_large] (and pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3 in the train env)."
