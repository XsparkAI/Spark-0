#!/bin/bash
# Install Spark-0 training + serving into the active Python environment
# (tested with Python 3.10, CUDA 12.4).
#
#   bash install.sh            # train + serve
#   bash install.sh --data     # also the data conversion / retargeting tools
#
# A robot-side client only needs `pip install -e .` (numpy, msgpack, websockets).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${REPO_ROOT}/third_party/starVLA"

with_data=0
for arg in "$@"; do
    case "${arg}" in
        --data) with_data=1 ;;
        *) echo "Usage: bash install.sh [--data]" >&2; exit 64 ;;
    esac
done

python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install -r "${STARVLA_ROOT}/requirements.txt"
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
# Pillow is pinned because training (dataloader _pack_sample) and serving
# (spark0/policy.py) both resize through PIL's default resample kernel; the pin
# keeps that kernel identical across environments. Installed after
# requirements.txt (which lists an unpinned `pillow`) so this version wins.
python -m pip install h5py pandas numpy==1.26.4 opencv-python==4.10.0.84 pillow==12.2.0
python -m pip install -e "${STARVLA_ROOT}"
if [[ "${with_data}" == "1" ]]; then
    python -m pip install -e "${REPO_ROOT}[data]"
else
    python -m pip install -e "${REPO_ROOT}"
fi
