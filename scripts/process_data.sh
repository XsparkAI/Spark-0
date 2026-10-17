#!/bin/bash
# Convert Spark HDF5 episodes into one LeRobot v3.0 training dataset.
#
#   bash scripts/process_data.sh <dataset_name> <input_dir> [<input_dir>...] [converter options...]
#
# Each <input_dir> is searched recursively for *.hdf5 / *.h5; a bare name that
# is not a directory is looked up under ${SPARK_0_DATA_ROOT:-data}/raw/. All
# inputs are merged into
#   ${SPARK_0_DATA_ROOT:-data}/lerobot/<dataset_name>/
# which `scripts/train.sh <recipe> --data <dataset_name>` reads.
#
# Examples:
#   bash scripts/process_data.sh pour_water data/raw/pour_water
#   bash scripts/process_data.sh kitchen_mix pour_water stack_bowls --max_episode 50
#   bash scripts/process_data.sh openaoe data/retargeted/openaoe --finger-only
#
# Options after the inputs go to data_tools/transform_lerobot_v30_format_tianji_ee.py
# (--max_episode, --threads, --num_shards/--shard_index, --mano-assets, --finger-only, ...).
set -euo pipefail

if [[ $# -lt 2 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SPARK_0_DATA_ROOT:-${REPO_ROOT}/data}"
dataset_name=$1
shift

input_dirs=()
while [[ $# -gt 0 && "$1" != -* ]]; do
    if [[ -d "$1" ]]; then
        input_dirs+=("$(realpath "$1")")
    elif [[ -d "${DATA_ROOT}/raw/$1" ]]; then
        input_dirs+=("${DATA_ROOT}/raw/$1")
    else
        echo "[process_data] input directory not found: $1 (also tried ${DATA_ROOT}/raw/$1)" >&2
        exit 66
    fi
    shift
done
if [[ ${#input_dirs[@]} -eq 0 ]]; then
    echo "[process_data] at least one input directory is required" >&2
    exit 64
fi

output_root="${DATA_ROOT}/lerobot"
mkdir -p "${output_root}"
echo "[process_data] ${input_dirs[*]} -> ${output_root}/${dataset_name}/"

cd "${REPO_ROOT}"
exec python -m data_tools.transform_lerobot_v30_format_tianji_ee \
    --input_dir "${input_dirs[@]}" \
    --repo_id "${dataset_name}" \
    --output_root "${output_root}" \
    "$@"
