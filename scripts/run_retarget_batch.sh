#!/usr/bin/env bash
# Batch camera-aligned retargeting driven by a run config under
# scripts/retarget_configs/. Relative paths in a config resolve against the
# repository root (override with SPARK_WORKSPACE_ROOT).
#
#   bash scripts/run_retarget_batch.sh openaoe
#   bash scripts/run_retarget_batch.sh human_touch --shard X001 --mano-assets /path/to/mano_v1_2
#   bash scripts/run_retarget_batch.sh openaoe --limit 2 --dry-run
#
# Any extra argument is forwarded to data_tools/batch_retarget_camera_aligned.py and
# overrides the config, e.g. --output-dir, --workers, --overwrite.
set -euo pipefail

CONFIG_ARG="${1:-}"
if [ -z "${CONFIG_ARG}" ]; then
  echo "usage: run_retarget_batch.sh <config-name|config.json> [batch args...]" >&2
  exit 64
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "${SCRIPT_DIR}")"

if [ -f "${CONFIG_ARG}" ]; then
  CONFIG="$(cd "$(dirname "${CONFIG_ARG}")" && pwd)/$(basename "${CONFIG_ARG}")"
else
  CONFIG="${SCRIPT_DIR}/retarget_configs/${CONFIG_ARG}.json"
fi
if [ ! -f "${CONFIG}" ]; then
  echo "Run config not found: ${CONFIG_ARG}" >&2
  echo "Available:" >&2
  ls -1 "${SCRIPT_DIR}/retarget_configs" >&2
  exit 66
fi

echo "RETARGET_RUN config=${CONFIG}" >&2
exec python "${PACKAGE_ROOT}/data_tools/batch_retarget_camera_aligned.py" \
  --config "${CONFIG}" \
  "$@"
