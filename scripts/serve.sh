#!/bin/bash
# Serve a Spark-0 checkpoint over websocket (spark0.serving.server).
#
#   bash scripts/serve.sh <ckpt> [--gpu ID] [server options...]
#
# <ckpt> is a weight file, a run directory, or a run name under
# ${SPARK_0_CKPT_ROOT:-checkpoints}. Everything after it is forwarded, e.g.
#   bash scripts/serve.sh Spark-0 --gpu 0 --port 8000 --set tactile=false
#   bash scripts/serve.sh sft_my_task --model-config sft_spark_0 --execute-horizon 16
# Run `python -m spark0.serving.server --help` for the full list.
set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ckpt=$1
shift

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

cd "${REPO_ROOT}"
exec python -m spark0.serving.server --ckpt "${ckpt}" "${args[@]+"${args[@]}"}"
