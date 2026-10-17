#!/bin/bash
# Offline precompute of state/action normalization statistics for a recipe.
#
#   bash scripts/precompute_stats.sh <recipe> [--data NAME] [overrides...]
#
# Takes the same arguments as scripts/train.sh. Training runs with
# SPARK_0_REQUIRE_PRECOMPUTED_STATS=1 by default: every rank READS the
# content-addressed caches under <dataset>/meta/stats_xpolicy*__<hash>.json and
# fails fast when one is missing, instead of scanning the dataset on rank 0 at
# launch. This builds those caches ahead of time, in one CPU process.
#
# It goes through train.sh's config resolution, so the cache hash (pose
# formats, camera frame, history frames, action indices, exclusions, data/meta
# fingerprints) matches the training launch exactly. Re-run it whenever the
# data or a stats-relevant knob changes; a new hash leaves old caches untouched.
set -euo pipefail

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env SPARK_0_PRECOMPUTE_STATS_ONLY=1 GPUS="${GPUS:-0}" bash "${SCRIPT_DIR}/train.sh" "$@"
