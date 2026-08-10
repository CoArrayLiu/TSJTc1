#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="${1:-configs/c2/pemsbay_to_metrla.yaml}"
export PYTHONPATH="$repo_root/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python -m tsjt_c2.run --config "$config"
