#!/bin/bash
# FID / Inception-Score evaluation in the ADM (evaluator_adm, TensorFlow) caliber,
# torch implementation (torch-fidelity Inception features).
#
# usage: bash evaluate.sh <sample_dir> <ref_stats.npz> [tag]
# or: SAMPLE_PATH=<sample_dir> REF_STATS=<ref_stats.npz> bash evaluate.sh
# <ref_stats.npz> holds the reference mu/sigma (ADM VIRTUAL_imagenet256_labeled.npz /
# VIRTUAL_imagenet512.npz, or the JiT statistics of the matching resolution).
# WEIGHTS=<inception .pth> torch-fidelity Inception-V3 weights [torch-fidelity cache / TORCH_HOME]
set -u
SAMPLE=${1:-${SAMPLE_PATH:-}}
REF=${2:-${REF_STATS:-}}
[ -n "${SAMPLE}" ] || { echo "usage: $0 <sample_dir> <ref_stats.npz> [tag]" >&2; exit 2; }
[ -n "${REF}" ] || { echo "usage: $0 <sample_dir> <ref_stats.npz> [tag]" >&2; exit 2; }
TAG=${3:-$(basename "${SAMPLE}")}
WEIGHTS_ARG=()
[ -n "${WEIGHTS:-}" ] && WEIGHTS_ARG=(--weights "${WEIGHTS}")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${HERE}/../ldm_is_ae/eval/fid_is.py" \
  --sample_path "${SAMPLE}" --ref_stats "${REF}" --tag "${TAG}" \
  "${WEIGHTS_ARG[@]}"
