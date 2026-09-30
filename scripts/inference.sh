#!/bin/bash
# Local sampling (class-conditional) -- writes the images that scripts/evaluate.sh consumes.
#
# Environment variables (no default = must be given):
# CKPT run directory holding checkpoint-last.pth, or an explicit .pth file (required)
# IMG_SIZE [256], MODEL [JiT-H/half], ZC [128], DEPTH_ENC [2], DEPTH_DEC [30]
# PIXEL_PATCH_SIZE[16 for 256 / 32 for 512]
# CFG classifier-free guidance scale [2.25]
# INTERVAL_MIN/INTERVAL_MAX [0.1 / 1.0]
# NUM_IMAGES [50000], GEN_BSZ [128], EPOCH (folder tag) [0], EMA_MODE [ema1]
# OUTPUT_DIR [<repo>/output_dir/inference/<tag>]
# LPIPS_WEIGHT [0] LPIPS term weight; the sampling path never uses it, so 0 = vgg.pth not needed
# LPIPS_MODEL_PATH [] LPIPS/VGG weights, read only when LPIPS_WEIGHT != 0
# XP_STE=0 disable the 8-bit STE quantisation used by the paper models [1]
# RANDOM_IDX=0 keep the class order instead of shuffling [1]
# PR_STEPS [0], DELTAT [0], SCHEDULE_POWER [1.0]
# NPROC [1], NNODES [1], MASTER_ADDR/MASTER_PORT/NODE_RANK
#
# Examples:
# CKPT=/runs/imagenet256/checkpoint-last.pth IMG_SIZE=256 CFG=2.25 NUM_IMAGES=50000 bash scripts/inference.sh
# CKPT=/runs/imagenet512 NPROC=8 IMG_SIZE=512 CFG=2.2 bash scripts/inference.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

CKPT="${CKPT:?set CKPT to a run directory (checkpoint-last.pth) or an explicit .pth file}"
MODEL="${MODEL:-JiT-H/half}"
IMG_SIZE="${IMG_SIZE:-256}"
ZC="${ZC:-128}"
DEPTH_ENC="${DEPTH_ENC:-2}"
DEPTH_DEC="${DEPTH_DEC:-30}"
if [ -z "${PIXEL_PATCH_SIZE:-}" ]; then
  if [ "${IMG_SIZE}" -ge 512 ]; then PIXEL_PATCH_SIZE=32; else PIXEL_PATCH_SIZE=16; fi
fi
CFG="${CFG:-2.25}"
INTERVAL_MIN="${INTERVAL_MIN:-0.1}"
INTERVAL_MAX="${INTERVAL_MAX:-1.0}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
GEN_BSZ="${GEN_BSZ:-128}"
EPOCH="${EPOCH:-0}"
EMA_MODE="${EMA_MODE:-ema1}"
PR_STEPS="${PR_STEPS:-0}"
DELTAT="${DELTAT:-0.0}"
SCHEDULE_POWER="${SCHEDULE_POWER:-1.0}"
TAG="${TAG:-cfg${CFG}_imin${INTERVAL_MIN}_imax${INTERVAL_MAX}_ep${EPOCH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/output_dir/inference/${TAG}}"
NPROC="${NPROC:-1}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"

if [ -d "${CKPT}" ]; then CKPT_ARG=(--resume "${CKPT}"); else CKPT_ARG=(--checkpoint "${CKPT}"); fi
STE_ARG=(); [ "${XP_STE:-1}" != "0" ] && STE_ARG=(--xp_ste)
RIDX_ARG=(); [ "${RANDOM_IDX:-1}" != "0" ] && RIDX_ARG=(--random_idx)
LPIPS_W_ARG=(); [ -n "${LPIPS_WEIGHT:-}" ] && LPIPS_W_ARG=(--lpips_weight "${LPIPS_WEIGHT}")
LPIPS_ARG=(); [ -n "${LPIPS_MODEL_PATH:-}" ] && LPIPS_ARG=(--lpips_model_path "${LPIPS_MODEL_PATH}")

INFER_ARGS=(
-m ldm_is_ae.sample
--model "${MODEL}" "${CKPT_ARG[@]}"
--img_size "${IMG_SIZE}" --zc "${ZC}" --pixel_patch_size "${PIXEL_PATCH_SIZE}"
--depth_enc "${DEPTH_ENC}" --depth_dec "${DEPTH_DEC}"
--num_images "${NUM_IMAGES}" --gen_bsz "${GEN_BSZ}"
--cfg "${CFG}" --interval_min "${INTERVAL_MIN}" --interval_max "${INTERVAL_MAX}"
--deltat "${DELTAT}" --schedule_power "${SCHEDULE_POWER}" --pr_steps "${PR_STEPS}"
--ema_mode "${EMA_MODE}" --epoch "${EPOCH}" --output_dir "${OUTPUT_DIR}"
"${STE_ARG[@]}" "${RIDX_ARG[@]}" "${LPIPS_W_ARG[@]}" "${LPIPS_ARG[@]}"
)

if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  INFER_ARGS+=(${EXTRA_ARGS})
fi

cd "${REPO}"
if [ "${NPROC}" -gt 1 ] || [ "${NNODES}" -gt 1 ]; then
  "${PYTHON}" -m torch.distributed.run --nproc_per_node="${NPROC}" --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    "${INFER_ARGS[@]}"
else
  "${PYTHON}" "${INFER_ARGS[@]}"
fi
echo "samples -> ${OUTPUT_DIR}/imgs"
