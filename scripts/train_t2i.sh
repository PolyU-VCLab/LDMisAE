#!/bin/bash
# Local text-to-image training (JiT-H/half, 512px, text-conditional) -- two stages.
# Base = the 512 class-conditional recipe (z_norm rms + xp_ste + enc_bp); the text branch is
# Qwen3 + BLIP3o. Stage semantics are explicit instead of the epoch-wise lr schedule:
# STAGE=1 -> --freeze_backbone --enc_lr_scale 0 : backbone frozen, only the text branch trains
# STAGE=2 -> --enc_lr_scale ${ENC_LR_SCALE:-1.0} : backbone unfrozen, symmetric lr, no decay
#
# Environment variables (no default = must be given):
# TEXT_ENCODER_PATH Qwen3 weights dir (local path or hub id) (required)
# BLIP3O_PATH BLIP3o caption json/dir (required)
# BATCH_SIZE per-GPU batch size (required)
# BLIP3O_60K_PATH optional second caption source []
# INIT_CKPT initialisation checkpoint (.pth) from the 512 line []
# RESUME_CKPT run dir to resume from (contains checkpoint-last) []
# OUTPUT_DIR run directory [output_dir/t2i_stage${STAGE}]
# STAGE 1|2, BLR [5e-5], EPOCHS [100000], WARMUP_EPOCHS [0], MAX_TOTAL_STEPS [0=off]
# SAVE_EVERY_N_STEPS / EVAL_EVERY_N_STEPS [30000], NPROC [1], NNODES [1], MASTER_PORT [29500]
# GEN_BSZ [128], NUM_IMAGES [5000], CFG [2.2], INTERVAL_MIN [0.1], INTERVAL_MAX [1.0]
# TRYRUN=1 -> 1-step smoke run
# LPIPS_MODEL_PATH [] LPIPS/VGG weights for the LPIPS term (train.py --lpips_model_path)
#
# Example: TEXT_ENCODER_PATH=/models/Qwen3_1.7B BLIP3O_PATH=/data/blip3o STAGE=1 \
# BATCH_SIZE=16 NPROC=8 bash scripts/train_t2i.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:?set TEXT_ENCODER_PATH to the Qwen3 text-encoder weights}"
BLIP3O_PATH="${BLIP3O_PATH:?set BLIP3O_PATH to the BLIP3o caption path}"
BATCH_SIZE="${BATCH_SIZE:?set BATCH_SIZE (per-GPU batch size)}"
BLIP3O_60K_PATH="${BLIP3O_60K_PATH:-}"
INIT_CKPT="${INIT_CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
STAGE="${STAGE:-1}"
BLR="${BLR:-5e-5}"
EPOCHS="${EPOCHS:-100000}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
MAX_TOTAL_STEPS="${MAX_TOTAL_STEPS:-0}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-30000}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-30000}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/output_dir/t2i_stage${STAGE}}"
MODEL="${MODEL:-JiT-H/half}"          # e.g. SiT-B/half for the backbone ablation
EPOCHS="${EPOCHS:-350}"            # paper ablations use EPOCHS=100
NPROC="${NPROC:-1}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
EVAL_PROMPTS_PATH="${EVAL_PROMPTS_PATH:-${REPO}/ldm_is_ae/data/eval_prompts.txt}"
EVAL_FREQ="${EVAL_FREQ:-10}"
GEN_BSZ="${GEN_BSZ:-128}"
NUM_IMAGES="${NUM_IMAGES:-5000}"
CFG="${CFG:-2.2}"
INTERVAL_MIN="${INTERVAL_MIN:-0.1}"
INTERVAL_MAX="${INTERVAL_MAX:-1.0}"
ACCUM_ITER="${ACCUM_ITER:-1}"

# stage-dependent flags (STAGE whitelist: anything else than 1 is the joint stage)
FREEZE_ARG=()
ENC_ARG=(--enc_lr_scale "${ENC_LR_SCALE:-1.0}")
if [ "${STAGE}" = "1" ]; then
  FREEZE_ARG=(--freeze_backbone)
  ENC_ARG=(--enc_lr_scale "${ENC_LR_SCALE:-0}")
elif [ "${STAGE}" != "2" ]; then
  echo "train_t2i.sh: STAGE must be 1 or 2 (got '${STAGE}')" >&2
  exit 2
fi

INIT_ARG=()
[ -n "${INIT_CKPT}" ] && INIT_ARG=(--init_from_ckpt "${INIT_CKPT}")
RESUME_ARG=()
[ -n "${RESUME_CKPT}" ] && RESUME_ARG=(--resume "${RESUME_CKPT}")
TRYRUN_ARG=()
[ "${TRYRUN:-0}" = "1" ] && TRYRUN_ARG=(--tryrun)
LPIPS_ARG=()
[ -n "${LPIPS_MODEL_PATH:-}" ] && LPIPS_ARG=(--lpips_model_path "${LPIPS_MODEL_PATH}")

JIT_MAIN_ARGS=(
-m ldm_is_ae.train
--model "${MODEL}"
--depth_enc 2 --depth_dec 30
--lpips_weight 1 --v_mse_weight 10
"${LPIPS_ARG[@]}"
--zc 128 --pixel_patch_size 32 --num_workers 4
--proj_dropout 0.2
--P_mean -0.8 --P_std 0.8 --save_last_freq 1
--xp_ste --enc_bp 1 --z_norm rms
--img_size 512 --noise_scale 1.0 --dec_lr_scale 1.0
"${ENC_ARG[@]}"
"${FREEZE_ARG[@]}"
--batch_size "${BATCH_SIZE}" --accum_iter "${ACCUM_ITER}" --blr "${BLR}"
--epochs "${EPOCHS}" --warmup_epochs "${WARMUP_EPOCHS}"
--clip_grad_norm 1.0 --lr_schedule constant
--max_total_optimizer_steps "${MAX_TOTAL_STEPS}"
--save_every_n_steps "${SAVE_EVERY_N_STEPS}" --eval_every_n_steps "${EVAL_EVERY_N_STEPS}"
--text_dim 2048 --text_len 128
--text_encoder_path "${TEXT_ENCODER_PATH}"
--blip3o_path "${BLIP3O_PATH}" --blip3o_60k_path "${BLIP3O_60K_PATH}"
"${INIT_ARG[@]}"
"${RESUME_ARG[@]}"
--output_dir "${OUTPUT_DIR}"
--eval_prompts "${EVAL_PROMPTS_PATH}"
--online_eval --eval_freq "${EVAL_FREQ}"
--gen_bsz "${GEN_BSZ}" --num_images "${NUM_IMAGES}" --cfg "${CFG}"
--interval_min "${INTERVAL_MIN}" --interval_max "${INTERVAL_MAX}"
--fp32
"${TRYRUN_ARG[@]}"
)

if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  JIT_MAIN_ARGS+=(${EXTRA_ARGS})
fi

cd "${REPO}"
if [ "${NPROC}" -gt 1 ] || [ "${NNODES}" -gt 1 ]; then
  "${PYTHON}" -m torch.distributed.run --nproc_per_node="${NPROC}" --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    "${JIT_MAIN_ARGS[@]}"
else
  "${PYTHON}" "${JIT_MAIN_ARGS[@]}"
fi
