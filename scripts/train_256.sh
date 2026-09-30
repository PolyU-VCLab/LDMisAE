#!/bin/bash
# Local training on ImageNet-256 (class-conditional, JiT-H/half: DiT-E 2 blocks / DiT-D 30 blocks).
#
# Every knob is an environment variable; the ones without a default must be given:
# IMAGENET_PATH parent folder of `train/` (the loader appends `train/`) (required)
# BATCH_SIZE per-GPU batch size (required)
# ACCUM_ITER gradient accumulation factor (global batch = BATCH_SIZE*ACCUM_ITER*NPROC) [1]
# OUTPUT_DIR run directory (checkpoints / logs) [<repo>/output_dir/imagenet256]
# NPROC GPUs on this node [1]
# NNODES nodes (multi-node: also set MASTER_ADDR/MASTER_PORT/NODE_RANK) [1]
# MASTER_PORT rendezvous port [29500]
# TRYRUN=1 one-step smoke run instead of the full schedule [0]
# LPIPS_MODEL_PATH [] LPIPS/VGG weights for the LPIPS term (train.py --lpips_model_path)
#
# Example: IMAGENET_PATH=/data/imagenet256 BATCH_SIZE=64 NPROC=8 bash scripts/train_256.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

IMAGENET_PATH="${IMAGENET_PATH:?set IMAGENET_PATH to the ImageNet 256 train image folder}"
BATCH_SIZE="${BATCH_SIZE:?set BATCH_SIZE (per-GPU batch size)}"
ACCUM_ITER="${ACCUM_ITER:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/output_dir/imagenet256}"
MODEL="${MODEL:-JiT-H/half}"          # e.g. SiT-B/half for the backbone ablation
EPOCHS="${EPOCHS:-350}"            # paper ablations use EPOCHS=100
NPROC="${NPROC:-1}"
NNODES="${NNODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"

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
--zc 128 --pixel_patch_size 16 --num_workers 4
--proj_dropout 0.2
--P_mean -0.8 --P_std 0.8 --save_last_freq 5 --xp_ste --enc_bp 1 --z_norm rms
--img_size 256 --noise_scale 1.0 --enc_lr_scale 0.1 --enc_lr_scale_decay 0.3333 --enc_lr_scale_decay_epochs 50 --dec_lr_scale 1.0
--batch_size "${BATCH_SIZE}" --accum_iter "${ACCUM_ITER}" --blr 5e-5
--epochs "${EPOCHS}" --warmup_epochs 5
--clip_grad_norm 1.0 --lr_schedule constant
--gen_bsz 128 --num_images 5000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0
--output_dir "${OUTPUT_DIR}"
--resume "${OUTPUT_DIR}"
--data_path "${IMAGENET_PATH}" --online_eval "${TRYRUN_ARG[@]}" --fp32 \
--dec_lr_scale_decay 0.1 --dec_lr_scale_decay_epochs 200 \
--enc_lr_scale_end_epoch 200
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
