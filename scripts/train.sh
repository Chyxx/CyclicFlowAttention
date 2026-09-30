#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_CONFIG="${MODEL_CONFIG:-${ROOT}/configs/cyclic_flow_attention_400m.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-fla-hub/gla-1.3B-100B}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to a tokenized Hugging Face Dataset saved to disk}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/cyclic_flow_attention_400m}"

NGPU="${NGPU:-8}"
NNODE="${NNODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-0}"

STEPS="${STEPS:-30720}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WORKERS="${WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-32}"
SAVE_STEPS="${SAVE_STEPS:-1024}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SEED="${SEED:-42}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
REPORT_TO="${REPORT_TO:-none}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ "${RESUME_FROM_CHECKPOINT}" == "auto" ]]; then
  RESUME_FROM_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [[ -z "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "No checkpoint found in ${OUTPUT_DIR}" >&2
    exit 1
  fi
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

torchrun \
  --nnodes="${NNODE}" \
  --nproc_per_node="${NGPU}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  -m training.train \
  --model_name_or_path "${MODEL_CONFIG}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --from_config true \
  --do_train \
  --cache_dir "${DATASET_PATH}" \
  --dataloader_num_workers "${WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps "${LOGGING_STEPS}" \
  --include_num_input_tokens_seen \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine_with_min_lr \
  --warmup_steps "${WARMUP_STEPS}" \
  --optim adamw_torch_fused \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --max_steps "${STEPS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --seed "${SEED}" \
  --bf16 \
  --report_to "${REPORT_TO}" \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config '{"version":2,"reshard_after_forward":false,"cpu_ram_efficient_loading":false}' \
  "${RESUME_ARGS[@]}" \
  "$@"
