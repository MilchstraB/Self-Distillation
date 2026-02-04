#!/bin/bash
set -euxo pipefail

# =================== User Configuration =====================
# Please modify these variables according to your environment
# ============================================================
MODEL_PATH="<model_path>"
OUTPUT_DIR="<output_path>"
TRAIN_DATA="data/tooluse_data/train_data.json"
VAL_DATA="data/tooluse_data/eval_data.json"
HF_HOME=".cache"

PROJECT_NAME="Self-Distillation"
EXP_NAME="Baseline"

# =================== Script Execution ===================
# You shouldn't need to modify anything below this line
# ========================================================
export WANDB_PROJECT="${PROJECT_NAME}"
export HF_HOME="${HF_HOME}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${OUTPUT_DIR}/${EXP_NAME}/${TIMESTAMP}"

mkdir -p "${SAVE_DIR}"

deepspeed main.py \
    --deepspeed ./scripts/zero2.json \
    --use_vllm True \
    --vllm_mode "colocate" \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_enable_sleep_mode True \
    --vllm_importance_sampling_correction True \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${SAVE_DIR}" \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --seed 42 \
    --train_path "${TRAIN_DATA}" \
    --eval_path "${VAL_DATA}" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --fp16 False \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --save_steps 100 \
    --save_total_limit 1 \
    --save_only_model True \
    --max_grad_norm 1 \
    --report_to "wandb" \
    --log_completions False \
    --sync_ref_model True \
    --ref_model_sync_steps 1 \
    --ref_model_mixup_alpha 0.01 \
    --num_loss_tokens_to_skip 3 \
    --run_name "${EXP_NAME}" > >(tee -a "${SAVE_DIR}/train.log") 2>&1