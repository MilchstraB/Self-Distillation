#!/bin/bash
set -euxo pipefail

# =================== User Configuration =====================
MODEL_PATH="Qwen/Qwen3-4B"
EXP_NAME="Baseline"
SAVE_PATH="$(dirname "$MODEL_PATH")/lm_eval_results.log"
PROJECT_NAME="Self-Distillation"
# =================== User Configuration =====================

export HF_ENDPOINT=https://hf-mirror.com
export HF_ALLOW_CODE_EVAL="1"

CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained="${MODEL_PATH}",max_model_len=8192,tensor_parallel_size=1 \
    --tasks hellaswag,humaneval,ifeval,mmlu,truthfulqa_gen,winogrande \
    --gen_kwargs temperature=0.7,top_p=0.8,top_k=20,min_p=0,do_sample=True \
    --batch_size auto \
    --apply_chat_template \
	--fewshot_as_multiturn \
    --seed 42 \
    --confirm_run_unsafe_code \
    --wandb_args project="${PROJECT_NAME}",name="${EXP_NAME}_eval" 2>&1 | tee "${SAVE_PATH}"