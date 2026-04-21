#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

MODEL_NAME="Qwen/Qwen3.5-4B"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/score_result_qwen35_vl_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/qwen35_vl_4b_iqa_lora}"
USE_4BIT="${USE_4BIT:-0}"

extra_args=()
if [[ "$USE_4BIT" == "1" ]]; then
  extra_args+=(--load-in-4bit)
fi

python "$SCRIPT_DIR/train_qwen35_vl_lora.py" \
  --model-name "$MODEL_NAME" \
  --train-file "$DATA_DIR/training.jsonl" \
  --eval-file "$DATA_DIR/validation.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-root "$SCRIPT_DIR" \
  --max-length 2048 \
  --num-train-epochs 2 \
  --learning-rate 2e-4 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --save-total-limit 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --bf16 \
  --report-to none \
  "${extra_args[@]}"