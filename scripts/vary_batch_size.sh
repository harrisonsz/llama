#!/usr/bin/env bash
# Sweep batch size for eager generation
set -euo pipefail

TOKENIZER="../tokenizer.model"
PROMPT_FILE="../prompt.txt"
CSV="../collected_data/torch_vary_batch_size.csv"
TORCHRUN=(torchrun --standalone --nproc_per_node=1)

BATCH_SIZES=(4 8 16 32 64)

for b in "${BATCH_SIZES[@]}"; do
  "${TORCHRUN[@]}" ../eager_generation.py \
    --tokenizer_path "$TOKENIZER" \
    --prompt_file "$PROMPT_FILE" \
    --max_seq_len 400 \
    --n_layers 12 \
    --dim 512 \
    --n_heads 4 \
    --batch_size "$b" \
    --results_csv "$CSV"
done

