#!/usr/bin/env bash
# Sweep dimension size for eager generation
set -euo pipefail

TOKENIZER="../tokenizer.model"
CSV="../collected_data/torch_vary_n_dims.csv"
PROMPT_FILE="../prompt.txt"
TORCHRUN=(torchrun --standalone --nproc_per_node=1)

DIMS=(128 256 512 1024 2048)

for d in "${DIMS[@]}"; do
  "${TORCHRUN[@]}" ../eager_generation.py \
    --tokenizer_path "$TOKENIZER" \
    --prompt_file "$PROMPT_FILE" \
    --batch_size 32 \
    --max_seq_len 400 \
    --n_layers 8 \
    --n_heads 4 \
    --dim "$d" \
    --results_csv "$CSV"
done

