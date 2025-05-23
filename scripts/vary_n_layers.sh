#!/usr/bin/env bash
# Sweep number of layers for eager generation
set -euo pipefail

TOKENIZER="../tokenizer.model"
CSV="../collected_data/torch_vary_n_layers.csv"
PROMPT_FILE="../prompt.txt"
TORCHRUN=(torchrun --standalone --nproc_per_node=1)

LAYERS=(4 6 8 10 12)

for l in "${LAYERS[@]}"; do
  "${TORCHRUN[@]}" ../eager_generation.py \
    --tokenizer_path "$TOKENIZER" \
    --prompt_file "$PROMPT_FILE" \
    --batch_size 32 \
    --max_seq_len 400 \
    --dim 512 \
    --n_heads 4 \
    --n_layers "$l" \
    --results_csv "$CSV"
done

