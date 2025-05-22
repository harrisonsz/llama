#!/usr/bin/env bash
# Sweep max sequence length for eager generation
set -euo pipefail

TOKENIZER="../tokenizer.model"
CSV="../collected_data/vary_max_seq_len.csv"
PROMPT_FILE="../prompt.txt"
TORCHRUN=(torchrun --standalone --nproc_per_node=1)

SEQ_LENS=(100 200 400 800 1600)

for s in "${SEQ_LENS[@]}"; do
  "${TORCHRUN[@]}" ../eager_generation.py \
    --tokenizer_path "$TOKENIZER" \
    --prompt_file "$PROMPT_FILE" \
    --batch_size 32 \
    --n_layers 8 \
    --dim 512 \
    --n_heads 4 \
    --max_seq_len "$s" \
    --results_csv "$CSV"
done

