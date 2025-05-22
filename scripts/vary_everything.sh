#!/usr/bin/env bash
# Exhaustively sweep parameters across implementations
set -euo pipefail

TOKENIZER="../tokenizer.model"
CSV="../collected_data/vary_everything.csv"
PROMPT_FILE="../prompt.txt"
TORCHRUN=(torchrun --standalone --nproc_per_node=1)

BATCH_SIZES=(4 8 16 32 64)
SEQ_LENS=(100 200 400 800 1600)
LAYERS=(4 6 8 10 12)
DIMS=(128 256 512 1024 2048)
IMPLEMENTATIONS=(eager graph)

for impl in "${IMPLEMENTATIONS[@]}"; do
  for b in "${BATCH_SIZES[@]}"; do
    for s in "${SEQ_LENS[@]}"; do
      for l in "${LAYERS[@]}"; do
        for d in "${DIMS[@]}"; do
          script="../${impl}_generation.py"
          "${TORCHRUN[@]}" "$script" \
            --tokenizer_path "$TOKENIZER" \
            --prompt_file= "$PROMPT_FILE" \
            --batch_size "$b" \
            --max_seq_len "$s" \
            --n_layers "$l" \
            --dim "$d" \
            --n_heads 4 \
            --results_csv "$CSV"
        done
      done
    done
  done
done

