#!/usr/bin/env bash
# run_generation.sh  –  profile eager- and graph-based generation variants
set -euo pipefail

###############################################################################
# Globals                                                                     #
###############################################################################
CSV_EAGER="profile_eager_generation_with_varied_seq_len.csv"
CSV_GRAPH="profile_graph_generation_with_varied_seq_len.csv"
TOKENIZER="/home/sz/.llama/checkpoints/Llama-2-7b/tokenizer.model"

N_HEADS=4
TORCHRUN=(torchrun --standalone --nproc_per_node=1)

COMMON_FLAGS=(--tokenizer_path "$TOKENIZER" --n_heads "$N_HEADS")

###############################################################################
# Helpers                                                                     #
###############################################################################
run_one () {      # $1=script  $2..$6 = dim n_layers seq_len batch csv
  local script=$1 d=$2 l=$3 s=$4 b=$5 csv=$6
  echo "▶ $(basename "$script")  dim=$d  L=$l  S=$s  B=$b"
  "${TORCHRUN[@]}" "$script" \
        --dim "$d" \
        --n_layers "$l" \
        --max_seq_len "$s" \
        --batch_size "$b" \
        --results_csv "$csv" \
        "${COMMON_FLAGS[@]}"
}

run_pair () {     # $1..$4 = dim n_layers seq_len batch
  run_one eager_generation.py "$1" "$2" "$3" "$4" "$CSV_EAGER"
  run_one graph_generation.py "$1" "$2" "$3" "$4" "$CSV_GRAPH"
}

seq_lens=(50 100 200 400 800 1600)
for s in "${seq_lens[@]}"; do
  run_pair 1024 12 "$s" 64
done