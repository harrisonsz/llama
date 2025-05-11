#!/usr/bin/env bash
# run_generation.sh  –  profile eager- and graph-based generation variants
set -euo pipefail

###############################################################################
# Globals                                                                     #
###############################################################################
CSV_EAGER="profile_eager_generation.csv"
CSV_GRAPH="profile_graph_generation.csv"
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

###############################################################################
# Sweep 1: vary dims & layers (seq=60, batch=16)                              #
###############################################################################
dims=(64 128 256 512 1024)
layers=(2 4 6 8 10)
for d in "${dims[@]}"; do
  for l in "${layers[@]}"; do
    run_pair "$d" "$l" 60 16
  done
done

###############################################################################
# Sweep 2: vary max-seq-len (dim=512, layers=6, batch=16)                     #
###############################################################################
seq_lens=(60 70 80 90 100)
for s in "${seq_lens[@]}"; do
  run_pair 512 6 "$s" 16
done

###############################################################################
# Sweep 3: vary batch-size (dim=512, layers=6, seq=60)                        #
###############################################################################
batch_sizes=(4 8 16 32 64)
for b in "${batch_sizes[@]}"; do
  run_pair 512 6 60 "$b"
done

echo "✓ All runs finished – results in $CSV_EAGER and $CSV_GRAPH"
