#!/usr/bin/env python
"""
run_once.py – Llama-2 PyTorch profiler (eager mode only)

• Builds the model once (regular eager execution).
• Executes the prompt batch once.
• Logs timing & peak-GPU-memory to a CSV.

CSV columns:
dim,n_layers,n_heads,exec_s,peak_gpu_MB,status
"""

from __future__ import annotations

import csv
import os
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, List

import fire
import torch
import torch._dynamo as dynamo
from llama import Llama


# ────────────────────────── helpers ──────────────────────────────────────────
def chunkify(seq: List[str], size: int) -> Iterable[List[str]]:
    it = iter(seq)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def _ensure_csv(path: Path) -> None:
    """Create the CSV file (with header) if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "impl",          # "eager"
                    "dim",
                    "batch_size",
                    "seq_len",
                    "n_layers",
                    "n_heads",
                    "compile_s",     # always 0 for eager
                    "exec_s",
                    "total_s",
                    "peak_gpu_MB",
                    "status",
                ]
            )


def _append_row(path: Path, row: list) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def _reset_cuda() -> None:
    dynamo.reset()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ──────────────────────────── main ───────────────────────────────────────────
def main(
    tokenizer_path: str,
    *,
    dim: int = 512,
    n_layers: int = 6,
    n_heads: int = 4,
    prompt_file: str | None = "./prompt.txt",
    prompt: str | None = None,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 60,
    max_gen_len: int = 60,
    batch_size: int = 32,
    results_csv: str = "eager_graph_data.csv",
) -> None:
    """Run a single configuration in eager mode."""
    # global results
    csv_path = Path(results_csv)
    _ensure_csv(csv_path)

    # Load prompts
    if prompt is not None:
        prompts = [prompt]
    elif prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()][:batch_size]
    else:
        prompts = ["I believe the meaning of life is"]

    params = {
        "dim": dim,
        "multiple_of": 256,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "norm_eps": 1e-5,
        "vocab_size": -1,
    }

    print(f"\ndim={dim} | layers={n_layers} | eager mode\n")

    _reset_cuda()

    # ---------------------- build model -----------------------------
    gen = Llama.build(
        ckpt_dir="",
        tokenizer_path=tokenizer_path,
        params=params,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    torch.cuda.synchronize()

    # ---------------------- execute ---------------------------------
    status = "success"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        for batch in chunkify(prompts, batch_size):
            results = gen.text_completion(
                batch,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
    except RuntimeError as e:
        status = f"OOM:{e}"
    except Exception as e:
        status = f"error:{e}"
    torch.cuda.synchronize()
    exec_s = time.time() - t0
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )
    # for prompt, result in zip(prompts, results):
    #     print(prompt)
    #     print(f"> {result['generation']}")
    #     print("\n==================================\n")

    # ───────── after generation finishes ─────────
    compile_s = 0.0
    total_s = exec_s  # no compile step in eager mode

    _append_row(
        csv_path,
        [
            "eager",
            dim,
            batch_size,
            max_seq_len,
            n_layers,
            n_heads,
            f"{compile_s:.4f}",
            f"{exec_s:.4f}",
            f"{total_s:.4f}",
            f"{peak_mb:.1f}",
            status,
        ],
    )
    print(
        f" logged: exec {exec_s:.2f}s | peak {peak_mb:.0f} MB | {status}"
    )


if __name__ == "__main__":
    fire.Fire(main)