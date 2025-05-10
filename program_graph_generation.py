#!/usr/bin/env python
"""
program_graph_generation.py – Llama-2 PyTorch profiler (graph mode via torch.compile)

• Builds the model once (graph mode).
• Executes the prompt batch once.
• Logs compile time, execution time & peak-GPU-memory to a CSV.

CSV columns:
impl,dim,batch_size,seq_len,n_layers,n_heads,compile_s,exec_s,total_s,peak_gpu_MB,status
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
import torch.distributed as dist
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
                    "impl",          # "graph"
                    "dim",
                    "batch_size",
                    "seq_len",
                    "n_layers",
                    "n_heads",
                    "compile_s",
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


# Make CUDA float the global default if desired
torch.set_default_tensor_type(torch.cuda.FloatTensor)

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
    results_csv: str = "single_run.csv",
    # Extra args for measuring compile vs. exec
    compile_s: float = 0.0,
    skip_gen: bool = False,
) -> None:
    """
    Run a single configuration in graph mode. The function no longer decorates with @torch.compile
    so that we can measure compile_s separately outside this function.
    If skip_gen=True, we skip the actual generation (dummy call) just to trigger compilation.
    """
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

    print(f"\ndim={dim} | layers={n_layers} | graph mode (torch.compile)\n")

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

    # If we're just here to force a dummy compile, skip the real generation
    if skip_gen:
        # Return early, no CSV row needed
        return

    # ---------------------- execute ---------------------------------
    status = "success"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        for batch in chunkify(prompts, batch_size):
            _ = gen.text_completion(
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

    total_s = compile_s + exec_s

    _append_row(
        csv_path,
        [
            "graph-program",
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
        f" logged: compile {compile_s:.2f}s | exec {exec_s:.2f}s "
        f"| peak {peak_mb:.0f} MB | {status}"
    )


if __name__ == "__main__":

    def _run(**kwargs):
        """
        1) Force the compile (dummy call) & measure compile time
        2) Actual run & measure exec time
        """
        _reset_cuda()

        # Measure how long it takes to compile the main function
        t_compile_start = time.time()
        compiled_main = torch.compile(main, mode="default")
        # Trigger a dummy run so that actual compilation happens
        compiled_main(**kwargs, skip_gen=True)
        torch.cuda.synchronize()
        compile_s = time.time() - t_compile_start

        # Now measure the real execution time
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        compiled_main(**kwargs, compile_s=compile_s, skip_gen=False)
        torch.cuda.synchronize()
        exec_s = time.time() - t0

        # We don't append CSV rows here; that happens inside main(...)
        # so it already has compile_s & exec_s.

    # Use Fire to parse command line, then call _run(...)
    fire.Fire(_run)

    # If you had an init_process_group call, you'd also destroy it here:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
