#!/usr/bin/env python
"""
run_once_graph.py – Llama-2 PyTorch profiler (graph mode via torch.compile)

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
import threading
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
                    "gpu_util",
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


class GPUUtilizationMonitor:
    """Background GPU utilization sampler using pynvml if available."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._running = False
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None

    def _get_utilization(self) -> float:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util.gpu)
            pynvml.nvmlShutdown()
            return gpu_util
        except Exception:
            return 0.0

    def _run(self) -> None:
        while self._running:
            self._samples.append(self._get_utilization())
            time.sleep(self.interval)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def average(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0


# ──────────────────────────── main ───────────────────────────────────────────
def main(
    tokenizer_path: str,
    *,
    dim: int = 512,
    n_layers: int = 6,
    n_heads: int = 4,
    prompt_file: str | None = "./prompt.txt",
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 60,
    batch_size: int = 32,
    results_csv: str = "eager_graph_data.csv",
) -> None:
    """Run a single configuration in graph mode with torch.compile."""
    csv_path = Path(results_csv)
    _ensure_csv(csv_path)

    # Load prompts
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()][:batch_size]

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
    monitor = GPUUtilizationMonitor(interval=0.001)
    monitor.start()
    torch.cuda.synchronize()
    t_compile_start = time.time()
    gen = Llama.build(
        ckpt_dir="",
        tokenizer_path=tokenizer_path,
        params=params,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    gen.model = torch.compile(gen.model, mode="default")
    _ = gen.text_completion(["warm-up"], max_gen_len=1, temperature=0, top_p=0)
    torch.cuda.synchronize()
    compile_s = time.time() - t_compile_start

    # ---------------------- execute ---------------------------------
    status = "success"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        for batch in chunkify(prompts, batch_size):
            _ = gen.text_completion(
                batch,
                max_gen_len=max_seq_len,
                temperature=temperature,
                top_p=top_p,
            )
    except RuntimeError as e:
        status = f"OOM:{e}"
    except Exception as e:
        status = f"error:{e}"
    torch.cuda.synchronize()
    exec_s = time.time() - t0
    monitor.stop()

    busy_samples = [u for u in monitor._samples if u > 0]
    avg_util = sum(busy_samples) / len(busy_samples)
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )

    # ───────── after generation finishes ─────────
    total_s = compile_s + exec_s

    _append_row(
        csv_path,
        [
            "graph",              # impl
            dim,
            batch_size,
            max_seq_len,          # seq_len (input context length requested)
            n_layers,
            n_heads,
            f"{compile_s:.4f}",
            f"{exec_s:.4f}",
            f"{total_s:.4f}",
            f"{peak_mb:.1f}",
            f"{avg_util:.1f}",
            status,
        ],
    )
    print(
        f" logged: compile {compile_s:.2f}s | exec {exec_s:.2f}s "
        f"| peak {peak_mb:.0f} MB | util {avg_util:.1f}% | {status}"
    )


if __name__ == "__main__":
    fire.Fire(main)
