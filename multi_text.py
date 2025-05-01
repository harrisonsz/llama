#!/usr/bin/env python
"""
multi_text.py  –  Llama-2 PyTorch profiler

* Sweeps over lists of `dims` and `n_layers` (defaults reproduce the OP grid).
* For each combination runs:
      • eager  – plain PyTorch
      • graph  – torch.compile(fullgraph=True, dynamic=True)
* Logs timing & peak-GPU-memory rows to `results_csv`.

CSV columns:
impl,dim,n_layers,n_heads,compile_s,exec_s,total_s,peak_gpu_MB,status
"""

from __future__ import annotations

import csv
import os
import time
from itertools import product, islice
from pathlib import Path
from typing import Iterable, List, Sequence

import fire
import torch
import torch._dynamo as dynamo
from llama import Llama

# ────────────────────────── helpers ──────────────────────────────────────────


def _parse_int_list(arg: str | Sequence[int]) -> List[int]:
    """Accepts  '64,128'  |  '[64, 128]'  |  [64, 128]."""
    if isinstance(arg, (list, tuple)):
        return list(map(int, arg))
    if isinstance(arg, str):
        s = arg.strip()
        if s.startswith("[") and s.endswith("]"):
            return list(map(int, eval(s)))
        return [int(x) for x in s.split(",") if x]
    raise ValueError(f"Cannot parse integer list from {arg!r}")


def chunkify(seq: List[str], size: int) -> Iterable[List[str]]:
    it = iter(seq)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def _ensure_csv(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "impl",
                    "dim",
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


# ──────────────────────────── main ───────────────────────────────────────────


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    *,
    dims: str = "128,256,512,1024,2048",
    layers: str = "1,2,4,6,8",
    n_heads: int = 4,                 # fixed
    prompt_file: str = "./prompt.txt",
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 60,
    max_gen_len: int = 60,
    max_batch_size: int = 10,
    results_csv: str = "pytorch_graph_metric.csv",
) -> None:
    csv_path = Path(results_csv)
    _ensure_csv(csv_path)

    dims_l = _parse_int_list(dims)
    layers_l = _parse_int_list(layers)
    combos = list(product(dims_l, layers_l))
    modes = ("eager", "graph")

    # load prompts once
    if prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        prompts = ["I believe the meaning of life is"]

    total_runs = len(combos) * len(modes)
    run_idx = 0

    for dim, n_layers in combos:
        params = {
            "dim": dim,
            "multiple_of": 256,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "norm_eps": 1e-5,
            "vocab_size": -1,
        }

        for mode in modes:
            run_idx += 1
            tag = f"[{run_idx}/{total_runs}] dim={dim} L={n_layers} mode={mode}"
            print("\n" + tag)

            _reset_cuda()

            # ---------------------- build model -----------------------------
            try:
                t0 = time.time()
                gen = Llama.build(
                    ckpt_dir=ckpt_dir,
                    tokenizer_path=tokenizer_path,
                    params=params,
                    max_seq_len=max_seq_len,
                    max_batch_size=max_batch_size,
                )
                torch.cuda.synchronize()
                build_s = time.time() - t0
                print(f" model built in {build_s:.2f}s")
            except Exception as e:
                _append_row(
                    csv_path,
                    [mode, dim, n_layers, n_heads, 0, 0, 0, 0, f"build-error:{e}"],
                )
                print(" build failed:", e)
                continue

            # ---------------------- optional compile ------------------------
            compile_s = 0.0
            if mode == "graph":
                try:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    gen.model = torch.compile(
                        gen.model,
                        mode="reduce-overhead",
                        fullgraph=True,
                        dynamic=True,
                    )
                    _ = gen.text_completion(["warm-up"], max_gen_len=1, temperature=0, top_p=0)
                    torch.cuda.synchronize()
                    compile_s = time.time() - t0
                    print(f" compiled in {compile_s:.2f}s")
                except Exception as e:
                    _append_row(
                        csv_path,
                        [mode, dim, n_layers, n_heads, 0, 0, 0, 0, f"compile-error:{e}"],
                    )
                    print(" compile failed:", e)
                    continue

            # ---------------------- execute ---------------------------------
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            status = "success"
            try:
                for batch in chunkify(prompts, max_batch_size):
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

            _append_row(
                csv_path,
                [
                    mode,
                    dim,
                    n_layers,
                    n_heads,
                    f"{compile_s:.4f}",
                    f"{exec_s:.4f}",
                    f"{compile_s + exec_s:.4f}",
                    f"{peak_mb:.1f}",
                    status,
                ],
            )
            print(
                f" logged: compile {compile_s:.2f}s | exec {exec_s:.2f}s |"
                f" peak {peak_mb:.0f} MB | {status}"
            )


if __name__ == "__main__":
    fire.Fire(main)
