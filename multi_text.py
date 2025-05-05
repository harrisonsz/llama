#!/usr/bin/env python
"""Llama‑2 PyTorch profiler (eager vs. torch.compile).

Improvements over the original script
-------------------------------------
* **Fair warm‑up** – both modes run one untimed text‑completion so that all
  Triton/Flash‑Attention kernels are compiled before we time inference.
* **Accurate GPU timing** – use `torch.cuda.Event` to measure execution on the
  device with micro‑second precision instead of coarse `time.time()`.
* **Peak‑memory reset** – CUDA peak counter is cleared *after* warm‑up so the
  recorded number reflects only the measured inference pass.
* **Clean CSV rows** – always write a row, even on failures, with the status
  field explaining what happened.
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

# ────────────────────────── util helpers ────────────────────────────────────


def _parse_int_list(arg: str | Sequence[int]) -> List[int]:
    """Accepts  '64,128'  |  '[64, 128]'  |  [64, 128]."""
    if isinstance(arg, (list, tuple)):
        return list(map(int, arg))
    if isinstance(arg, str):
        s = arg.strip()
        if s.startswith("[") and s.endswith("]"):
            return list(map(int, eval(s)))  # noqa: S307 (controlled input)
        return [int(x) for x in s.split(",") if x]
    raise ValueError(f"Cannot parse integer list from {arg!r}")


def _chunkify(seq: List[str], size: int) -> Iterable[List[str]]:
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

# ────────────────────────── timing helpers ──────────────────────────────────


def _warmup(gen: Llama, prompts: List[str], *, batch_size: int, max_gen_len: int, temperature: float, top_p: float) -> None:
    """Run an untimed pass to trigger kernel compilation & autotuning."""
    _ = gen.text_completion(  # noqa: F841 – side‑effects only
        prompts[:batch_size],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _measure(gen: Llama, batches: Iterable[List[str]], *, max_gen_len: int, temperature: float, top_p: float) -> tuple[float, float]:
    """Return (exec_seconds, peak_MB) using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for batch in batches:
        _ = gen.text_completion(batch, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    end.record()
    end.synchronize()

    exec_ms = start.elapsed_time(end)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return exec_ms / 1000.0, peak_mb

# ──────────────────────────── main entry ‑ fire CLI ─────────────────────────


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    *,
    dims: str = "128,256,512,1024,2048",
    layers: str = "1,2,4,6,8",
    n_heads: int = 4,
    prompt_file: str = "./prompt.txt",
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 60,
    max_gen_len: int = 60,
    max_batch_size: int = 64,
    results_csv: str = "eager_and_graph.csv",
) -> None:
    csv_path = Path(results_csv)
    _ensure_csv(csv_path)

    dims_l = _parse_int_list(dims)
    layers_l = _parse_int_list(layers)
    combos = list(product(dims_l, layers_l))
    modes = ("eager", "graph")

    # prompts
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
            status = "success"
            compile_s = 0.0
            exec_s = 0.0
            peak_mb = 0.0

            # ---------------- build model -----------------------------
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
                status = f"build‑error:{e}"
                _append_row(csv_path, [mode, dim, n_layers, n_heads, 0, 0, 0, 0, status])
                print(" build failed:", e)
                continue

            # --------------- optional torch.compile -------------------
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
                    compile_s = time.time() - t0
                    print(f" compiled graph in {compile_s:.2f}s")
                except Exception as e:
                    status = f"compile‑error:{e}"
                    _append_row(csv_path, [mode, dim, n_layers, n_heads, 0, 0, 0, 0, status])
                    print(" compile failed:", e)
                    continue

            # --------------- fair warm‑up for both modes --------------
            try:
                _warmup(gen, prompts, batch_size=max_batch_size, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
            except RuntimeError as e:
                status = f"warmup‑OOM:{e}"
            except Exception as e:
                status = f"warmup‑error:{e}"

            if status != "success":
                _append_row(csv_path, [mode, dim, n_layers, n_heads, f"{compile_s:.4f}", 0, f"{compile_s:.4f}", 0, status])
                print(" warm‑up failed:", status)
                continue

            # ---------------- measured execution ----------------------
            try:
                exec_s, peak_mb = _measure(
                    gen,
                    _chunkify(prompts, max_batch_size),
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
            except RuntimeError as e:
                status = f"OOM:{e}"
            except Exception as e:
                status = f"error:{e}"

            total_s = compile_s + exec_s
            _append_row(
                csv_path,
                [
                    mode,
                    dim,
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
                f" logged: compile {compile_s:.2f}s | exec {exec_s:.2f}s |"
                f" peak {peak_mb:.0f} MB | {status}"
            )


if __name__ == "__main__":
    fire.Fire(main)
