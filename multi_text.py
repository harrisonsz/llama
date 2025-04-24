import time
import os
import csv
from typing import List, Optional

import fire
import torch

from llama import Llama


def chunkify(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def _log_metrics(row: list, csv_path: str):
    """Append a single row of metrics to the CSV file, creating the header if needed."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
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
        writer.writerow(row)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    dim: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    prompt_file: Optional[str] = "./prompt.txt",
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 60,
    max_gen_len: int = 60,
    max_batch_size: int = 4,
    graph_mode: bool = False,
    results_csv: str = "pytorch_graph_metric.csv",
):
    """
    Runs text completion on a single (dim, n_layers, n_heads) configuration.
    Logs timing & memory metrics to a CSV file with columns:
        impl, dim, n_layers, n_heads, compile_s, exec_s, total_s, peak_gpu_MB, status

    Usage (example):
    ----------------
    python test_pytorch_single_config.py \\
        --ckpt_dir="." \\
        --tokenizer_path="/path/to/tokenizer.model" \\
        --dim=512 \\
        --n_layers=4 \\
        --n_heads=8 \\
        --graph_mode=True
    """

    # Set up model hyperparameters
    params = {
        "dim": dim,
        "multiple_of": 256,  # if your model requires multiples
        "n_heads": n_heads,
        "n_layers": n_layers,
        "norm_eps": 1e-05,
        "vocab_size": -1,    # typically overridden in the checkpoint
    }

    impl = "graph" if graph_mode else "eager"
    print(f"===== Running with config: {params} | mode={impl} =====")

    # Prepare prompts
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts: List[str] = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "I believe the meaning of life is",
            "Simply put, the theory of relativity states that ",
        ]

    # ============= Build Model =============
    try:
        model_build_start = time.time()
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            params=params,
        )
        torch.cuda.synchronize()
        build_time = time.time() - model_build_start
        print(f"Model built in {build_time:.2f} s.")
    except Exception as e:
        status = f"build-error: {e}"
        _log_metrics(
            [
                impl,
                dim,
                n_layers,
                n_heads,
                0.0,   # compile_s
                0.0,   # exec_s
                0.0,   # total_s
                0.0,   # peak_gpu_MB
                status,
            ],
            results_csv,
        )
        print(f"[ERROR] Failed to build model: {e}")
        return

    # ============= Graph Compilation (optional) =============
    t_compile = 0.0
    if graph_mode:
        print("Compiling with torch.compile() ...")
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t_compile_start = time.time()
            generator.model = torch.compile(
                generator.model,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=True,
            )
            # Force warm-up pass so we measure compilation overhead
            _ = generator.text_completion(["warm-up"], max_gen_len=1, temperature=0.0, top_p=0.0)
            torch.cuda.synchronize()
            t_compile = time.time() - t_compile_start
            print(f"Compilation time: {t_compile:.2f} s.")
        except Exception as e:
            status = f"compile-error: {e}"
            _log_metrics(
                [
                    impl,
                    dim,
                    n_layers,
                    n_heads,
                    0.0,   # compile_s
                    0.0,   # exec_s
                    0.0,   # total_s
                    0.0,   # peak_gpu_MB
                    status,
                ],
                results_csv,
            )
            print(f"[ERROR] Graph compilation failed: {e}")
            return

    # ============= Measure Execution =============
    print("Starting text generation...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_exec_start = time.time()

    status = "success"
    all_results = []
    try:
        # Generate results in small batches
        for batch_prompts in chunkify(prompts, max_batch_size):
            batch_results = generator.text_completion(
                batch_prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            all_results.extend(batch_results)
    except RuntimeError as rt_err:
        status = f"OOM: {rt_err}"
        print("[ERROR] Out of memory or runtime error.")
    except Exception as e:
        status = f"error: {e}"
        print(f"[ERROR] Generation failed: {e}")
    finally:
        torch.cuda.synchronize()
        t_exec = time.time() - t_exec_start
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        total_s = t_compile + t_exec

        # Log metrics
        _log_metrics(
            [
                impl,
                dim,
                n_layers,
                n_heads,
                f"{t_compile:.4f}",
                f"{t_exec:.4f}",
                f"{total_s:.4f}",
                f"{peak_gpu_mb:.2f}",
                status,
            ],
            results_csv,
        )

    # ============= Print Results (if successful) =============
    if status == "success":
        for prompt_text, out_dict in zip(prompts, all_results):
            print("==================================")
            print(f"Prompt:\n{prompt_text}")
            print("\nCompletion:")
            print(out_dict["generation"])
            print("==================================\n")


if __name__ == "__main__":
    # Example command:
    # python test_pytorch_single_config.py \
    #    --ckpt_dir="." \
    #    --tokenizer_path="/path/to/tokenizer.model" \
    #    --dim=512 --n_layers=4 --n_heads=8 \
    #    --graph_mode=True
    fire.Fire(main)
