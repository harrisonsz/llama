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
        yield lst[i: i + chunk_size]


def _log_metrics(row: list, csv_path: str):
    """Append a single row of metrics to the CSV file, creating the header if needed."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "impl",
                "dim",
                "n_layers",
                "n_heads",
                "compile_s",
                "exec_s",
                "total_s",
                "peak_gpu_MB",
                "status",
            ])
        writer.writerow(row)


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        prompt_file: Optional[str] = "./prompt.txt",
        temperature: float = 0,
        top_p: float = 0.9,
        max_seq_len: int = 128,
        max_gen_len: int = 64,
        max_batch_size: int = 4,
        graph_mode: Optional[bool] = False,
        results_csv: str = "pytorch_graph_metric.csv",
):
    """
    Simple text completion script that can optionally read a list of prompts from a file.

    Now also records timing & memory metrics to a local CSV file with the following columns::
        impl,dim,n_layers,n_heads,compile_s,exec_s,total_s,peak_gpu_MB,status
    """
    params = {
        "dim": 1024,
        "multiple_of": 256,
        "n_heads": 4,
        "n_layers": 8,
        "norm_eps": 1e-05,
        "vocab_size": -1,
    }

    impl = "graph" if graph_mode else "eager"
    t_compile = 0.0  # default if we don't compile

    # Build the model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        params=params,
    )

    # === Prepare prompts ===
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts: List[str] = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "I believe the meaning of life is",
            "Simply put, the theory of relativity states that ",
            """A brief message congratulating the team on the launch:

            Hi everyone,

            I just """,
            """Translate English to French:

            sea otter => loutre de mer
            peppermint => menthe poivrée
            plush girafe => girafe peluche
            cheese =>""",
        ]

    # === Measure compilation (graph) ===
    if graph_mode:
        generator.model = torch.compile(
            generator.model,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=True,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_compile_start = time.time()
        _ = generator.text_completion(["warm-up"], max_gen_len=1, temperature=0.0, top_p=0.0)
        torch.cuda.synchronize()
        t_compile = time.time() - t_compile_start
        print(f"Compilation time: {t_compile:.2f} seconds.")

    # === Measure execution ===
    print("Starting generation...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_exec_start = time.time()

    status = "success"
    try:
        all_results = []
        for batch_prompts in chunkify(prompts, max_batch_size):
            batch_results = generator.text_completion(
                batch_prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            all_results.extend(batch_results)
    except Exception as e:
        status = f"error: {e}"
        raise  # Re‑raise so the caller still sees the exception
    finally:
        torch.cuda.synchronize()
        t_exec = time.time() - t_exec_start
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        total_s = t_compile + t_exec
        _log_metrics([
            impl,
            params["dim"],
            params["n_layers"],
            params["n_heads"],
            f"{t_compile:.4f}",
            f"{t_exec:.4f}",
            f"{total_s:.4f}",
            f"{peak_gpu_mb:.2f}",
            status,
        ], results_csv)

    # Print the outputs if successful
    if status == "success":
        for prompt, result in zip(prompts, all_results):
            print("==================================")
            print(f"Prompt:\n{prompt}")
            print("\nCompletion:")
            print(result["generation"])
            print("==================================\n")


if __name__ == "__main__":
    # Example invocation:
    # torchrun --standalone --nproc_per_node=1 multi_text_with_logging.py   --ckpt_dir="."   --tokenizer_path="/home/sz/.llama/checkpoints/Llama-2-7b/tokenizer.model"   --max_seq_len=256   --max_batch_size=8 --max_gen_len=256 --graph_mode True
    fire.Fire(main)
