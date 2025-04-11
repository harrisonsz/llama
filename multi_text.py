import time
import fire
from typing import List, Optional

from llama import Llama


def chunkify(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i + chunk_size]


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        prompt_file: Optional[str] = "./prompt.txt",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 128,
        max_gen_len: int = 64,
        max_batch_size: int = 4,
):
    """
    Simple text completion script that can optionally read a list of prompts from a file.

    Args:
        ckpt_dir (str): Path to your checkpoint directory.
        tokenizer_path (str): Path to the tokenizer model file.
        prompt_file (str, optional): If provided, read prompts from this file (one prompt per line).
        temperature (float, optional): Sampling temperature. Default 0.6.
        top_p (float, optional): Top-p sampling. Default 0.9.
        max_seq_len (int, optional): Max input sequence length. Default 128.
        max_gen_len (int, optional): Max tokens to generate. Default 64.
        max_batch_size (int, optional): Max batch size for generation. Default 4.
    """
    # Build the model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # If a file of prompts is specified, read them; otherwise use a small hardcoded list
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            # read lines, strip whitespace, skip empty lines
            prompts: List[str] = [line.strip() for line in f if line.strip()]
    else:
        prompts: List[str] = [
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

    # Time the entire generation
    print("Starting generation...")
    start_time = time.time()

    all_results = []
    # We can process prompts in batches
    for batch_prompts in chunkify(prompts, max_batch_size):
        batch_results = generator.text_completion(
            batch_prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        all_results.extend(batch_results)

    end_time = time.time()

    # Print the outputs
    for prompt, result in zip(prompts, all_results):
        print("==================================")
        print(f"Prompt:\n{prompt}")
        print("\nCompletion:")
        print(result["generation"])
        print("==================================\n")

    print(f"Finished generation in {end_time - start_time:.2f} seconds.\n")


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=1 multi_text.py   --ckpt_dir="."   --tokenizer_path="/home/sz/.llama/checkpoints/Llama-2-7b/tokenizer.model"   --max_seq_len=256   --max_batch_size=8 --max_gen_len=256
    fire.Fire(main)
