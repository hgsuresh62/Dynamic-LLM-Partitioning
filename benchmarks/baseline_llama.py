import csv
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"
RESULT_FILE = r"D:\Dynamic-LLM-Partitioning\benchmarks\baseline_results.csv"

PROMPT = "Explain machine learning in one simple sentence."
MAX_NEW_TOKENS = 10


print("=" * 60)
print("Llama 3.2 3B - Baseline Benchmark")
print("=" * 60)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded successfully.")

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")

inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

print("\nRunning benchmark...")

start_time = time.perf_counter()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

end_time = time.perf_counter()

generation_time = end_time - start_time

input_tokens = inputs["input_ids"].shape[1]
total_tokens = output.shape[1]
new_tokens = total_tokens - input_tokens

throughput = new_tokens / generation_time

print("\n" + "=" * 60)
print("BASELINE RESULT")
print("=" * 60)
print(f"Model: Llama 3.2 3B")
print(f"Execution device: CPU")
print(f"New tokens: {new_tokens}")
print(f"Generation time: {generation_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} tokens/sec")
print("=" * 60)


file_exists = os.path.exists(RESULT_FILE)

with open(
    RESULT_FILE,
    mode="a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "model",
            "method",
            "device",
            "new_tokens",
            "generation_time_seconds",
            "throughput_tokens_per_second"
        ])

    writer.writerow([
        "Llama 3.2 3B",
        "Baseline",
        "CPU",
        new_tokens,
        round(generation_time, 4),
        round(throughput, 4)
    ])


print("\nResult saved to:")
print(RESULT_FILE)

print("\nBaseline benchmark completed successfully!")