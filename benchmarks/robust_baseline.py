import csv
import os
import statistics
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

RUNS_FILE = r"D:\Dynamic-LLM-Partitioning\benchmarks\baseline_runs.csv"
SUMMARY_FILE = r"D:\Dynamic-LLM-Partitioning\benchmarks\baseline_summary.csv"

PROMPT = "Explain machine learning in one simple sentence."

WARMUP_RUNS = 2
MEASURED_RUNS = 5
MAX_NEW_TOKENS = 10


print("=" * 70)
print("Llama 3.2 3B - Robust Baseline Benchmark")
print("=" * 70)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded successfully.")

print("\nLoading Llama 3.2 3B model...")

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

print("\nPrompt:")
print(PROMPT)

print("\n" + "=" * 70)
print("WARM-UP")
print("=" * 70)

for i in range(WARMUP_RUNS):

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    print(f"Warm-up {i + 1}/{WARMUP_RUNS} completed.")


print("\n" + "=" * 70)
print("MEASURED RUNS")
print("=" * 70)

generation_times = []
throughputs = []
token_counts = []

for i in range(MEASURED_RUNS):

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

    generation_times.append(generation_time)
    throughputs.append(throughput)
    token_counts.append(new_tokens)

    print(
        f"Run {i + 1}/{MEASURED_RUNS}: "
        f"{generation_time:.4f} sec | "
        f"{new_tokens} tokens | "
        f"{throughput:.4f} tokens/sec"
    )


mean_time = statistics.mean(generation_times)
median_time = statistics.median(generation_times)
std_time = statistics.stdev(generation_times)

mean_throughput = statistics.mean(throughputs)
median_throughput = statistics.median(throughputs)
std_throughput = statistics.stdev(throughputs)


print("\n" + "=" * 70)
print("BASELINE SUMMARY")
print("=" * 70)

print(f"Model: Llama 3.2 3B")
print("Device: CPU")
print(f"Warm-up runs: {WARMUP_RUNS}")
print(f"Measured runs: {MEASURED_RUNS}")
print(f"Tokens per run: {MAX_NEW_TOKENS}")

print("\nGeneration Time:")
print(f"Mean:   {mean_time:.4f} seconds")
print(f"Median: {median_time:.4f} seconds")
print(f"Std:    {std_time:.4f} seconds")

print("\nThroughput:")
print(f"Mean:   {mean_throughput:.4f} tokens/sec")
print(f"Median: {median_throughput:.4f} tokens/sec")
print(f"Std:    {std_throughput:.4f} tokens/sec")

print("=" * 70)


timestamp = datetime.now().isoformat(timespec="seconds")


runs_exists = os.path.exists(RUNS_FILE)

with open(
    RUNS_FILE,
    mode="a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if not runs_exists:
        writer.writerow([
            "timestamp",
            "model",
            "device",
            "run_number",
            "new_tokens",
            "generation_time_seconds",
            "throughput_tokens_per_second"
        ])

    for i in range(MEASURED_RUNS):
        writer.writerow([
            timestamp,
            "Llama 3.2 3B",
            "CPU",
            i + 1,
            token_counts[i],
            round(generation_times[i], 6),
            round(throughputs[i], 6)
        ])


summary_exists = os.path.exists(SUMMARY_FILE)

with open(
    SUMMARY_FILE,
    mode="a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if not summary_exists:
        writer.writerow([
            "timestamp",
            "model",
            "device",
            "warmup_runs",
            "measured_runs",
            "mean_time_seconds",
            "median_time_seconds",
            "std_time_seconds",
            "mean_throughput_tokens_per_second",
            "median_throughput_tokens_per_second",
            "std_throughput_tokens_per_second"
        ])

    writer.writerow([
        timestamp,
        "Llama 3.2 3B",
        "CPU",
        WARMUP_RUNS,
        MEASURED_RUNS,
        round(mean_time, 6),
        round(median_time, 6),
        round(std_time, 6),
        round(mean_throughput, 6),
        round(median_throughput, 6),
        round(std_throughput, 6)
    ])


print("\nResults saved to:")
print(RUNS_FILE)
print(SUMMARY_FILE)

print("\nRobust baseline benchmark completed successfully!")