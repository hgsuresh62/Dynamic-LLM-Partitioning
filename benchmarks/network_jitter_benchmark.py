import time
import statistics
import csv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

# Best static split found in the previous experiment
SPLIT_POINT = 12

# Simulated network delays
JITTER_VALUES_MS = [2, 5, 10, 20, 50, 100]

WARMUP_RUNS = 1
MEASURED_RUNS = 3

PROMPT = "Explain machine learning in one simple sentence."

NEW_TOKENS = 10


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Llama 3.2 3B - Network Jitter Benchmark")
print("=" * 70)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded.")

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")

layers = model.model.layers
num_layers = len(layers)

print(f"Total transformer layers: {num_layers}")
print(f"Static split: {SPLIT_POINT} / {num_layers - SPLIT_POINT}")


# ============================================================
# PREPARE INPUT
# ============================================================

inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]

attention_mask = inputs.get("attention_mask")

if attention_mask is not None:
    attention_mask = attention_mask.to(torch.bool)

position_ids = torch.arange(
    input_ids.shape[1],
    dtype=torch.long
).unsqueeze(0)


# ============================================================
# PARTITIONED FORWARD
# ============================================================

def run_partitioned(simulated_delay_ms):

    start_total = time.perf_counter()

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    hidden_states = model.model.embed_tokens(input_ids)

    # --------------------------------------------------------
    # Rotary positional embeddings
    # --------------------------------------------------------

    position_embeddings = model.model.rotary_emb(
        hidden_states,
        position_ids
    )

    # --------------------------------------------------------
    # Worker 1
    # Layers 0 -> 11
    # --------------------------------------------------------

    worker1_start = time.perf_counter()

    for layer in layers[:SPLIT_POINT]:

        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False
        )

    worker1_time = time.perf_counter() - worker1_start

    # --------------------------------------------------------
    # Simulated network communication
    # --------------------------------------------------------

    communication_start = time.perf_counter()

    time.sleep(simulated_delay_ms / 1000.0)

    communication_time = time.perf_counter() - communication_start

    # --------------------------------------------------------
    # Worker 2
    # Layers 12 -> 27
    # --------------------------------------------------------

    worker2_start = time.perf_counter()

    for layer in layers[SPLIT_POINT:]:

        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False
        )

    worker2_time = time.perf_counter() - worker2_start

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    final_start = time.perf_counter()

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)

    final_time = time.perf_counter() - final_start

    total_time = time.perf_counter() - start_total

    return (
        worker1_time,
        communication_time,
        worker2_time,
        final_time,
        total_time
    )


# ============================================================
# BENCHMARK
# ============================================================

results = []

print("\n" + "=" * 70)
print("TESTING NETWORK JITTER")
print("=" * 70)

for jitter_ms in JITTER_VALUES_MS:

    print("\n" + "-" * 70)
    print(f"Network delay: {jitter_ms} ms")
    print("-" * 70)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for warmup in range(WARMUP_RUNS):

        run_partitioned(jitter_ms)

        print(
            f"Warm-up {warmup + 1}/{WARMUP_RUNS} completed."
        )

    run_times = []

    # --------------------------------------------------------
    # Measured runs
    # --------------------------------------------------------

    for run in range(MEASURED_RUNS):

        (
            worker1_time,
            communication_time,
            worker2_time,
            final_time,
            total_time
        ) = run_partitioned(jitter_ms)

        run_times.append(total_time)

        throughput = NEW_TOKENS / total_time

        print(
            f"Run {run + 1}/{MEASURED_RUNS} | "
            f"W1: {worker1_time:.4f}s | "
            f"Comm: {communication_time * 1000:.2f}ms | "
            f"W2: {worker2_time:.4f}s | "
            f"Total: {total_time:.4f}s | "
            f"Throughput: {throughput:.4f} tok/s"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_time = statistics.mean(run_times)

    median_time = statistics.median(run_times)

    std_time = (
        statistics.stdev(run_times)
        if len(run_times) > 1
        else 0
    )

    mean_throughput = NEW_TOKENS / mean_time

    print(
        f"\nMean total time: {mean_time:.4f}s"
    )

    print(
        f"Median total time: {median_time:.4f}s"
    )

    print(
        f"Std deviation: {std_time:.4f}s"
    )

    print(
        f"Mean throughput: {mean_throughput:.4f} tokens/sec"
    )

    results.append({
        "network_delay_ms": jitter_ms,
        "mean_total_time_s": mean_time,
        "median_total_time_s": median_time,
        "std_total_time_s": std_time,
        "mean_throughput_tokens_sec": mean_throughput
    })


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NETWORK JITTER SUMMARY")
print("=" * 70)

for result in results:

    print(
        f"{result['network_delay_ms']:>3} ms | "
        f"{result['mean_total_time_s']:.4f}s | "
        f"{result['mean_throughput_tokens_sec']:.4f} tok/s"
    )


# ============================================================
# FIND BEST / WORST
# ============================================================

best = min(
    results,
    key=lambda x: x["mean_total_time_s"]
)

worst = max(
    results,
    key=lambda x: x["mean_total_time_s"]
)

print("\n" + "-" * 70)

print(
    f"Best network condition: "
    f"{best['network_delay_ms']} ms"
)

print(
    f"Best mean time: "
    f"{best['mean_total_time_s']:.4f}s"
)

print(
    f"Best throughput: "
    f"{best['mean_throughput_tokens_sec']:.4f} tokens/sec"
)

print()

print(
    f"Worst network condition: "
    f"{worst['network_delay_ms']} ms"
)

print(
    f"Worst mean time: "
    f"{worst['mean_total_time_s']:.4f}s"
)

print(
    f"Worst throughput: "
    f"{worst['mean_throughput_tokens_sec']:.4f} tokens/sec"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    r"D:\Dynamic-LLM-Partitioning\benchmarks"
    r"\network_jitter_results.csv"
)

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "network_delay_ms",
            "mean_total_time_s",
            "median_total_time_s",
            "std_total_time_s",
            "mean_throughput_tokens_sec"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print("\nResults saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("Network jitter benchmark completed successfully!")
print("=" * 70)