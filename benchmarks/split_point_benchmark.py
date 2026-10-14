import csv
import os
import statistics
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

RESULT_FILE = r"D:\Dynamic-LLM-Partitioning\benchmarks\split_point_results.csv"

PROMPT = "Explain machine learning in one simple sentence."

SPLIT_POINTS = [8, 10, 12, 14, 16, 18, 20]

WARMUP_RUNS = 1
MEASURED_RUNS = 3

COMMUNICATION_DELAY_SECONDS = 0.010


print("=" * 70)
print("Llama 3.2 3B - Static Split-Point Benchmark")
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
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

layers = model.model.layers

print("Model loaded successfully.")
print(f"Total transformer layers: {len(layers)}")

inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]


def run_partition(split_layer):

    with torch.no_grad():

        hidden_states = model.model.embed_tokens(input_ids)

        position_ids = torch.arange(
            input_ids.shape[1],
            device=input_ids.device
        ).unsqueeze(0)

        position_embeddings = model.model.rotary_emb(
            hidden_states,
            position_ids
        )

        # Worker 1
        worker_1_start = time.perf_counter()

        for layer in layers[:split_layer]:

            layer_output = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False
            )

            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        worker_1_time = time.perf_counter() - worker_1_start

        # Simulated communication
        communication_start = time.perf_counter()

        time.sleep(COMMUNICATION_DELAY_SECONDS)

        communication_time = (
            time.perf_counter() - communication_start
        )

        # Worker 2
        worker_2_start = time.perf_counter()

        for layer in layers[split_layer:]:

            layer_output = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False
            )

            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        worker_2_time = time.perf_counter() - worker_2_start

        # Final computation
        final_start = time.perf_counter()

        hidden_states = model.model.norm(hidden_states)

        logits = model.lm_head(hidden_states)

        final_time = time.perf_counter() - final_start

    total_time = (
        worker_1_time
        + communication_time
        + worker_2_time
        + final_time
    )

    return (
        worker_1_time,
        communication_time,
        worker_2_time,
        final_time,
        total_time
    )


all_results = []

print("\n" + "=" * 70)
print("TESTING SPLIT POINTS")
print("=" * 70)


for split_layer in SPLIT_POINTS:

    worker_1_count = split_layer
    worker_2_count = len(layers) - split_layer

    print("\n" + "-" * 70)
    print(
        f"Testing split: "
        f"{worker_1_count} / {worker_2_count}"
    )
    print("-" * 70)

    for i in range(WARMUP_RUNS):

        run_partition(split_layer)

        print(
            f"Warm-up {i + 1}/{WARMUP_RUNS} completed."
        )

    measurements = []

    for run_number in range(MEASURED_RUNS):

        (
            worker_1_time,
            communication_time,
            worker_2_time,
            final_time,
            total_time
        ) = run_partition(split_layer)

        measurements.append({
            "worker_1_time": worker_1_time,
            "communication_time": communication_time,
            "worker_2_time": worker_2_time,
            "final_time": final_time,
            "total_time": total_time
        })

        print(
            f"Run {run_number + 1}/{MEASURED_RUNS} | "
            f"W1: {worker_1_time:.4f}s | "
            f"Comm: {communication_time * 1000:.2f}ms | "
            f"W2: {worker_2_time:.4f}s | "
            f"Total: {total_time:.4f}s"
        )

    mean_w1 = statistics.mean(
        r["worker_1_time"] for r in measurements
    )

    mean_comm = statistics.mean(
        r["communication_time"] for r in measurements
    )

    mean_w2 = statistics.mean(
        r["worker_2_time"] for r in measurements
    )

    mean_final = statistics.mean(
        r["final_time"] for r in measurements
    )

    mean_total = statistics.mean(
        r["total_time"] for r in measurements
    )

    throughput = 10 / mean_total

    result = {
        "split_layer": split_layer,
        "worker_1_layers": worker_1_count,
        "worker_2_layers": worker_2_count,
        "worker_1_time": mean_w1,
        "communication_time": mean_comm,
        "worker_2_time": mean_w2,
        "final_time": mean_final,
        "total_time": mean_total,
        "throughput": throughput
    }

    all_results.append(result)

    print(
        f"\nMean total time: {mean_total:.4f}s"
    )

    print(
        f"Approx. throughput: "
        f"{throughput:.4f} tokens/sec"
    )


# ----------------------------------------------------------
# FIND BEST STATIC SPLIT
# ----------------------------------------------------------

best_result = min(
    all_results,
    key=lambda x: x["total_time"]
)

print("\n" + "=" * 70)
print("SPLIT-POINT SUMMARY")
print("=" * 70)

print(
    f"Best static split: "
    f"{best_result['worker_1_layers']} / "
    f"{best_result['worker_2_layers']}"
)

print(
    f"Best mean total time: "
    f"{best_result['total_time']:.4f} seconds"
)

print(
    f"Best throughput: "
    f"{best_result['throughput']:.4f} tokens/sec"
)

print("\nAll tested splits:")

for result in all_results:

    print(
        f"{result['worker_1_layers']:>2} / "
        f"{result['worker_2_layers']:<2} | "
        f"{result['total_time']:.4f}s | "
        f"{result['throughput']:.4f} tok/s"
    )

print("=" * 70)


# ----------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------

timestamp = datetime.now().isoformat(
    timespec="seconds"
)

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
            "timestamp",
            "model",
            "split_layer",
            "worker_1_layers",
            "worker_2_layers",
            "communication_delay_ms",
            "worker_1_time_seconds",
            "communication_time_seconds",
            "worker_2_time_seconds",
            "final_time_seconds",
            "total_time_seconds",
            "throughput_tokens_per_second"
        ])

    for result in all_results:

        writer.writerow([
            timestamp,
            "Llama 3.2 3B",
            result["split_layer"],
            result["worker_1_layers"],
            result["worker_2_layers"],
            COMMUNICATION_DELAY_SECONDS * 1000,
            round(result["worker_1_time"], 6),
            round(result["communication_time"], 6),
            round(result["worker_2_time"], 6),
            round(result["final_time"], 6),
            round(result["total_time"], 6),
            round(result["throughput"], 6)
        ])


print("\nResults saved to:")
print(RESULT_FILE)

print("\nSplit-point benchmark completed successfully!")
print("=" * 70)