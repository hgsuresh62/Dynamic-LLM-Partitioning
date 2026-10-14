import csv
import os
import statistics
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

RESULT_FILE = r"D:\Dynamic-LLM-Partitioning\benchmarks\static_partition_runs.csv"

SPLIT_LAYER = 14

PROMPT = "Explain machine learning in one simple sentence."

WARMUP_RUNS = 2
MEASURED_RUNS = 5

COMMUNICATION_DELAY_SECONDS = 0.010


print("=" * 70)
print("Llama 3.2 3B - Static Partition Benchmark")
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

print("Model loaded successfully.")

layers = model.model.layers

print(f"\nTotal layers: {len(layers)}")
print(f"Worker 1: layers 0-{SPLIT_LAYER - 1}")
print(f"Worker 2: layers {SPLIT_LAYER}-{len(layers) - 1}")

inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]


def run_partition():

    with torch.no_grad():

        # --------------------------------------------------
        # EMBEDDING
        # --------------------------------------------------

        hidden_states = model.model.embed_tokens(input_ids)

        position_ids = torch.arange(
            input_ids.shape[1],
            device=input_ids.device
        ).unsqueeze(0)

        position_embeddings = model.model.rotary_emb(
            hidden_states,
            position_ids
        )

        # --------------------------------------------------
        # WORKER 1
        # --------------------------------------------------

        worker_1_start = time.perf_counter()

        for layer in layers[:SPLIT_LAYER]:

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

        # --------------------------------------------------
        # COMMUNICATION
        # --------------------------------------------------

        communication_start = time.perf_counter()

        time.sleep(COMMUNICATION_DELAY_SECONDS)

        communication_time = (
            time.perf_counter() - communication_start
        )

        # --------------------------------------------------
        # WORKER 2
        # --------------------------------------------------

        worker_2_start = time.perf_counter()

        for layer in layers[SPLIT_LAYER:]:

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

        # --------------------------------------------------
        # FINAL LAYERS
        # --------------------------------------------------

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
        total_time,
        logits
    )


# ----------------------------------------------------------
# WARM-UP
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("WARM-UP")
print("=" * 70)

for i in range(WARMUP_RUNS):

    run_partition()

    print(f"Warm-up {i + 1}/{WARMUP_RUNS} completed.")


# ----------------------------------------------------------
# MEASURED RUNS
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("MEASURED RUNS")
print("=" * 70)

results = []

for i in range(MEASURED_RUNS):

    (
        worker_1_time,
        communication_time,
        worker_2_time,
        final_time,
        total_time,
        logits
    ) = run_partition()

    results.append({
        "worker_1_time": worker_1_time,
        "communication_time": communication_time,
        "worker_2_time": worker_2_time,
        "final_time": final_time,
        "total_time": total_time
    })

    print(
        f"Run {i + 1}/{MEASURED_RUNS} | "
        f"W1: {worker_1_time:.4f}s | "
        f"Comm: {communication_time * 1000:.2f}ms | "
        f"W2: {worker_2_time:.4f}s | "
        f"Final: {final_time:.4f}s | "
        f"Total: {total_time:.4f}s"
    )


# ----------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------

worker_1_times = [
    r["worker_1_time"] for r in results
]

communication_times = [
    r["communication_time"] for r in results
]

worker_2_times = [
    r["worker_2_time"] for r in results
]

final_times = [
    r["final_time"] for r in results
]

total_times = [
    r["total_time"] for r in results
]


mean_w1 = statistics.mean(worker_1_times)
mean_comm = statistics.mean(communication_times)
mean_w2 = statistics.mean(worker_2_times)
mean_final = statistics.mean(final_times)
mean_total = statistics.mean(total_times)


print("\n" + "=" * 70)
print("STATIC PARTITION SUMMARY")
print("=" * 70)

print(f"Split: {SPLIT_LAYER} / {len(layers) - SPLIT_LAYER}")

print(f"\nWorker 1 mean time: {mean_w1:.4f} seconds")
print(f"Communication mean: {mean_comm * 1000:.2f} ms")
print(f"Worker 2 mean time: {mean_w2:.4f} seconds")
print(f"Final computation:   {mean_final:.4f} seconds")
print(f"Total mean time:     {mean_total:.4f} seconds")

if mean_total > 0:
    print(
        f"\nApprox. partition throughput: "
        f"{10 / mean_total:.4f} tokens/sec"
    )

print("=" * 70)


# ----------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------

timestamp = datetime.now().isoformat(timespec="seconds")

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
            "worker_1_time_seconds",
            "communication_time_seconds",
            "worker_2_time_seconds",
            "final_time_seconds",
            "total_time_seconds",
            "throughput_tokens_per_second"
        ])

    writer.writerow([
        timestamp,
        "Llama 3.2 3B",
        SPLIT_LAYER,
        SPLIT_LAYER,
        len(layers) - SPLIT_LAYER,
        round(mean_w1, 6),
        round(mean_comm, 6),
        round(mean_w2, 6),
        round(mean_final, 6),
        round(mean_total, 6),
        round(10 / mean_total, 6)
    ])


print("\nResults saved to:")
print(RESULT_FILE)

print("\nStatic partition benchmark completed successfully!")
print("=" * 70)