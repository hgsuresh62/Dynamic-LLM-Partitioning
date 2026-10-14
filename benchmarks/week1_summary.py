import csv
import os


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\Dynamic-LLM-Partitioning\benchmarks"

SPLIT_FILE = os.path.join(
    BASE_DIR,
    "split_point_results.csv"
)

JITTER_FILE = os.path.join(
    BASE_DIR,
    "network_jitter_results.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "week1_summary.csv"
)


# ============================================================
# READ SPLIT-POINT RESULTS
# ============================================================

with open(SPLIT_FILE, "r", newline="") as file:
    split_results = list(csv.DictReader(file))


# Find the best static split
best_split = min(
    split_results,
    key=lambda row: float(row["total_time_seconds"])
)


# ============================================================
# READ NETWORK-JITTER RESULTS
# ============================================================

with open(JITTER_FILE, "r", newline="") as file:
    jitter_results = list(csv.DictReader(file))


# Find best and worst network conditions
best_network = min(
    jitter_results,
    key=lambda row: float(row["mean_total_time_s"])
)

worst_network = max(
    jitter_results,
    key=lambda row: float(row["mean_total_time_s"])
)


# ============================================================
# CALCULATE THROUGHPUT CHANGE
# ============================================================

best_throughput = float(
    best_network["mean_throughput_tokens_sec"]
)

worst_throughput = float(
    worst_network["mean_throughput_tokens_sec"]
)

throughput_change_percent = (
    (best_throughput - worst_throughput)
    / best_throughput
) * 100


# ============================================================
# CREATE WEEK 1 SUMMARY
# ============================================================

summary = [
    {
        "experiment": "Best Static Split",
        "condition": (
            f"{best_split['worker_1_layers']} / "
            f"{best_split['worker_2_layers']}"
        ),
        "mean_time_s": best_split["total_time_seconds"],
        "throughput_tokens_sec": best_split[
            "throughput_tokens_per_second"
        ],
        "additional_info": "Best tested static partition"
    },
    {
        "experiment": "Best Network Condition",
        "condition": (
            f"{best_network['network_delay_ms']} ms"
        ),
        "mean_time_s": best_network["mean_total_time_s"],
        "throughput_tokens_sec": best_network[
            "mean_throughput_tokens_sec"
        ],
        "additional_info": "Lowest measured network delay"
    },
    {
        "experiment": "Worst Network Condition",
        "condition": (
            f"{worst_network['network_delay_ms']} ms"
        ),
        "mean_time_s": worst_network["mean_total_time_s"],
        "throughput_tokens_sec": worst_network[
            "mean_throughput_tokens_sec"
        ],
        "additional_info": "Highest measured network delay"
    },
    {
        "experiment": "Network Throughput Change",
        "condition": (
            f"{best_network['network_delay_ms']} ms "
            f"to {worst_network['network_delay_ms']} ms"
        ),
        "mean_time_s": "",
        "throughput_tokens_sec": "",
        "additional_info": (
            f"{throughput_change_percent:.2f}% decrease"
        )
    }
]


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    fieldnames = [
        "experiment",
        "condition",
        "mean_time_s",
        "throughput_tokens_sec",
        "additional_info"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(summary)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("WEEK 1 EXPERIMENT SUMMARY")
print("=" * 70)

print()

print(
    "Best static split: "
    f"{best_split['worker_1_layers']} / "
    f"{best_split['worker_2_layers']}"
)

print(
    "Best static mean time: "
    f"{float(best_split['total_time_seconds']):.4f}s"
)

print(
    "Best static throughput: "
    f"{float(best_split['throughput_tokens_per_second']):.4f} tok/s"
)

print()

print(
    "Best network condition: "
    f"{best_network['network_delay_ms']} ms"
)

print(
    "Best network throughput: "
    f"{best_throughput:.4f} tok/s"
)

print()

print(
    "Worst network condition: "
    f"{worst_network['network_delay_ms']} ms"
)

print(
    "Worst network throughput: "
    f"{worst_throughput:.4f} tok/s"
)

print()

print(
    "Throughput decrease from best to worst: "
    f"{throughput_change_percent:.2f}%"
)

print()

print("Summary saved to:")
print(OUTPUT_FILE)

print()
print("=" * 70)
print("Week 1 summary completed successfully!")
print("=" * 70)