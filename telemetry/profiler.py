import time
import random
import psutil


# ============================================================
# TELEMETRY PROFILER
# Dynamic RL-Based LLM Execution Graph Partitioning
# ============================================================


class TelemetryProfiler:

    def __init__(self):

        self.start_time = time.perf_counter()

        print("=" * 70)
        print("TELEMETRY PROFILER")
        print("=" * 70)

        print("\nProfiler initialized successfully.")


    # ========================================================
    # CPU UTILIZATION
    # ========================================================

    def get_cpu_utilization(self):

        return psutil.cpu_percent(
            interval=0.1
        )


    # ========================================================
    # RAM UTILIZATION
    # ========================================================

    def get_ram_utilization(self):

        memory = psutil.virtual_memory()

        return memory.percent


    # ========================================================
    # SIMULATED WORKER VRAM UTILIZATION
    # ========================================================

    def get_worker_vram_utilization(self):

        # Your laptop does not have an NVIDIA CUDA GPU.
        # Therefore, real CUDA VRAM cannot be measured.
        #
        # We simulate worker resource utilization so that
        # the RL environment can still be developed.

        worker_1_vram = random.uniform(
            40.0,
            80.0
        )

        worker_2_vram = random.uniform(
            40.0,
            80.0
        )

        return (
            worker_1_vram,
            worker_2_vram
        )


    # ========================================================
    # SIMULATED NETWORK LATENCY
    # ========================================================

    def measure_network_latency(self):

        # Simulated latency in milliseconds.
        #
        # Later this can be replaced with real
        # inter-worker network measurements.

        latency_ms = random.uniform(
            2.0,
            100.0
        )

        return latency_ms


    # ========================================================
    # TOKEN SEQUENCE POSITION
    # ========================================================

    def get_token_position(
        self,
        current_token,
        total_tokens
    ):

        if total_tokens <= 0:

            return 0.0

        position = (
            current_token
            / total_tokens
        )

        return min(
            max(position, 0.0),
            1.0
        )


    # ========================================================
    # COLLECT COMPLETE TELEMETRY STATE
    # ========================================================

    def collect_state(
        self,
        current_token=1,
        total_tokens=10
    ):

        cpu_utilization = (
            self.get_cpu_utilization()
        )

        ram_utilization = (
            self.get_ram_utilization()
        )

        worker_1_vram, worker_2_vram = (
            self.get_worker_vram_utilization()
        )

        network_latency = (
            self.measure_network_latency()
        )

        token_position = (
            self.get_token_position(
                current_token,
                total_tokens
            )
        )

        state = {

            "worker_1_vram_utilization":
                worker_1_vram,

            "worker_2_vram_utilization":
                worker_2_vram,

            "network_latency_ms":
                network_latency,

            "token_position":
                token_position,

            "cpu_utilization":
                cpu_utilization,

            "ram_utilization":
                ram_utilization
        }

        return state


    # ========================================================
    # DISPLAY TELEMETRY
    # ========================================================

    def display_state(self, state):

        print("\n" + "-" * 70)
        print("CURRENT TELEMETRY STATE")
        print("-" * 70)

        print(
            f"Worker 1 resource utilization : "
            f"{state['worker_1_vram_utilization']:.2f}%"
        )

        print(
            f"Worker 2 resource utilization : "
            f"{state['worker_2_vram_utilization']:.2f}%"
        )

        print(
            f"Network latency                : "
            f"{state['network_latency_ms']:.2f} ms"
        )

        print(
            f"Token sequence position        : "
            f"{state['token_position']:.2f}"
        )

        print(
            f"CPU utilization                : "
            f"{state['cpu_utilization']:.2f}%"
        )

        print(
            f"RAM utilization                : "
            f"{state['ram_utilization']:.2f}%"
        )


# ============================================================
# TEST PROFILER
# ============================================================

if __name__ == "__main__":

    profiler = TelemetryProfiler()

    print("\nCollecting telemetry samples...")

    for token in range(1, 6):

        state = profiler.collect_state(
            current_token=token,
            total_tokens=10
        )

        print(
            f"\nSample {token}/5"
        )

        profiler.display_state(state)

        time.sleep(0.5)


    print("\n" + "=" * 70)
    print("TELEMETRY PROFILER TEST COMPLETED")
    print("=" * 70)