import torch

from partitioning.rl_partition_controller import (
    RLPartitionController
)

from partitioning import graph_rewriter


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_LAYERS = 28


# ============================================================
# TEST TELEMETRY STATES
# ============================================================

test_states = [
    [0.50, 0.50, 0.02, 0.10],
    [0.80, 0.40, 0.10, 0.30],
    [0.40, 0.80, 0.20, 0.50],
    [0.70, 0.70, 0.50, 0.70],
    [0.30, 0.60, 0.90, 0.90]
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DQN + FX GRAPH REWRITER INTEGRATION")
print("=" * 70)


# ============================================================
# CREATE RL CONTROLLER
# ============================================================

print("\nLoading trained RL controller...")

controller = RLPartitionController()

print("RL controller loaded successfully.")


# ============================================================
# USE EXISTING GRAPH REWRITER
# ============================================================

print("\nExisting FX graph rewriter loaded.")

original_graph = graph_rewriter.graph_module

print(
    f"Original graph layers: "
    f"{len(graph_rewriter.layers)}"
)


# ============================================================
# DYNAMIC GRAPH INTEGRATION
# ============================================================

print("\n" + "-" * 70)
print("DYNAMIC RL-BASED GRAPH PARTITIONING")
print("-" * 70)


successful_tests = 0


for index, state in enumerate(
    test_states,
    start=1
):

    print("\n" + "-" * 70)

    print(f"Test {index}")
    print(f"Telemetry state: {state}")

    # --------------------------------------------------------
    # DQN chooses action
    # --------------------------------------------------------

    result = controller.decide(state)

    action = result["action"]
    action_name = result["action_name"]

    split_point = result["new_split"]

    print(
        f"DQN action: "
        f"{action} - {action_name}"
    )

    print(
        f"Selected split: "
        f"{split_point}/"
        f"{TOTAL_LAYERS - split_point}"
    )

    # --------------------------------------------------------
    # Rewrite FX graph using DQN split
    # --------------------------------------------------------

    try:

        rewritten_graph = (
            graph_rewriter.create_partitioned_graph(
                original_graph,
                split_point
            )
        )

        # ----------------------------------------------------
        # Inspect rewritten graph
        # ----------------------------------------------------

        layer_nodes = []

        communication_nodes = []

        for node in rewritten_graph.graph.nodes:

            if node.op == "call_module":

                module = rewritten_graph.get_submodule(
                    node.target
                )

                if (
                    module.__class__.__name__
                    == "LlamaDecoderLayer"
                ):

                    layer_nodes.append(node)

            elif (
                node.op == "call_function"
                and node.target
                == graph_rewriter.simulate_communication
            ):

                communication_nodes.append(node)

        print(
            f"Worker 1: layers "
            f"0 - {split_point - 1}"
        )

        print(
            f"Worker 2: layers "
            f"{split_point} - {TOTAL_LAYERS - 1}"
        )

        print(
            f"Layer nodes: "
            f"{len(layer_nodes)}"
        )

        print(
            f"Communication boundaries: "
            f"{len(communication_nodes)}"
        )

        # ----------------------------------------------------
        # Validate graph
        # ----------------------------------------------------

        if (
            len(layer_nodes) == TOTAL_LAYERS
            and len(communication_nodes) == 1
            and 8 <= split_point <= 20
        ):

            print(
                "PASS: DQN decision successfully "
                "rewrote the FX graph."
            )

            successful_tests += 1

        else:

            print(
                "FAIL: Rewritten graph structure "
                "is incorrect."
            )

    except Exception as error:

        print(
            f"ERROR during graph rewriting: "
            f"{error}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INTEGRATION SUMMARY")
print("=" * 70)

print(
    f"Successful tests: "
    f"{successful_tests} / {len(test_states)}"
)

print()

if successful_tests == len(test_states):

    print(
        "PASS: DQN + FX graph rewriter "
        "integration is working."
    )

else:

    print(
        "WARNING: Some integration tests failed."
    )

print("=" * 70)