import torch
import torch.fx as fx

from transformers import AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

# Split points we want to test
SPLIT_POINTS = [8, 10, 12, 14, 16, 18, 20]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("WEEK 2 - DYNAMIC GRAPH REWRITING")
print("=" * 70)

print("\nLoading Llama 3.2 3B...")

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


# ============================================================
# LAYER STACK
# ============================================================

class LlamaLayerStack(torch.nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


layer_stack = LlamaLayerStack(layers)


# ============================================================
# CUSTOM FX TRACER
# ============================================================

class LlamaTracer(fx.Tracer):

    def is_leaf_module(self, m, module_qualified_name):

        if m.__class__.__name__ == "LlamaDecoderLayer":
            return True

        return super().is_leaf_module(
            m,
            module_qualified_name
        )


# ============================================================
# TRACE ORIGINAL GRAPH
# ============================================================

print("\nTracing original graph...")

tracer = LlamaTracer()

graph = tracer.trace(layer_stack)

graph_module = fx.GraphModule(
    layer_stack,
    graph
)

print("Original graph traced successfully.")


# ============================================================
# GRAPH REWRITER
# ============================================================

def create_partitioned_graph(original_graph_module, split_point):

    # Make a fresh graph.
    new_graph = fx.Graph()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_node = new_graph.placeholder("x")

    current_node = input_node

    # --------------------------------------------------------
    # Partition 1
    # --------------------------------------------------------

    for layer_index in range(split_point):

        layer_name = f"layers.{layer_index}"

        current_node = new_graph.call_module(
            layer_name,
            args=(current_node,)
        )

    # --------------------------------------------------------
    # Partition boundary
    # --------------------------------------------------------

    worker_1_output = current_node

    # This node represents the communication boundary.
    communication_node = new_graph.call_function(
        simulate_communication,
        args=(worker_1_output,)
    )

    # --------------------------------------------------------
    # Partition 2
    # --------------------------------------------------------

    current_node = communication_node

    total_layers = len(layers)

    for layer_index in range(
        split_point,
        total_layers
    ):

        layer_name = f"layers.{layer_index}"

        current_node = new_graph.call_module(
            layer_name,
            args=(current_node,)
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    new_graph.output(current_node)

    # Create GraphModule
    rewritten_module = fx.GraphModule(
        original_graph_module,
        new_graph
    )

    return rewritten_module


# ============================================================
# SIMULATED COMMUNICATION FUNCTION
# ============================================================

def simulate_communication(x):

    # This function represents the boundary between
    # two distributed workers.

    return x


# ============================================================
# TEST GRAPH REWRITING
# ============================================================

print("\n" + "=" * 70)
print("TESTING DYNAMIC PARTITION REWRITING")
print("=" * 70)


successful_splits = []


for split_point in SPLIT_POINTS:

    print("\n" + "-" * 70)

    print(
        f"Testing dynamic split: "
        f"{split_point} / {len(layers) - split_point}"
    )

    try:

        rewritten = create_partitioned_graph(
            graph_module,
            split_point
        )

        # Count decoder layer nodes
        layer_nodes = []

        communication_nodes = []

        for node in rewritten.graph.nodes:

            if node.op == "call_module":

                module = rewritten.get_submodule(
                    node.target
                )

                if (
                    module.__class__.__name__
                    == "LlamaDecoderLayer"
                ):

                    layer_nodes.append(node)

            elif (
                node.op == "call_function"
                and node.target == simulate_communication
            ):

                communication_nodes.append(node)

        print(
            f"Worker 1 layers: 0 - {split_point - 1}"
        )

        print(
            f"Worker 2 layers: "
            f"{split_point} - {len(layers) - 1}"
        )

        print(
            f"Total layer nodes: "
            f"{len(layer_nodes)}"
        )

        print(
            f"Communication boundaries: "
            f"{len(communication_nodes)}"
        )

        # Verify graph
        if (
            len(layer_nodes) == len(layers)
            and len(communication_nodes) == 1
        ):

            print("PASS: Graph successfully rewritten.")

            successful_splits.append(split_point)

        else:

            print("FAIL: Graph structure is incorrect.")

    except Exception as error:

        print(
            f"ERROR while rewriting split "
            f"{split_point}:"
        )

        print(error)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAPH REWRITING SUMMARY")
print("=" * 70)

print(
    f"Tested split points: {SPLIT_POINTS}"
)

print(
    f"Successful split points: {successful_splits}"
)

print(
    f"Successful tests: "
    f"{len(successful_splits)} / "
    f"{len(SPLIT_POINTS)}"
)

print()

if len(successful_splits) == len(SPLIT_POINTS):

    print(
        "PASS: Dynamic graph rewriting works "
        "for all tested split points."
    )

else:

    print(
        "WARNING: Some split points failed."
    )

print("\n" + "=" * 70)
print("Week 2 graph rewriting experiment completed!")
print("=" * 70)