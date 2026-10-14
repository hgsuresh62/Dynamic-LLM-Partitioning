import torch
import torch.fx as fx

from transformers import AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("WEEK 2 - TORCH.FX GRAPH TRACING")
print("=" * 70)

print("\nLoading Llama 3.2 3B...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# GET TRANSFORMER LAYERS
# ============================================================

layers = model.model.layers

print(f"\nNumber of transformer layers: {len(layers)}")


# ============================================================
# CREATE LAYER STACK
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

        # Keep every Llama decoder layer as one graph node.
        if m.__class__.__name__ == "LlamaDecoderLayer":
            return True

        return super().is_leaf_module(
            m,
            module_qualified_name
        )


# ============================================================
# TRACE MODEL
# ============================================================

print("\nTracing transformer layer structure...")

tracer = LlamaTracer()

graph = tracer.trace(layer_stack)

graph_module = fx.GraphModule(
    layer_stack,
    graph
)

print("Graph tracing completed successfully.")


# ============================================================
# DISPLAY GRAPH
# ============================================================

print("\n" + "=" * 70)
print("TORCH.FX GRAPH")
print("=" * 70)

print(graph_module.graph)


# ============================================================
# ANALYZE GRAPH NODES
# ============================================================

print("\n" + "=" * 70)
print("GRAPH NODE ANALYSIS")
print("=" * 70)

layer_nodes = []

for node in graph_module.graph.nodes:

    if node.op == "call_module":

        module = graph_module.get_submodule(
            node.target
        )

        if module.__class__.__name__ == "LlamaDecoderLayer":

            layer_nodes.append(node)

            print(
                f"Layer node {len(layer_nodes) - 1}: "
                f"{node.target}"
            )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAPH SUMMARY")
print("=" * 70)

print(
    f"Total FX graph nodes: "
    f"{len(list(graph_module.graph.nodes))}"
)

print(
    f"Llama decoder layer nodes detected: "
    f"{len(layer_nodes)}"
)

print()

if len(layer_nodes) == len(layers):

    print("PASS: All 28 transformer layers were detected.")

else:

    print(
        "WARNING: Expected "
        f"{len(layers)} layers but detected "
        f"{len(layer_nodes)}."
    )

print("\n" + "=" * 70)
print("Week 2 FX tracing experiment completed!")
print("=" * 70)