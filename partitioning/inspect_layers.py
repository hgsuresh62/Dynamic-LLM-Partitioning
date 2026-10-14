import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"


print("=" * 70)
print("Llama 3.2 3B - Transformer Layer Inspection")
print("=" * 70)

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")

print("\nModel class:")
print(type(model).__name__)

print("\nTransformer layers:")
print("-" * 70)

layers = model.model.layers

print(f"Number of layers found: {len(layers)}")

for i, layer in enumerate(layers):
    print(f"Layer {i}: {type(layer).__name__}")

print("\n" + "=" * 70)
print("Layer inspection completed")
print("=" * 70)