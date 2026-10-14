import inspect

from transformers import AutoModelForCausalLM


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"


print("=" * 70)
print("Llama 3.2 3B - Decoder Layer API Inspection")
print("=" * 70)

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="bfloat16",
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

layer = model.model.layers[0]

print("\nModel class:")
print(type(model).__name__)

print("\nDecoder layer class:")
print(type(layer).__name__)

print("\nDecoder layer forward signature:")
print("-" * 70)
print(inspect.signature(layer.forward))

print("\nModel forward signature:")
print("-" * 70)
print(inspect.signature(model.forward))

print("\n" + "=" * 70)
print("Layer API inspection completed")
print("=" * 70)