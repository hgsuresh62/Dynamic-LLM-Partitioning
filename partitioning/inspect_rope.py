import inspect

from transformers import AutoModelForCausalLM


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"


print("=" * 70)
print("Llama 3.2 3B - RoPE Inspection")
print("=" * 70)

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="bfloat16",
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

print("Model loaded successfully.")

print("\nRotary embedding module:")
print("-" * 70)

print(type(model.model.rotary_emb).__name__)

print("\nRotary embedding forward signature:")
print("-" * 70)

print(inspect.signature(model.model.rotary_emb.forward))

print("\nRotary embedding object:")
print("-" * 70)

print(model.model.rotary_emb)

print("\n" + "=" * 70)
print("RoPE inspection completed")
print("=" * 70)