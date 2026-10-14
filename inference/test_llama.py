import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

print("=" * 60)
print("Llama 3.2 3B - Inference Test")
print("=" * 60)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Tokenizer loaded successfully.")

print("\nLoading Llama 3.2 3B model...")
print("This may take some time on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True
)

model.eval()

print("Model loaded successfully!")

prompt = "Explain machine learning in one simple sentence."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

print("\nPrompt:")
print(prompt)

print("\nGenerating 10 tokens...")

start_time = time.perf_counter()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

end_time = time.perf_counter()

generated_text = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

generation_time = end_time - start_time

input_tokens = inputs["input_ids"].shape[1]
total_tokens = output.shape[1]
new_tokens = total_tokens - input_tokens

print("\nGenerated text:")
print(generated_text)

print("\n" + "=" * 60)
print("Llama baseline test completed")
print("=" * 60)
print(f"Generation time: {generation_time:.2f} seconds")
print(f"New tokens generated: {new_tokens}")

if generation_time > 0:
    throughput = new_tokens / generation_time
    print(f"Approx. throughput: {throughput:.2f} tokens/sec")

print("=" * 60)