import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

SPLIT_LAYER = 14


print("=" * 70)
print("Static Llama 3.2 3B Layer Partition")
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

print(f"\nTotal transformer layers: {len(layers)}")

worker_1_layers = layers[:SPLIT_LAYER]
worker_2_layers = layers[SPLIT_LAYER:]

print(f"Worker 1 layers: 0 - {SPLIT_LAYER - 1}")
print(f"Worker 2 layers: {SPLIT_LAYER} - {len(layers) - 1}")

prompt = "Explain machine learning in one simple sentence."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]

print("\nPrompt:")
print(prompt)

print("\nRunning Worker 1...")

with torch.no_grad():

    hidden_states = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(
        input_ids.shape[1],
        device=input_ids.device
    ).unsqueeze(0)

    position_embeddings = model.model.rotary_emb(
    hidden_states,
    position_ids
)

    for layer in worker_1_layers:

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

print("Worker 1 completed.")

print("\nSimulating communication between workers...")

communication_start = time.perf_counter()

time.sleep(0.010)

communication_time = time.perf_counter() - communication_start

print(f"Simulated communication time: {communication_time * 1000:.2f} ms")

print("\nRunning Worker 2...")

with torch.no_grad():

    for layer in worker_2_layers:

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

print("Worker 2 completed.")

print("\nRunning final normalization and LM head...")

with torch.no_grad():

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)

print("Final computation completed.")

print("\n" + "=" * 70)
print("STATIC PARTITION RESULT")
print("=" * 70)

print(f"Input shape: {input_ids.shape}")
print(f"Hidden-state shape: {hidden_states.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Worker 1 layers: {len(worker_1_layers)}")
print(f"Worker 2 layers: {len(worker_2_layers)}")
print(f"Communication delay: {communication_time * 1000:.2f} ms")

print("=" * 70)
print("Static partition test completed successfully.")
print("=" * 70)