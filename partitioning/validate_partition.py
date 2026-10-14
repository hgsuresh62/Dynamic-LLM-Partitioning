import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"

SPLIT_LAYER = 14

PROMPT = "Explain machine learning in one simple sentence."


print("=" * 70)
print("Llama 3.2 3B - Partition Correctness Validation")
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

inputs = tokenizer(
    PROMPT,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]

print("\nInput shape:")
print(input_ids.shape)


# ------------------------------------------------------------
# 1. NORMAL LLAMA EXECUTION
# ------------------------------------------------------------

print("\nRunning normal Llama execution...")

with torch.no_grad():

    normal_output = model(
        input_ids=input_ids,
        use_cache=False
    )

normal_logits = normal_output.logits

print("Normal execution completed.")


# ------------------------------------------------------------
# 2. STATIC 14/14 PARTITION
# ------------------------------------------------------------

print("\nRunning 14/14 static partition...")

layers = model.model.layers

worker_1_layers = layers[:SPLIT_LAYER]
worker_2_layers = layers[SPLIT_LAYER:]

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

    # Worker 1
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

    # Worker 2
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

    # Final normalization
    hidden_states = model.model.norm(hidden_states)

    # LM head
    partitioned_logits = model.lm_head(hidden_states)

print("Partitioned execution completed.")


# ------------------------------------------------------------
# 3. COMPARE RESULTS
# ------------------------------------------------------------

print("\nComparing logits...")

difference = torch.abs(
    normal_logits - partitioned_logits
)

max_difference = difference.max().item()
mean_difference = difference.mean().item()

print("\n" + "=" * 70)
print("CORRECTNESS RESULT")
print("=" * 70)

print(f"Normal logits shape:      {normal_logits.shape}")
print(f"Partitioned logits shape: {partitioned_logits.shape}")

print(f"\nMaximum absolute difference: {max_difference:.10f}")
print(f"Mean absolute difference:    {mean_difference:.10f}")

print("=" * 70)

if max_difference < 1e-2:
    print("PASS: Partition output matches normal Llama closely.")
else:
    print("WARNING: Partition output differs from normal Llama.")

print("=" * 70)
print("Partition correctness validation completed.")
print("=" * 70)