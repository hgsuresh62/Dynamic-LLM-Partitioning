from transformers import AutoConfig


MODEL_PATH = r"D:\Dynamic-LLM-Partitioning\models\llama-3.2-3b"


print("=" * 60)
print("Llama 3.2 3B - Architecture Inspection")
print("=" * 60)

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("\nModel type:")
print(config.model_type)

print("\nHidden size:")
print(config.hidden_size)

print("\nNumber of transformer layers:")
print(config.num_hidden_layers)

print("\nNumber of attention heads:")
print(config.num_attention_heads)

print("\nIntermediate size:")
print(config.intermediate_size)

print("\nVocabulary size:")
print(config.vocab_size)

print("\n" + "=" * 60)
print("Architecture inspection completed")
print("=" * 60)