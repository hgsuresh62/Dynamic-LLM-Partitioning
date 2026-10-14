import time
from transformers import pipeline

print("==========================================")
print("LLM Inference Performance Baseline")
print("==========================================")

print("\nLoading test language model...")

generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2"
)

prompt = "Machine learning is"

# ------------------------------------------------
# Warm-up
# ------------------------------------------------
print("\nRunning warm-up...")

generator(
    prompt,
    max_new_tokens=10
)

# ------------------------------------------------
# Actual benchmark
# ------------------------------------------------
num_tokens = 20

print("\nRunning benchmark...")

start_time = time.perf_counter()

result = generator(
    prompt,
    max_new_tokens=num_tokens
)

end_time = time.perf_counter()

# ------------------------------------------------
# Calculate performance
# ------------------------------------------------
generation_time = end_time - start_time

generated_text = result[0]["generated_text"]

throughput = num_tokens / generation_time

print("\n==========================================")
print("RESULTS")
print("==========================================")

print("\nGenerated text:")
print(generated_text)

print("\nGeneration time:",
      round(generation_time, 4),
      "seconds")

print("Requested new tokens:",
      num_tokens)

print("Approx. throughput:",
      round(throughput, 2),
      "tokens/sec")

print("\n==========================================")
print("Baseline measurement completed!")
print("==========================================")