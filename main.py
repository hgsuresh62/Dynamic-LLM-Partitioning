import torch
import numpy as np

print("===================================")
print("Dynamic LLM Partitioning Project")
print("===================================")

print("Python environment: OK")
print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)

x = torch.tensor([10, 20, 30])
y = x * 2

print("Input tensor:", x)
print("Output tensor:", y)

print("===================================")
print("Project environment is working!")
print("===================================")