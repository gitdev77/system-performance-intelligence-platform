import torch
import torch.nn as nn
import time

# Simple feedforward model
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

def heavy_computation(batch_size):
    start = time.time()

    input_tensor = torch.randn(batch_size, 512)

    output = model(input_tensor)

    end = time.time()

    return end - start