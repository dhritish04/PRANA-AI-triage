import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

print(model)