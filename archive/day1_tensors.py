import torch

print("PyTorch Version:", torch.__version__)

# Vector
x = torch.tensor([1, 2, 3, 4])

print("\nVector:")
print(x)

print("\nShape:")
print(x.shape)

# Matrix
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("\nMatrix:")
print(A)

print("\nMatrix Shape:")
print(A.shape)

# Basic math
print("\nMultiply by 2:")
print(A * 2)

print("\nAdd 10:")
print(A + 10)