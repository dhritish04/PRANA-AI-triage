import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1,1)

criterion = nn.MSELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)

x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0]])

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final Loss:", loss)

for p in model.parameters():
    print(p)