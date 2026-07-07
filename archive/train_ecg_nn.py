import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("real_labeled_ecg_dataset.csv")

X = df[[
    "Mean_HR",
    "Mean_RR",
    "SDNN",
    "RMSSD",
    "pNN50"
]]

y = df["Label"]

# --------------------------------------------------
# Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --------------------------------------------------
# Convert to tensors
# --------------------------------------------------

X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train.values)
y_test  = torch.LongTensor(y_test.values)

# --------------------------------------------------
# Neural network
# --------------------------------------------------

model = nn.Sequential(
    nn.Linear(5,16),
    nn.ReLU(),

    nn.Linear(16,8),
    nn.ReLU(),

    nn.Linear(8,2)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(200):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d} "
            f"Loss = {loss.item():.4f}"
        )

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with torch.no_grad():

    outputs = model(X_test)

    _, predicted = torch.max(outputs, 1)

    accuracy = (
        (predicted == y_test)
        .sum()
        .item()
        / len(y_test)
    )

print()
print("Test Accuracy =", accuracy * 100)