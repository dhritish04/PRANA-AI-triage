import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class ECGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------
# Load data
# -----------------------------------------

df = pd.read_csv(
    "real_labeled_ecg_dataset.csv"
)

X = df[
    [
        "Mean_HR",
        "Mean_RR",
        "SDNN",
        "RMSSD",
        "pNN50"
    ]
].values

y = df["Label"].values

# -----------------------------------------
# Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------
# Scale
# -----------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------
# Dataset + Loader
# -----------------------------------------

train_dataset = ECGDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# -----------------------------------------
# Model
# -----------------------------------------

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

# -----------------------------------------
# Training
# -----------------------------------------

for epoch in range(50):

    total_loss = 0

    for features, labels in train_loader:

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:3d} "
            f"Loss = {total_loss:.4f}"
        )

# -----------------------------------------
# Evaluation
# -----------------------------------------

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

with torch.no_grad():

    outputs = model(X_test_tensor)

    _, predicted = torch.max(outputs, 1)

    accuracy = (
        (predicted == y_test_tensor)
        .sum()
        .item()
        / len(y_test_tensor)
    )

print()
print(f"Test Accuracy = {accuracy*100:.2f}%")

cm = confusion_matrix(
    y_test_tensor.numpy(),
    predicted.numpy()
)

print("\nConfusion Matrix:")
print(cm)