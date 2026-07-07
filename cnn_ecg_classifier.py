import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ==================================================
# Dataset
# ==================================================

class RawECGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# ==================================================
# Load Data
# ==================================================

print("Loading dataset...")

df = pd.read_csv("raw_ecg_dataset.csv")

X = df.drop(columns=["Label"]).values
y = df["Label"].values

print("Shape:", X.shape)

# Use only part of dataset initially
X = X[:10000]
y = y[:10000]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

train_dataset = RawECGDataset(
    X_train,
    y_train
)

test_dataset = RawECGDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# ==================================================
# CNN Model
# ==================================================

class ECGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=5
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(
            16 * 248,
            64
        )

        self.fc2 = nn.Linear(
            64,
            2
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x


model = ECGCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==================================================
# Training
# ==================================================

print("\nTraining CNN...\n")

for epoch in range(10):

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

    print(
        f"Epoch {epoch+1:2d} "
        f"Loss = {total_loss:.4f}"
    )
    torch.save(
    model.state_dict(),
    "ecg_cnn_model.pth"
    )

    print("Model saved!")

# ==================================================
# Evaluation
# ==================================================

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for features, labels in DataLoader(
        test_dataset,
        batch_size=64
    ):

        outputs = model(features)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(
            predicted.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = (correct / total) * 100

print()
print(f"Test Accuracy = {accuracy:.2f}%")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)