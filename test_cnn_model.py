import torch
import torch.nn as nn
import pandas as pd


class ECGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            1, 16, 5
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

model.load_state_dict(
    torch.load("ecg_cnn_model.pth")
)

model.eval()

df = pd.read_csv(
    "raw_ecg_dataset.csv"
)

sample = torch.FloatTensor(
    df.drop(columns=["Label"])
      .iloc[0:1]
      .values
)

with torch.no_grad():

    output = model(sample)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    )

print("Prediction:", prediction.item())

print("Confidence:",
      probabilities.max().item())