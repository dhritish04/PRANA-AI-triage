import torch
import torch.nn as nn


# ==================================================
# CNN Architecture
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


# ==================================================
# Load Model Once
# ==================================================

model = ECGCNN()

model.load_state_dict(
    torch.load(
        "ecg_cnn_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# ==================================================
# Prediction Function
# ==================================================

def predict_ecg_cnn(ecg_segment):

    """
    ecg_segment:
        list or numpy array
        containing 500 ECG samples
    """

    segment = torch.FloatTensor(
        ecg_segment
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(segment)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities.max().item()
            * 100
        )

    return {
        "status":
            "ANOMALOUS"
            if prediction == 1
            else "NORMAL",

        "confidence":
            confidence
    }

if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv(
        "raw_ecg_dataset.csv"
    )

    segment = (
        df.drop(columns=["Label"])
          .iloc[0]
          .values
    )

    result = predict_ecg_cnn(
        segment
    )

    print(result)