import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ECGDataset(Dataset):

    def __init__(self, csv_file):

        df = pd.read_csv(csv_file)

        self.X = torch.FloatTensor(
            df[
                [
                    "Mean_HR",
                    "Mean_RR",
                    "SDNN",
                    "RMSSD",
                    "pNN50"
                ]
            ].values
        )

        self.y = torch.LongTensor(
            df["Label"].values
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = ECGDataset(
    "real_labeled_ecg_dataset.csv"
)

print("Dataset Size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("Dataset Size:", len(dataset))

for features, labels in loader:

    print("\nBatch Features Shape:")
    print(features.shape)

    print("\nBatch Labels Shape:")
    print(labels.shape)

    break