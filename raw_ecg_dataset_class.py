import pandas as pd
import torch

from torch.utils.data import Dataset


class RawECGDataset(Dataset):

    def __init__(self, csv_file):

        df = pd.read_csv(csv_file)

        self.X = torch.FloatTensor(
            df.drop(
                columns=["Label"]
            ).values
        )

        self.y = torch.LongTensor(
            df["Label"].values
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )


dataset = RawECGDataset(
    "raw_ecg_dataset.csv"
)

print("Dataset Size:", len(dataset))

x, y = dataset[0]

print()
print("Waveform Shape:")
print(x.shape)

print()
print("Label:")
print(y)