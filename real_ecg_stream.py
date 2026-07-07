import pandas as pd
import numpy as np

df = pd.read_csv("raw_ecg_dataset.csv")

def get_real_ecg_segment():

    row = df.sample(1).iloc[0]

    signal = row[:-1].values.astype(float)

    label = int(row["Label"])

    return signal, label