import pandas as pd

df = pd.read_csv("real_labeled_ecg_dataset.csv")

print(df.head())

print()
print(df.shape)

print()
print(df.columns)

print()
print(df["Label"].value_counts())