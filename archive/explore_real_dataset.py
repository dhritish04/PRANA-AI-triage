import pandas as pd

df = pd.read_csv(
    "real_ecg_features.csv"
)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())