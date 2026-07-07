import wfdb
import pandas as pd
import numpy as np

NORMAL_SYMBOLS = {"N"}

ABNORMAL_SYMBOLS = {
    "A",
    "V",
    "Q",
    "f",
    "/"
}

records = [
    "100","101","102","103","104",
    "105","106","107","108","109",
    "111","112","113","114","115",
    "116","117","118","119","121",
    "122","123","124","200","201",
    "202","203","205","207","208"
]

WINDOW_SIZE = 500

all_rows = []

for record_name in records:

    print(f"\nProcessing {record_name}")

    try:

        record = wfdb.rdrecord(
            record_name,
            pn_dir="mitdb"
        )

        ann = wfdb.rdann(
            record_name,
            "atr",
            pn_dir="mitdb"
        )

    except Exception as e:

        print("Skipped:", e)
        continue

    ecg = record.p_signal[:, 0]

    ann_samples = ann.sample
    ann_symbols = ann.symbol

    for start in range(
        0,
        len(ecg) - WINDOW_SIZE,
        WINDOW_SIZE
    ):

        end = start + WINDOW_SIZE

        segment = ecg[start:end]

        symbols_in_window = []

        for s, sym in zip(
            ann_samples,
            ann_symbols
        ):
            if start <= s < end:
                symbols_in_window.append(sym)

        label = 0

        if any(
            sym in ABNORMAL_SYMBOLS
            for sym in symbols_in_window
        ):
            label = 1

        row = segment.tolist()

        row.append(label)

        all_rows.append(row)

print("\nCreating dataframe...")

columns = [
    f"x{i}"
    for i in range(WINDOW_SIZE)
]

columns.append("Label")

df = pd.DataFrame(
    all_rows,
    columns=columns
)

print("\nShape:")
print(df.shape)

print("\nLabel Counts:")
print(df["Label"].value_counts())

df.to_csv(
    "raw_ecg_dataset.csv",
    index=False
)

print("\nSaved raw_ecg_dataset.csv")