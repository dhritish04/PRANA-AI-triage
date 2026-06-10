import wfdb
import neurokit2 as nk
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
    "100",
    "101",
    "102",
    "103",
    "104"
]

all_rows = []

WINDOW_SIZE = 5000

for record_name in records:

    print(f"\nProcessing {record_name}")

    record = wfdb.rdrecord(
        record_name,
        pn_dir="mitdb"
    )

    ann = wfdb.rdann(
        record_name,
        "atr",
        pn_dir="mitdb"
    )

    ecg = record.p_signal[:,0]
    fs = record.fs

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

        try:

            signals, info = nk.ecg_process(
                segment,
                sampling_rate=fs
            )

            r_peaks = info["ECG_R_Peaks"]

            if len(r_peaks) < 3:
                continue

            rr = np.diff(r_peaks)

            rr_sec = rr / fs

            hr = 60 / rr_sec

            mean_hr = np.mean(hr)

            mean_rr = np.mean(rr_sec)

            sdnn = np.std(rr_sec)

            rmssd = np.sqrt(
                np.mean(
                    np.diff(rr_sec) ** 2
                )
            )

            rr_diff = np.abs(
                np.diff(rr_sec)
            )

            pnn50 = (
                np.sum(
                    rr_diff > 0.05
                )
                /
                len(rr_diff)
            ) * 100

            all_rows.append([
                mean_hr,
                mean_rr,
                sdnn,
                rmssd,
                pnn50,
                label
            ])

        except:
            continue

df = pd.DataFrame(
    all_rows,
    columns=[
        "Mean_HR",
        "Mean_RR",
        "SDNN",
        "RMSSD",
        "pNN50",
        "Label"
    ]
)

print("\nShape:")
print(df.shape)

print("\nLabel Counts:")
print(df["Label"].value_counts())

df.to_csv(
    "real_labeled_ecg_dataset.csv",
    index=False
)

print("\nSaved!")