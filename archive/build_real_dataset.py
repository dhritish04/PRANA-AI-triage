import wfdb
import neurokit2 as nk
import pandas as pd
import numpy as np

records = [
    "100",
    "101",
    "102",
    "103",
    "104"
]

all_features = []

for record_name in records:

    print(f"Processing record {record_name}")

    record = wfdb.rdrecord(
        record_name,
        pn_dir="mitdb"
    )

    ecg = record.p_signal[:, 0]

    fs = record.fs

    segment_length = 5000

    for start in range(
        0,
        len(ecg) - segment_length,
        segment_length
    ):

        segment = ecg[
            start:start + segment_length
        ]

        try:

            signals, info = nk.ecg_process(
                segment,
                sampling_rate=fs
            )

            r_peaks = info["ECG_R_Peaks"]

            if len(r_peaks) < 3:
                continue

            rr_intervals = np.diff(r_peaks)

            rr_seconds = rr_intervals / fs

            heart_rates = 60 / rr_seconds

            mean_hr = np.mean(heart_rates)

            mean_rr = np.mean(rr_seconds)

            sdnn = np.std(rr_seconds)

            rmssd = np.sqrt(
                np.mean(
                    np.diff(rr_seconds) ** 2
                )
            )

            rr_diff = np.abs(
                np.diff(rr_seconds)
            )

            nn50 = np.sum(
                rr_diff > 0.05
            )

            pnn50 = (
                nn50 /
                len(rr_diff)
            ) * 100

            all_features.append([
                mean_hr,
                mean_rr,
                sdnn,
                rmssd,
                pnn50
            ])

        except:
            continue

df = pd.DataFrame(
    all_features,
    columns=[
        "Mean_HR",
        "Mean_RR",
        "SDNN",
        "RMSSD",
        "pNN50"
    ]
)

print("\nDataset Shape:")
print(df.shape)

df.to_csv(
    "real_ecg_features.csv",
    index=False
)

print("\nCSV Saved!")