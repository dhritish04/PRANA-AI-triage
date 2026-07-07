import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np

# Load real ECG
record = wfdb.rdrecord(
    "100",
    pn_dir="mitdb"
)

# First ECG channel
ecg = record.p_signal[:, 0]

# Analyze first 5000 samples
ecg_segment = ecg[:5000]

# Detect R peaks
signals, info = nk.ecg_process(
    ecg_segment,
    sampling_rate=360
)

r_peaks = info["ECG_R_Peaks"]

print("R Peaks:")
print(r_peaks)

# RR intervals in samples
rr_intervals = np.diff(r_peaks)

print("\nRR Intervals (samples):")
print(rr_intervals)

# Convert RR intervals to seconds
rr_seconds = rr_intervals / 360

print("\nRR Intervals (seconds):")
print(rr_seconds)

# Heart Rate
heart_rates = 60 / rr_seconds

print("\nHeart Rate (BPM):")
print(heart_rates)

mean_hr = np.mean(heart_rates)

print("\nAverage Heart Rate:")
print(mean_hr)

# Mean RR
mean_rr = np.mean(rr_seconds)

print("\nMean RR:")
print(mean_rr)

# SDNN
sdnn = np.std(rr_seconds)

print("\nSDNN:")
print(sdnn)

# RMSSD
rmssd = np.sqrt(
    np.mean(
        np.diff(rr_seconds) ** 2
    )
)

print("\nRMSSD:")
print(rmssd)

# pNN50
rr_diff = np.abs(np.diff(rr_seconds))

nn50 = np.sum(rr_diff > 0.05)

pnn50 = (nn50 / len(rr_diff)) * 100

print("\npNN50:")
print(pnn50)

# Feature Vector
features = {
    "Mean_HR": mean_hr,
    "Mean_RR": mean_rr,
    "SDNN": sdnn,
    "RMSSD": rmssd,
    "pNN50": pnn50
}

print("\nFeature Vector:")
print(features)

# Plot ECG with R peaks
plt.figure(figsize=(12,4))
plt.plot(ecg_segment)

plt.scatter(
    r_peaks,
    ecg_segment[r_peaks],
    color="red",
    label="R Peaks"
)

plt.legend()
plt.title("Real ECG with R Peak Detection")
plt.show()