# =============================================================
#  PRANA PROJECT — ecg_anomaly_pipeline.py
#  Step 6: Complete ECG Anomaly Detection Pipeline
#
#  This file combines EVERYTHING from your 7 previous scripts:
#    ecg_test.py           → ECG signal generation
#    r_peak_detection.py   → R-peak detection
#    hrv_features.py       → HRV feature extraction (SDNN, RMSSD, pNN50...)
#    real_ecg.py           → Loading real ECG data
#    real_ecg_rpeaks.py    → Real ECG + R-peaks together
#    first_ml_model.py     → RandomForest classifier
#    train_test_demo.py    → Train/test split + evaluation
#
#  New things this file adds:
#    - A REAL labeled dataset (Normal vs Anomalous windows)
#    - Proper evaluation: confusion matrix, precision, recall
#    - A structured clinical TRIAGE REPORT (the actual PRĀNA output)
# =============================================================

import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
SAMPLING_RATE   = 360     # samples per second
WINDOW_SECONDS  = 10      # analyse ECG in 10-second windows
N_SAMPLES       = 80      # samples per class in synthetic mode

# Set to False when you have internet to download real MIT-BIH data
USE_SYNTHETIC   = True

FEATURE_NAMES = ["Mean_HR", "Mean_RR", "SDNN", "RMSSD", "pNN50"]


# =============================================================
# SECTION 1: Feature Extraction
# (combines hrv_features.py + r_peak_detection.py)
# =============================================================

def extract_hrv_features(ecg_signal, sampling_rate=360):
    """
    Extract 5 HRV features from one ECG window.

    Input:
        ecg_signal    — 1D array of ECG voltage values
        sampling_rate — samples per second (360 for MIT-BIH)

    Output:
        [Mean_HR, Mean_RR, SDNN, RMSSD, pNN50]
        or None if R-peaks could not be detected

    What each feature means:
        Mean_HR  — average heart rate in BPM (normal: 60-100)
        Mean_RR  — average time between beats in seconds
        SDNN     — overall HRV; low = stress/arrhythmia
        RMSSD    — short-term HRV; low = poor autonomic control
        pNN50    — % of beat intervals differing by >50ms
    """
    try:
        # Detect R-peaks (from r_peak_detection.py)
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]

        # Need at least 3 beats to compute intervals
        if len(r_peaks) < 3:
            return None

        # RR intervals → seconds (from hrv_features.py)
        rr_intervals = np.diff(r_peaks)
        rr_seconds   = rr_intervals / sampling_rate
        heart_rates  = 60 / rr_seconds

        # Compute each feature (from hrv_features.py)
        mean_hr  = np.mean(heart_rates)
        mean_rr  = np.mean(rr_seconds)
        sdnn     = np.std(rr_seconds)
        rmssd    = np.sqrt(np.mean(np.diff(rr_seconds) ** 2))
        rr_diff  = np.abs(np.diff(rr_seconds))
        pnn50    = (np.sum(rr_diff > 0.05) / len(rr_diff)) * 100

        return [mean_hr, mean_rr, sdnn, rmssd, pnn50]

    except Exception:
        return None


# =============================================================
# SECTION 2: Dataset Builder
# Two modes: synthetic (always works) + real MIT-BIH (needs internet)
# =============================================================

def build_synthetic_dataset(n_per_class=80):
    """
    Build a labeled ECG dataset using neurokit2 synthetic signals.

    Normal class (label=0):
        Heart rate 60-90 BPM, low noise — healthy sinus rhythm

    Anomalous class (label=1):
        Tachycardia:  HR > 110 BPM (too fast)
        Bradycardia:  HR < 50 BPM  (too slow)
        High noise:   irregular signal (simulates arrhythmia signal quality)
    """
    X, y = [], []
    window_samples = WINDOW_SECONDS * SAMPLING_RATE

    print("  Generating Normal ECG windows (60–90 BPM)...")
    for _ in range(n_per_class):
        hr = np.random.uniform(60, 90)
        ecg = nk.ecg_simulate(
            duration=WINDOW_SECONDS,
            sampling_rate=SAMPLING_RATE,
            heart_rate=hr,
            noise=0.05
        )
        features = extract_hrv_features(ecg, SAMPLING_RATE)
        if features:
            X.append(features)
            y.append(0)   # Label 0 = Normal

    print("  Generating Anomalous ECG windows...")
    for _ in range(n_per_class):
        anomaly = np.random.choice(['tachycardia', 'bradycardia', 'irregular'])

        if anomaly == 'tachycardia':
            # Too fast: HR > 110
            ecg = nk.ecg_simulate(
                duration=WINDOW_SECONDS,
                sampling_rate=SAMPLING_RATE,
                heart_rate=np.random.uniform(110, 160),
                noise=0.15
            )
        elif anomaly == 'bradycardia':
            # Too slow: HR < 50
            ecg = nk.ecg_simulate(
                duration=WINDOW_SECONDS,
                sampling_rate=SAMPLING_RATE,
                heart_rate=np.random.uniform(30, 50),
                noise=0.10
            )
        else:
            # Irregular: heavy noise on normal rate
            ecg = nk.ecg_simulate(
                duration=WINDOW_SECONDS,
                sampling_rate=SAMPLING_RATE,
                heart_rate=np.random.uniform(60, 90),
                noise=0.45
            )

        features = extract_hrv_features(ecg, SAMPLING_RATE)
        if features:
            X.append(features)
            y.append(1)   # Label 1 = Anomalous

    return X, y


def build_real_dataset(record_ids):
    """
    Build labeled dataset from MIT-BIH Arrhythmia Database.
    Uses beat-level annotations to label each 10-second window.

    Beat symbols meaning:
        N, L, R, e, j  → Normal beats
        V, F, A, a, J, S, E, /, f, Q → Arrhythmia beats

    A window is ANOMALOUS if it contains ANY arrhythmia beat.
    """
    import wfdb

    NORMAL_BEATS    = {'N', 'L', 'R', 'e', 'j'}
    ANOMALOUS_BEATS = {'V', 'F', 'A', 'a', 'J', 'S', 'E', '/', 'f', 'Q'}

    X, y = [], []
    window_samples = WINDOW_SECONDS * SAMPLING_RATE

    for record_id in record_ids:
        print(f"  Loading record {record_id}...", end="", flush=True)
        try:
            # Load signal + annotations together (from real_ecg.py + real_ecg_rpeaks.py)
            record     = wfdb.rdrecord(str(record_id), pn_dir="mitdb")
            annotation = wfdb.rdann(str(record_id), 'atr', pn_dir="mitdb")

            ecg         = record.p_signal[:, 0]
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol

            # Slide a 10-second window over the full signal
            for start in range(0, len(ecg) - window_samples, window_samples // 2):
                end    = start + window_samples
                window = ecg[start:end]

                # Find all annotated beats within this window
                in_window = [
                    ann_symbols[i]
                    for i in range(len(ann_samples))
                    if start <= ann_samples[i] < end
                ]

                if not in_window:
                    continue

                # Label: 1 if ANY beat in window is anomalous
                label    = 1 if any(b in ANOMALOUS_BEATS for b in in_window) else 0
                features = extract_hrv_features(window, SAMPLING_RATE)

                if features:
                    X.append(features)
                    y.append(label)

            print(f" done ({len(y)} total windows)")

        except Exception as e:
            print(f" FAILED: {e}")

    return X, y


# =============================================================
# SECTION 3: Main Pipeline
# =============================================================

print("=" * 58)
print("  PRANA — ECG Anomaly Detection Pipeline")
print("=" * 58)

# ── Step 1: Build Dataset ──────────────────────────────────
print("\n[Step 1] Building dataset...")

if USE_SYNTHETIC:
    print("  Mode: Synthetic (neurokit2)")
    X, y = build_synthetic_dataset(n_per_class=N_SAMPLES)
else:
    print("  Mode: Real MIT-BIH Database")
    # Normal-dominant records: 100, 101, 103
    # Arrhythmia records: 200, 201, 208, 210
    RECORD_IDS = ['100', '101', '103', '200', '201', '208']
    X, y = build_real_dataset(RECORD_IDS)

y_arr = np.array(y)
n_normal    = int(np.sum(y_arr == 0))
n_anomalous = int(np.sum(y_arr == 1))

print(f"\n  Dataset ready:")
print(f"    Normal windows    → {n_normal}")
print(f"    Anomalous windows → {n_anomalous}")
print(f"    Total             → {len(y)}")


# ── Step 2: Train the Model ───────────────────────────────
print("\n[Step 2] Training RandomForest model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y       # keeps class balance in train & test
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
print("  Model trained successfully!")


# ── Step 3: Evaluate ─────────────────────────────────────
print("\n[Step 3] Evaluating model...")

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)
cm          = confusion_matrix(y_test, predictions)

print(f"\n  Accuracy: {accuracy * 100:.1f}%")

print("\n  Full Classification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Normal", "Anomalous"]
))

print("  Feature Importance (what matters most to the model):")
importances = model.feature_importances_
for name, imp in sorted(
    zip(FEATURE_NAMES, importances),
    key=lambda x: x[1], reverse=True
):
    bar = "█" * int(imp * 36)
    print(f"    {name:9s}  {bar:<36s}  {imp:.3f}")


# ── Step 4: Generate Triage Report for a new patient ──────
print("\n[Step 4] Generating PRANA Triage Report for new patient...")

# Simulate a new patient with tachycardia (HR = 130 BPM)
patient_ecg = nk.ecg_simulate(
    duration=WINDOW_SECONDS,
    sampling_rate=SAMPLING_RATE,
    heart_rate=130,
    noise=0.1
)

patient_features = extract_hrv_features(patient_ecg, SAMPLING_RATE)

if patient_features:
    pred       = model.predict([patient_features])[0]
    proba      = model.predict_proba([patient_features])[0]
    confidence = max(proba) * 100

    status = "ANOMALOUS" if pred == 1 else "NORMAL"
    action = "Refer to specialist immediately" if pred == 1 else "No immediate concern"

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │         PRANA ECG TRIAGE REPORT         │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Mean Heart Rate  : {patient_features[0]:>6.1f} BPM           │")
    print(f"  │  Mean RR Interval : {patient_features[1]*1000:>6.0f} ms            │")
    print(f"  │  SDNN             : {patient_features[2]*1000:>6.1f} ms            │")
    print(f"  │  RMSSD            : {patient_features[3]*1000:>6.1f} ms            │")
    print(f"  │  pNN50            : {patient_features[4]:>6.1f} %             │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Status     : {status:<27s}  │")
    print(f"  │  Action     : {action:<27s}  │")
    print(f"  │  Confidence : {confidence:>5.1f}%                          │")
    print("  └─────────────────────────────────────────┘")


# =============================================================
# SECTION 4: Visualisation (save to file)
# =============================================================

print("\n[Step 5] Generating plots...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("PRANA — ECG Anomaly Detection Results", fontsize=16, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Patient ECG ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
signals, info = nk.ecg_process(patient_ecg, sampling_rate=SAMPLING_RATE)
r_peaks_idx = info["ECG_R_Peaks"]
ax1.plot(patient_ecg, color='royalblue', linewidth=0.8, label='ECG signal')
ax1.scatter(r_peaks_idx, patient_ecg[r_peaks_idx], color='red', s=40,
            zorder=5, label='R-peaks')
ax1.set_title(f"Patient ECG — Status: {status}  (Confidence: {confidence:.0f}%)",
              fontsize=12)
ax1.set_xlabel("Samples")
ax1.set_ylabel("Amplitude (mV)")
ax1.legend(fontsize=9)
ax1.set_xlim(0, len(patient_ecg))

# ── Plot 2: Triage Result Badge ──────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
badge_color = '#e74c3c' if pred == 1 else '#27ae60'
ax2.add_patch(plt.Rectangle((0.1, 0.3), 0.8, 0.4,
              facecolor=badge_color, alpha=0.15, edgecolor=badge_color,
              linewidth=2, transform=ax2.transAxes))
ax2.text(0.5, 0.58, status, ha='center', va='center', fontsize=18,
         fontweight='bold', color=badge_color, transform=ax2.transAxes)
ax2.text(0.5, 0.44, action, ha='center', va='center', fontsize=9,
         color='#555', transform=ax2.transAxes)
ax2.text(0.5, 0.30, f"Confidence: {confidence:.1f}%", ha='center', va='center',
         fontsize=11, color=badge_color, transform=ax2.transAxes)
ax2.set_title("Triage Decision", fontsize=12)

# ── Plot 3: HRV Features ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']
bars = ax3.bar(FEATURE_NAMES, patient_features, color=colors, alpha=0.8)
ax3.set_title("HRV Features (new patient)", fontsize=11)
ax3.set_ylabel("Value")
ax3.tick_params(axis='x', rotation=20)
for bar, val in zip(bars, patient_features):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# ── Plot 4: Confusion Matrix ──────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
im = ax4.imshow(cm, cmap='Blues', aspect='auto')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Normal', 'Anomalous'])
ax4.set_yticklabels(['Normal', 'Anomalous'])
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_title("Confusion Matrix", fontsize=11)
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')

# ── Plot 5: Feature Importance ───────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
sorted_idx = np.argsort(importances)
ax5.barh(
    [FEATURE_NAMES[i] for i in sorted_idx],
    importances[sorted_idx],
    color='teal', alpha=0.8
)
ax5.set_title("Feature Importance", fontsize=11)
ax5.set_xlabel("Importance score")


plt.savefig("prana_ecg_results.png",
            dpi=150, bbox_inches='tight', facecolor='white')
print("  Plot saved as prana_ecg_results.png")
print("\n[Done] Pipeline complete.")