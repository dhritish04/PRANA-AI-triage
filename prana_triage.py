# =============================================================
#  PRANA PROJECT — prana_triage.py
#  Step 8: Combined ECG + X-Ray Triage System
#
#  This is the actual PRANA product.
#  Give it an ECG signal + a chest X-ray,
#  it gives back ONE unified clinical triage report.
#
#  Combines:
#    ecg_anomaly_pipeline.py  → ECG analysis
#    xray_module.py           → X-ray analysis
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
PATIENT_ID       = "PAT001"
SAMPLING_RATE    = 360
WINDOW_SECONDS   = 10
ALERT_THRESHOLD  = 0.3
N_TRAIN_SAMPLES  = 80   # synthetic ECG samples for training


# =============================================================
# MODULE 1: ECG
# (from ecg_anomaly_pipeline.py)
# =============================================================

def extract_hrv_features(ecg_signal, sampling_rate=360):
    try:
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        r_peaks      = info["ECG_R_Peaks"]
        if len(r_peaks) < 3:
            return None
        rr_intervals = np.diff(r_peaks)
        rr_seconds   = rr_intervals / sampling_rate
        heart_rates  = 60 / rr_seconds
        mean_hr  = np.mean(heart_rates)
        mean_rr  = np.mean(rr_seconds)
        sdnn     = np.std(rr_seconds)
        rmssd    = np.sqrt(np.mean(np.diff(rr_seconds) ** 2))
        rr_diff  = np.abs(np.diff(rr_seconds))
        pnn50    = (np.sum(rr_diff > 0.05) / len(rr_diff)) * 100
        return [mean_hr, mean_rr, sdnn, rmssd, pnn50]
    except Exception:
        return None


def train_ecg_model():
    """Train the ECG anomaly detector on synthetic data."""
    X, y = [], []
    for _ in range(N_TRAIN_SAMPLES):
        ecg = nk.ecg_simulate(duration=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE,
                               heart_rate=np.random.uniform(60, 90), noise=0.05)
        f = extract_hrv_features(ecg, SAMPLING_RATE)
        if f:
            X.append(f); y.append(0)

    for _ in range(N_TRAIN_SAMPLES):
        anomaly = np.random.choice(['tachycardia', 'bradycardia', 'irregular'])
        if anomaly == 'tachycardia':
            ecg = nk.ecg_simulate(duration=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE,
                                   heart_rate=np.random.uniform(110, 160), noise=0.15)
        elif anomaly == 'bradycardia':
            ecg = nk.ecg_simulate(duration=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE,
                                   heart_rate=np.random.uniform(30, 50), noise=0.10)
        else:
            ecg = nk.ecg_simulate(duration=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE,
                                   heart_rate=np.random.uniform(60, 90), noise=0.45)
        f = extract_hrv_features(ecg, SAMPLING_RATE)
        if f:
            X.append(f); y.append(1)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model


def analyse_ecg(ecg_model, patient_ecg):
    """Run ECG through trained model, return results dict."""
    features = extract_hrv_features(patient_ecg, SAMPLING_RATE)
    if features is None:
        return None

    pred       = ecg_model.predict([features])[0]
    proba      = ecg_model.predict_proba([features])[0]
    confidence = max(proba) * 100

    return {
        "features"   : features,
        "status"     : "ANOMALOUS" if pred == 1 else "NORMAL",
        "confidence" : confidence,
        "mean_hr"    : features[0],
        "mean_rr"    : features[1] * 1000,   # convert to ms
        "sdnn"       : features[2] * 1000,
        "rmssd"      : features[3] * 1000,
        "pnn50"      : features[4],
    }


# =============================================================
# MODULE 2: X-RAY
# (from xray_module.py)
# =============================================================

def create_demo_xray():
    np.random.seed(42)
    img = np.zeros((224, 224), dtype=np.float32)
    for y in range(224):
        for x in range(224):
            if ((x-75)**2/40**2 + (y-112)**2/70**2) < 1:
                img[y,x] = 0.3 + np.random.normal(0, 0.05)
            elif ((x-149)**2/40**2 + (y-112)**2/70**2) < 1:
                img[y,x] = 0.3 + np.random.normal(0, 0.05)
            elif ((x-112)**2/20**2 + (y-120)**2/30**2) < 1:
                img[y,x] = 0.7 + np.random.normal(0, 0.03)
            else:
                img[y,x] = 0.15 + np.random.normal(0, 0.02)
    return np.clip(img, 0, 1)


def load_xray_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def analyse_xray(xray_model, use_demo=True, image_path=None):
    """Run X-ray through DenseNet model, return results dict."""
    if use_demo:
        img = create_demo_xray()
    else:
        import skimage.io
        img = skimage.io.imread(image_path)
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        img = img.astype(np.float32)

    img = xrv.datasets.normalize(img, img.max())
    img_array = img[None, :]

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img_array = transform(img_array)
    tensor = torch.from_numpy(img_array).unsqueeze(0)

    with torch.no_grad():
        preds = xray_model(tensor).cpu().numpy()[0]

    results = dict(zip(xray_model.pathologies, preds))
    flagged = {k: v for k, v in results.items() if v >= ALERT_THRESHOLD}
    critical = {k: v for k, v in flagged.items() if v >= 0.5}

    xray_status = "CRITICAL" if critical else \
                  "ATTENTION" if flagged else "NORMAL"

    return {
        "img_array"    : img_array,
        "results"      : results,
        "flagged"      : flagged,
        "critical"     : critical,
        "status"       : xray_status,
    }


# =============================================================
# MODULE 3: COMBINED TRIAGE REPORT
# =============================================================

def overall_severity(ecg_result, xray_result):
    """
    Combine ECG + X-ray findings into one severity level.

    Logic:
      CRITICAL  → ECG anomalous AND X-ray critical
      HIGH      → Either ECG anomalous OR X-ray critical
      MODERATE  → ECG normal but X-ray has attention findings
      NORMAL    → Both clear
    """
    ecg_bad   = ecg_result["status"] == "ANOMALOUS"
    xray_crit = xray_result["status"] == "CRITICAL"
    xray_att  = xray_result["status"] == "ATTENTION"

    if ecg_bad and xray_crit:
        return "CRITICAL", "Immediate referral required"
    elif ecg_bad or xray_crit:
        return "HIGH", "Urgent specialist review needed"
    elif xray_att:
        return "MODERATE", "Monitor and follow up within 48 hours"
    else:
        return "NORMAL", "No immediate action required"


def print_combined_report(ecg_result, xray_result, patient_id, timestamp):
    """Print the full unified PRANA triage report to terminal."""
    severity, action = overall_severity(ecg_result, xray_result)

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║           PRANA UNIFIED TRIAGE REPORT               ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Patient   : {patient_id:<41s}║")
    print(f"  ║  Timestamp : {timestamp:<41s}║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print("  ║  ECG FINDINGS                                        ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Mean Heart Rate   : {ecg_result['mean_hr']:>6.1f} BPM                    ║")
    print(f"  ║  Mean RR Interval  : {ecg_result['mean_rr']:>6.0f} ms                     ║")
    print(f"  ║  SDNN              : {ecg_result['sdnn']:>6.1f} ms                     ║")
    print(f"  ║  RMSSD             : {ecg_result['rmssd']:>6.1f} ms                     ║")
    print(f"  ║  pNN50             : {ecg_result['pnn50']:>6.1f} %                      ║")
    print(f"  ║  ECG Status        : {ecg_result['status']:<33s}║")
    print(f"  ║  ECG Confidence    : {ecg_result['confidence']:>5.1f}%                         ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print("  ║  X-RAY FINDINGS                                      ║")
    print("  ╠══════════════════════════════════════════════════════╣")

    # Top 5 X-ray findings
    top5 = sorted(
        xray_result["results"].items(),
        key=lambda x: x[1], reverse=True
    )[:5]
    for condition, score in top5:
        flag = " ⚠" if score >= 0.5 else " △" if score >= ALERT_THRESHOLD else "  "
        print(f"  ║  {condition:<26s}: {score:.2f}{flag:<26s}║")

    print(f"  ║  X-Ray Status      : {xray_result['status']:<33s}║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  OVERALL SEVERITY  : {severity:<33s}║")
    print(f"  ║  ACTION            : {action:<33s}║")
    print("  ╚══════════════════════════════════════════════════════╝")

    return severity, action


def save_combined_report(ecg_result, xray_result, severity, action, patient_id):
    """Save a visual combined report as PNG."""

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"PRANA Unified Triage Report — {patient_id}  |  Severity: {severity}",
        fontsize=15, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── 1: X-ray image ───────────────────────────────────
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(xray_result["img_array"][0], cmap='gray')
    ax1.set_title("Chest X-Ray", fontsize=12)
    ax1.axis('off')
    colors = {'CRITICAL': '#e74c3c', 'HIGH': '#e74c3c',
               'MODERATE': '#e67e22', 'NORMAL': '#27ae60'}
    badge_color = colors.get(severity, '#27ae60')
    ax1.text(0.5, 0.04, f"  {severity}  ",
             transform=ax1.transAxes, ha='center', fontsize=12,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor=badge_color, alpha=0.9))

    # ── 2: ECG signal ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    patient_ecg_plot = nk.ecg_simulate(
        duration=WINDOW_SECONDS, sampling_rate=SAMPLING_RATE,
        heart_rate=ecg_result["mean_hr"], noise=0.05
    )
    ax2.plot(patient_ecg_plot, color='royalblue', linewidth=0.8)
    ecg_color = '#e74c3c' if ecg_result['status'] == 'ANOMALOUS' else '#27ae60'
    ax2.set_title(
        f"ECG — {ecg_result['status']} ({ecg_result['confidence']:.0f}% confidence)",
        fontsize=11, color=ecg_color
    )
    ax2.set_xlabel("Samples"); ax2.set_ylabel("mV")

    # ── 3: HRV features ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    feat_names = ["Mean_HR", "Mean_RR", "SDNN", "RMSSD", "pNN50"]
    feat_vals  = ecg_result["features"]
    ax3.bar(feat_names, feat_vals,
            color=['#3498db','#2ecc71','#e67e22','#e74c3c','#9b59b6'],
            alpha=0.8)
    ax3.set_title("ECG HRV Features", fontsize=11)
    ax3.tick_params(axis='x', rotation=20)

    # ── 4: X-ray condition scores ─────────────────────────
    ax4 = fig.add_subplot(gs[:, 2])
    conditions = list(xray_result["results"].keys())
    scores     = list(xray_result["results"].values())
    bar_colors = ['#e74c3c' if s>=0.5 else
                  '#e67e22' if s>=ALERT_THRESHOLD else '#2ecc71'
                  for s in scores]
    ax4.barh(conditions, scores, color=bar_colors, alpha=0.85)
    ax4.axvline(x=ALERT_THRESHOLD, color='orange', linestyle='--',
                linewidth=1, label=f'Alert ({ALERT_THRESHOLD})')
    ax4.axvline(x=0.5, color='red', linestyle='--',
                linewidth=1, label='Critical (0.5)')
    ax4.set_xlim(0, 1.0)
    ax4.set_xlabel("Probability Score", fontsize=10)
    ax4.set_title("X-Ray: Condition Scores", fontsize=11)
    ax4.legend(fontsize=8)

    # ── Action box at bottom ──────────────────────────────
    fig.text(0.5, 0.01,
             f"Recommended Action: {action}",
             ha='center', fontsize=12, fontweight='bold',
             color=badge_color)

    plt.savefig("prana_combined_report.png",
                dpi=150, bbox_inches='tight', facecolor='white')
    print("\n  Report saved as prana_combined_report.png")


# =============================================================
# MAIN — Run the full PRANA triage pipeline
# =============================================================

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("=" * 60)
print("  PRANA — Unified Triage System")
print("  ECG + Chest X-Ray → Combined Clinical Report")
print("=" * 60)

# ── Step 1: Train ECG model ───────────────────────────────
print("\n[Step 1] Training ECG anomaly detector...")
ecg_model = train_ecg_model()
print("  ECG model ready!")

# ── Step 2: Simulate patient ECG (tachycardia) ───────────
print("\n[Step 2] Analysing patient ECG...")
patient_ecg = nk.ecg_simulate(
    duration=WINDOW_SECONDS,
    sampling_rate=SAMPLING_RATE,
    heart_rate=130,       # tachycardia
    noise=0.1
)
ecg_result = analyse_ecg(ecg_model, patient_ecg)
print(f"  ECG Status     : {ecg_result['status']}")
print(f"  Heart Rate     : {ecg_result['mean_hr']:.1f} BPM")
print(f"  Confidence     : {ecg_result['confidence']:.1f}%")

# ── Step 3: Load X-ray model ──────────────────────────────
print("\n[Step 3] Loading X-ray model...")
xray_model = load_xray_model()
print("  X-ray model ready!")

# ── Step 4: Analyse X-ray ─────────────────────────────────
print("\n[Step 4] Analysing chest X-ray...")
xray_result = analyse_xray(xray_model, use_demo=True)
print(f"  X-Ray Status   : {xray_result['status']}")
print(f"  Flagged        : {len(xray_result['flagged'])} conditions")

# ── Step 5: Print unified report ──────────────────────────
print("\n[Step 5] Generating unified triage report...")
severity, action = print_combined_report(
    ecg_result, xray_result, PATIENT_ID, timestamp
)

# ── Step 6: Save visual report ────────────────────────────
print("\n[Step 6] Saving visual report...")
save_combined_report(
    ecg_result, xray_result, severity, action, PATIENT_ID
)

print("\n[Done] PRANA triage complete.")
print(f"  Overall Severity : {severity}")
print(f"  Action           : {action}")