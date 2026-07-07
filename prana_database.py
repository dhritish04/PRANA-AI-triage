# =============================================================
#  PRANA PROJECT — prana_database.py
#  Step 9: Model Persistence + SQLite Patient Database
#
#  Two problems this solves:
#    PROBLEM 1: prana_triage.py retrains the ECG model from
#               scratch every single run. That takes time and
#               wastes CPU — bad for an edge device.
#
#    PROBLEM 2: Every triage report is printed and forgotten.
#               A real medical device must store every report
#               so doctors can look up any patient later.
#
#  What this file adds:
#    1. save_ecg_model()   — train once, save to disk as .pkl
#    2. load_ecg_model()   — load instantly on next run
#    3. init_database()    — create SQLite DB (prana_records.db)
#    4. save_report()      — insert one triage report into DB
#    5. get_patient()      — look up all reports for a patient
#    6. get_all_reports()  — show full patient history table
#    7. get_stats()        — count normals vs anomalies seen
#
#  After running this once:
#    ecg_model.pkl        — your trained model (never retrain)
#    prana_records.db     — your SQLite patient database
# =============================================================

import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "ecg_model.pkl"
DB_PATH    = "prana_records.db"

# ECG config (must match prana_triage.py)
SAMPLING_RATE   = 360
WINDOW_SECONDS  = 10
N_TRAIN_SAMPLES = 80


# =============================================================
# PART 1: MODEL PERSISTENCE
# Train once → save → load instantly every future run
# =============================================================

def extract_hrv_features(ecg_signal, sampling_rate=360):
    """Extract 5 HRV features from one ECG window."""
    try:
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]
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


# def train_and_save_ecg_model(model_path=MODEL_PATH):
#     """
#     Train the RandomForest ECG model on synthetic data,
#     then save it to disk using joblib.

#     After this runs once, you never need to retrain.
#     Next run: just call load_ecg_model() — instant load.
#     """
#     print("  Training ECG anomaly model on synthetic data...")

#     X, y = [], []

#     # Normal class: HR 60–90 BPM, low noise
#     for _ in range(N_TRAIN_SAMPLES):
#         ecg = nk.ecg_simulate(
#             duration=WINDOW_SECONDS,
#             sampling_rate=SAMPLING_RATE,
#             heart_rate=np.random.uniform(60, 90),
#             noise=0.05
#         )
#         f = extract_hrv_features(ecg, SAMPLING_RATE)
#         if f:
#             X.append(f)
#             y.append(0)

#     # Anomalous class: tachycardia / bradycardia / irregular
#     for _ in range(N_TRAIN_SAMPLES):
#         anomaly = np.random.choice(['tachycardia', 'bradycardia', 'irregular'])
#         if anomaly == 'tachycardia':
#             ecg = nk.ecg_simulate(
#                 duration=WINDOW_SECONDS,
#                 sampling_rate=SAMPLING_RATE,
#                 heart_rate=np.random.uniform(110, 160),
#                 noise=0.15
#             )
#         elif anomaly == 'bradycardia':
#             ecg = nk.ecg_simulate(
#                 duration=WINDOW_SECONDS,
#                 sampling_rate=SAMPLING_RATE,
#                 heart_rate=np.random.uniform(30, 50),
#                 noise=0.10
#             )
#         else:
#             ecg = nk.ecg_simulate(
#                 duration=WINDOW_SECONDS,
#                 sampling_rate=SAMPLING_RATE,
#                 heart_rate=np.random.uniform(60, 90),
#                 noise=0.45
#             )
#         f = extract_hrv_features(ecg, SAMPLING_RATE)
#         if f:
#             X.append(f)
#             y.append(1)

#     model = RandomForestClassifier(
#         n_estimators=100,
#         max_depth=10,
#         random_state=42
#     )
#     model.fit(X, y)

#     # ── Save to disk ──────────────────────────────────────
#     joblib.dump(model, model_path)
#     print(f"  Model saved → {model_path}")
#     print(f"  Trained on {len(y)} windows  "
#           f"({y.count(0)} normal, {y.count(1)} anomalous)")
#     return model

def train_and_save_ecg_model(model_path=MODEL_PATH):

    print("  Loading real ECG dataset...")

    df = pd.read_csv(
        "real_labeled_ecg_dataset.csv"
    )

    X = df.drop(
        "Label",
        axis=1
    )

    y = df["Label"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(
        model,
        model_path
    )

    print(f"  Model saved → {model_path}")
    print(f"  Trained on {len(df)} real ECG windows")

    return model


def load_ecg_model(model_path=MODEL_PATH):
    """
    Load saved ECG model from disk.
    If model file doesn't exist yet, train and save it first.

    This is what prana_triage.py should call instead of
    train_ecg_model() — instant load, no retraining.
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"  Model loaded from {model_path}  (no retraining needed)")
        return model
    else:
        print(f"  No saved model found at {model_path}")
        print("  Training from scratch and saving...")
        return train_and_save_ecg_model(model_path)


# =============================================================
# PART 2: SQLITE PATIENT DATABASE
# Every triage report → one row in the database
# =============================================================

def init_database(db_path=DB_PATH):
    """
    Create the SQLite database and the triage_reports table.
    Safe to call multiple times — only creates table if missing.

    Table columns:
        id            — auto-incrementing row number
        patient_id    — e.g. "PAT001"
        timestamp     — "2026-06-10 14:23:01"
        mean_hr       — heart rate in BPM
        mean_rr       — mean RR interval in ms
        sdnn          — SDNN in ms
        rmssd         — RMSSD in ms
        pnn50         — pNN50 %
        ecg_status    — "NORMAL" or "ANOMALOUS"
        ecg_confidence— e.g. 94.2
        xray_status   — "NORMAL", "ATTENTION", or "CRITICAL"
        xray_flagged  — comma-separated flagged conditions
        overall       — "NORMAL", "MODERATE", "HIGH", "CRITICAL"
        action        — recommended clinical action text
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS triage_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL,
            mean_hr         REAL,
            mean_rr         REAL,
            sdnn            REAL,
            rmssd           REAL,
            pnn50           REAL,
            ecg_status      TEXT,
            ecg_confidence  REAL,
            xray_status     TEXT,
            xray_flagged    TEXT,
            overall         TEXT,
            action          TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"  Database ready → {db_path}")


def save_report(
    patient_id,
    ecg_result,
    xray_result,
    severity,
    action,
    db_path=DB_PATH
):
    """
    Insert one complete triage report into the database.

    Parameters:
        patient_id   — string like "PAT001"
        ecg_result   — dict returned by analyse_ecg()
        xray_result  — dict returned by analyse_xray()
        severity     — "NORMAL" / "MODERATE" / "HIGH" / "CRITICAL"
        action       — recommended action string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Flatten flagged X-ray conditions into a comma-separated string
    flagged_str = ", ".join(
        f"{k}({v:.2f})"
        for k, v in sorted(
            xray_result["flagged"].items(),
            key=lambda x: x[1],
            reverse=True
        )
    ) if xray_result["flagged"] else "None"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO triage_reports (
            patient_id, timestamp,
            mean_hr, mean_rr, sdnn, rmssd, pnn50,
            ecg_status, ecg_confidence,
            xray_status, xray_flagged,
            overall, action
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        timestamp,
        round(ecg_result["mean_hr"], 2),
        round(ecg_result["mean_rr"], 2),
        round(ecg_result["sdnn"], 2),
        round(ecg_result["rmssd"], 2),
        round(ecg_result["pnn50"], 2),
        ecg_result["status"],
        round(ecg_result["confidence"], 2),
        xray_result["status"],
        flagged_str,
        severity,
        action
    ))

    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    
    os.makedirs(
        "reports",
        exist_ok=True
    )
    
    report_path = f"reports/{patient_id}_report.txt"
    
    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:
    
        report.write("====================================\n")
        report.write("PRANA AI TRIAGE REPORT\n")
        report.write("====================================\n\n")
    
        report.write(f"Patient ID: {patient_id}\n")
        report.write(f"Timestamp: {timestamp}\n\n")
    
        report.write("ECG ANALYSIS\n")
        report.write("------------\n")
        report.write(f"Status: {ecg_result['status']}\n")
        report.write(
            f"Confidence: {ecg_result['confidence']:.2f}%\n\n"
        )
    
        report.write("X-RAY ANALYSIS\n")
        report.write("--------------\n")
        report.write(f"Status: {xray_result['status']}\n")
        report.write(f"Findings: {flagged_str}\n\n")
    
        report.write("OVERALL TRIAGE\n")
        report.write("--------------\n")
        report.write(f"Severity: {severity}\n")
        report.write(f"Action: {action}\n")
    
    print(
        f"  Report file saved → {report_path}"
    )
    
    print(
        f"  Report saved → DB row #{row_id} "
        f"(Patient: {patient_id}, Severity: {severity})"
    )
    
    return row_id


def get_patient_history(patient_id, db_path=DB_PATH):
    """
    Retrieve all triage reports for a specific patient.
    Returns a list of dicts, newest first.
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM triage_reports
        WHERE patient_id = ?
        ORDER BY timestamp DESC
    """, (patient_id,))

    rows    = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    return [dict(zip(columns, row)) for row in rows]


def get_all_reports(db_path=DB_PATH, limit=20):
    """
    Show the most recent triage reports across all patients.
    Useful for a daily summary view.
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, patient_id, timestamp,
               ecg_status, xray_status, overall, action
        FROM triage_reports
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows    = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    return [dict(zip(columns, row)) for row in rows]


def get_stats(db_path=DB_PATH):
    """
    Count how many normals vs anomalies have been seen.
    Useful for a quick clinic summary.
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT overall, COUNT(*) as count
        FROM triage_reports
        GROUP BY overall
    """)

    stats = dict(cursor.fetchall())
    conn.close()
    return stats


def print_patient_report(patient_id, db_path=DB_PATH):
    """Pretty-print all records for a patient."""
    records = get_patient_history(patient_id, db_path)

    if not records:
        print(f"  No records found for patient: {patient_id}")
        return

    print(f"\n  Patient History: {patient_id}  ({len(records)} records)")
    print("  " + "─" * 72)

    for r in records:
        sev_color = {
            "CRITICAL": "⛔", "HIGH": "🔴",
            "MODERATE": "🟡", "NORMAL": "🟢"
        }.get(r["overall"], "⬜")

        print(f"  {sev_color}  [{r['timestamp']}]  "
              f"ECG: {r['ecg_status']:<9s}  "
              f"X-Ray: {r['xray_status']:<9s}  "
              f"Overall: {r['overall']}")
        print(f"      HR: {r['mean_hr']:.1f} BPM  |  "
              f"SDNN: {r['sdnn']:.1f} ms  |  "
              f"X-Ray flags: {r['xray_flagged']}")
        print(f"      Action: {r['action']}")
        print()


# =============================================================
# DEMO — Run everything end-to-end
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  PRANA — Database & Model Persistence Module")
    print("=" * 60)

    print("\n[Step 1] ECG Model Persistence")
    print("  " + "─" * 40)
    ecg_model = load_ecg_model()

    # Quick test prediction
    test_ecg = nk.ecg_simulate(
        duration=WINDOW_SECONDS,
        sampling_rate=SAMPLING_RATE,
        heart_rate=130,
        noise=0.1
    )

    features = extract_hrv_features(
        test_ecg,
        SAMPLING_RATE
    )

    if features:
        sample = pd.DataFrame(
         [features],
         columns=[
            "Mean_HR",
            "Mean_RR",
            "SDNN",
            "RMSSD",
            "pNN50"
          ]
        )

        pred = ecg_model.predict(sample)[0]
        proba = ecg_model.predict_proba(sample)[0]

        print(f"\n  Test prediction (HR=130 BPM tachycardia):")
        print(f"    Result     : {'ANOMALOUS' if pred == 1 else 'NORMAL'}")
        print(f"    Confidence : {max(proba)*100:.1f}%")


    # ── PART 2: Database demo ─────────────────────────────────

    print("\n[Step 2] SQLite Database Setup")
    print("  " + "─" * 40)
    init_database()

    # Simulate 3 patient records with different severities
    print("\n[Step 3] Inserting Demo Patient Records")
    print("  " + "─" * 40)

    demo_patients = [
        {
            "patient_id": "PAT001",
            "ecg_result": {
                "mean_hr": 130.0, "mean_rr": 461.0,
                "sdnn": 18.2, "rmssd": 22.1, "pnn50": 5.3,
                "status": "ANOMALOUS", "confidence": 94.0
            },
            "xray_result": {
                "status": "CRITICAL",
                "flagged": {"Pneumonia": 0.71, "Lung Opacity": 0.63, "Effusion": 0.52}
            },
            "severity": "CRITICAL",
            "action": "Immediate referral required"
        },
        {
            "patient_id": "PAT002",
            "ecg_result": {
                "mean_hr": 72.0, "mean_rr": 833.0,
                "sdnn": 42.5, "rmssd": 38.0, "pnn50": 18.2,
                "status": "NORMAL", "confidence": 88.0
            },
            "xray_result": {
                "status": "ATTENTION",
                "flagged": {"Atelectasis": 0.34}
            },
            "severity": "MODERATE",
            "action": "Monitor and follow up within 48 hours"
        },
        {
            "patient_id": "PAT003",
            "ecg_result": {
                "mean_hr": 68.0, "mean_rr": 882.0,
                "sdnn": 55.1, "rmssd": 47.3, "pnn50": 24.7,
                "status": "NORMAL", "confidence": 91.0
            },
            "xray_result": {
                "status": "NORMAL",
                "flagged": {}
            },
            "severity": "NORMAL",
            "action": "No immediate action required"
        },
    ]

    for p in demo_patients:
        save_report(
            patient_id  = p["patient_id"],
            ecg_result  = p["ecg_result"],
            xray_result = p["xray_result"],
            severity    = p["severity"],
            action      = p["action"]
        )

    # ── PART 3: Query the database ────────────────────────────

    print("\n[Step 4] Querying the Database")
    print("  " + "─" * 40)

    # Show PAT001's history
    print_patient_report("PAT001")

    # Show all recent reports
    print("  All recent reports:")
    all_reports = get_all_reports()
    print(f"  {'ID':<4} {'Patient':<8} {'Timestamp':<20} "
          f"{'ECG':<10} {'X-Ray':<10} {'Overall'}")
    print("  " + "─" * 70)
    for r in all_reports:
        print(f"  {r['id']:<4} {r['patient_id']:<8} {r['timestamp']:<20} "
              f"{r['ecg_status']:<10} {r['xray_status']:<10} {r['overall']}")

    # Show clinic summary stats
    print("\n  Clinic Summary Statistics:")
    stats = get_stats()
    total = sum(stats.values())
    for severity_level, count in sorted(stats.items()):
        bar = "█" * count
        print(f"    {severity_level:<10}: {bar}  ({count}/{total})")

    print("\n" + "=" * 60)
    print("  Files created:")
    print(f"    {MODEL_PATH:30s} — trained ECG model")
    print(f"    {DB_PATH:30s} — patient triage database")
    print("=" * 60)
    print("\n[Done] prana_database.py complete.")
    print("\nNext step: nats_pipeline.py")
    print("  That connects everything to NATS messaging so")
    print("  sensors can publish data and PRANA subscribes,")
    print("  processes, and saves — just like your pitch deck shows.")