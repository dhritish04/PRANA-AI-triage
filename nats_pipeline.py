# =============================================================
#  PRANA PROJECT — nats_pipeline.py
#  Step 10: Real-Time NATS Messaging Pipeline
#
#  This is the architecture shown in your pitch deck:
#
#    Sensors (ECG/X-Ray)
#         ↓
#    NATS Publisher   ← publishes patient data as JSON
#         ↓
#    NATS Subscriber  ← receives, runs AI, saves to DB
#         ↓
#    SQLite Database  ← every report stored permanently
#
#  In your pitch deck this was:
#    Raspberry Pi → SQLite → NATS → Hailo AI → SQLite
#
#  This file simulates that full flow on your laptop.
#  When you move to Raspberry Pi, only the hardware
#  reading part changes — the NATS logic stays identical.
#
#  HOW TO RUN:
#    Step 1: Install NATS server (one time)
#            Windows : winget install nats-io.nats-server
#            OR download from https://nats.io/download
#
#    Step 2: Start NATS server in a terminal
#            nats-server
#
#    Step 3: Install Python client
#            pip install nats-py
#
#    Step 4: Run this file (it runs both publisher + subscriber)
#            python nats_pipeline.py
# =============================================================

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime
import neurokit2 as nk
import nats

# Import our existing modules
from prana_database import (
    load_ecg_model,
    extract_hrv_features,
    init_database,
    save_report,
    get_stats
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
NATS_URL        = "nats://localhost:4222"   # local NATS server
ECG_TOPIC       = "prana.ecg"              # topic for ECG data
XRAY_TOPIC      = "prana.xray"             # topic for X-ray data
REPORT_TOPIC    = "prana.report"           # topic for final reports

SAMPLING_RATE   = 360
WINDOW_SECONDS  = 10
ALERT_THRESHOLD = 0.3

# How many simulated patients to send
N_PATIENTS = 5


# =============================================================
# PART 1: PUBLISHER
# Simulates sensors sending patient data over NATS
# On real device: replace simulate_* with actual sensor reads
# =============================================================

def simulate_ecg_data(patient_id, scenario="random"):
    """
    Simulate ECG sensor reading for one patient.
    On Raspberry Pi this would read from the actual ECG sensor.

    Scenarios:
        normal      → healthy HR 60–90 BPM
        tachycardia → fast HR 110–160 BPM
        bradycardia → slow HR 30–50 BPM
        irregular   → normal rate but noisy signal
        random      → pick one randomly
    """
    if scenario == "random":
        scenario = np.random.choice(
            ["normal", "tachycardia", "bradycardia", "irregular"],
            p=[0.5, 0.2, 0.15, 0.15]   # realistic distribution
        )

    if scenario == "normal":
        ecg = nk.ecg_simulate(
            duration=WINDOW_SECONDS,
            sampling_rate=SAMPLING_RATE,
            heart_rate=np.random.uniform(60, 90),
            noise=0.05
        )
    elif scenario == "tachycardia":
        ecg = nk.ecg_simulate(
            duration=WINDOW_SECONDS,
            sampling_rate=SAMPLING_RATE,
            heart_rate=np.random.uniform(110, 160),
            noise=0.15
        )
    elif scenario == "bradycardia":
        ecg = nk.ecg_simulate(
            duration=WINDOW_SECONDS,
            sampling_rate=SAMPLING_RATE,
            heart_rate=np.random.uniform(30, 50),
            noise=0.10
        )
    else:   # irregular
        ecg = nk.ecg_simulate(
            duration=WINDOW_SECONDS,
            sampling_rate=SAMPLING_RATE,
            heart_rate=np.random.uniform(60, 90),
            noise=0.45
        )

    return {
        "patient_id" : patient_id,
        "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario"   : scenario,
        "ecg_signal" : ecg.tolist(),   # JSON-serializable list
        "sampling_rate": SAMPLING_RATE
    }


def simulate_xray_data(patient_id):
    """
    Simulate X-ray findings for one patient.
    On real device this would call the DenseNet model on actual image.

    We simulate realistic scores here to keep this file
    runnable without downloading the 100MB DenseNet model.
    To use real X-ray: replace this with analyse_xray() from
    prana_triage.py
    """
    # Randomly pick a scenario
    scenario = np.random.choice(
        ["clear", "mild", "serious"],
        p=[0.5, 0.3, 0.2]
    )

    if scenario == "clear":
        flagged  = {}
        status   = "NORMAL"
    elif scenario == "mild":
        flagged  = {
            "Atelectasis": round(np.random.uniform(0.30, 0.45), 2)
        }
        status   = "ATTENTION"
    else:
        flagged  = {
            "Pneumonia"   : round(np.random.uniform(0.55, 0.80), 2),
            "Lung Opacity": round(np.random.uniform(0.50, 0.75), 2),
            "Effusion"    : round(np.random.uniform(0.40, 0.65), 2)
        }
        status = "CRITICAL"

    return {
        "patient_id" : patient_id,
        "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status"     : status,
        "flagged"    : flagged
    }


async def publisher(nc, n_patients=N_PATIENTS):
    """
    NATS Publisher — simulates sensor data being sent.
    Publishes ECG + X-ray data for each patient.
    """
    print(f"\n  [Publisher] Sending data for {n_patients} patients...")

    for i in range(1, n_patients + 1):
        patient_id = f"PAT{i:03d}"

        # Publish ECG data
        ecg_data = simulate_ecg_data(patient_id)
        await nc.publish(
            ECG_TOPIC,
            json.dumps(ecg_data).encode()
        )
        print(f"  [Publisher] → {ECG_TOPIC}  |  "
              f"{patient_id}  |  scenario: {ecg_data['scenario']}")

        # Small delay between patients (simulates real-time arrival)
        await asyncio.sleep(0.3)

        # Publish X-ray data
        xray_data = simulate_xray_data(patient_id)
        await nc.publish(
            XRAY_TOPIC,
            json.dumps(xray_data).encode()
        )
        print(f"  [Publisher] → {XRAY_TOPIC} |  "
              f"{patient_id}  |  status: {xray_data['status']}")

        await asyncio.sleep(0.3)

    print(f"\n  [Publisher] All {n_patients} patients sent.")


# =============================================================
# PART 2: SUBSCRIBER (the AI processing side)
# Receives data from NATS, runs models, saves to DB
# =============================================================

def compute_overall_severity(ecg_status, xray_status):
    """
    Same logic as prana_triage.py — combine ECG + X-ray.
    Returns (severity, action) tuple.
    """
    ecg_bad   = ecg_status == "ANOMALOUS"
    xray_crit = xray_status == "CRITICAL"
    xray_att  = xray_status == "ATTENTION"

    if ecg_bad and xray_crit:
        return "CRITICAL", "Immediate referral required"
    elif ecg_bad or xray_crit:
        return "HIGH",     "Urgent specialist review needed"
    elif xray_att:
        return "MODERATE", "Monitor and follow up within 48 hours"
    else:
        return "NORMAL",   "No immediate action required"


class PranaSubscriber:
    """
    Stateful subscriber — holds the ECG model and a buffer
    that matches ECG and X-ray messages by patient_id before
    processing them together.
    """

    def __init__(self, ecg_model):
        self.ecg_model    = ecg_model
        self.ecg_buffer   = {}   # patient_id → ecg_data
        self.xray_buffer  = {}   # patient_id → xray_data
        self.reports_done = 0

    def process_patient(self, patient_id):
        """
        Called when BOTH ECG and X-ray data are available
        for the same patient. Runs AI, saves to DB.
        """
        ecg_raw  = self.ecg_buffer.pop(patient_id)
        xray_raw = self.xray_buffer.pop(patient_id)

        # ── ECG Analysis ──────────────────────────────────
        ecg_signal = np.array(ecg_raw["ecg_signal"])
        features   = extract_hrv_features(ecg_signal, SAMPLING_RATE)

        if features is None:
            print(f"  [Subscriber] ⚠ ECG feature extraction failed for {patient_id}")
            return
        print("\nDEBUG FEATURES:")
        print(features)
        print("Length =", len(features))

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

        pred = self.ecg_model.predict(sample)[0]
        proba = self.ecg_model.predict_proba(sample)[0]
        # pred       = self.ecg_model.predict([features])[0]
        # proba      = self.ecg_model.predict_proba([features])[0]
        confidence = max(proba) * 100

        ecg_result = {
            "mean_hr"    : features[0],
            "mean_rr"    : features[1] * 1000,
            "sdnn"       : features[2] * 1000,
            "rmssd"      : features[3] * 1000,
            "pnn50"      : features[4],
            "status"     : "ANOMALOUS" if pred == 1 else "NORMAL",
            "confidence" : confidence,
            "features"   : features,
        }

        # ── X-ray Analysis (already done by publisher side) ──
        xray_result = {
            "status"  : xray_raw["status"],
            "flagged" : xray_raw["flagged"],
        }

        # ── Combined Severity ─────────────────────────────
        severity, action = compute_overall_severity(
            ecg_result["status"],
            xray_result["status"]
        )

        # ── Save to SQLite ────────────────────────────────
        save_report(
            patient_id  = patient_id,
            ecg_result  = ecg_result,
            xray_result = xray_result,
            severity    = severity,
            action      = action
        )

        # ── Print triage summary ─────────────────────────
        sev_icon = {
            "CRITICAL": "⛔", "HIGH": "🔴",
            "MODERATE": "🟡", "NORMAL": "🟢"
        }.get(severity, "⬜")

        print(f"\n  {sev_icon} TRIAGE COMPLETE — {patient_id}")
        print(f"     ECG    : {ecg_result['status']:<10s}  "
              f"HR={ecg_result['mean_hr']:.0f} BPM  "
              f"Confidence={confidence:.0f}%")
        print(f"     X-Ray  : {xray_result['status']:<10s}  "
              f"Flags: {', '.join(xray_result['flagged'].keys()) or 'None'}")
        print(f"     Overall: {severity}  →  {action}")

        self.reports_done += 1

    async def on_ecg(self, msg):
        """Handler called when ECG message arrives on NATS."""
        data       = json.loads(msg.data.decode())
        patient_id = data["patient_id"]
        print(f"  [Subscriber] ← ECG received    {patient_id}")

        self.ecg_buffer[patient_id] = data

        # If X-ray already arrived for this patient, process now
        if patient_id in self.xray_buffer:
            self.process_patient(patient_id)

    async def on_xray(self, msg):
        """Handler called when X-ray message arrives on NATS."""
        data       = json.loads(msg.data.decode())
        patient_id = data["patient_id"]
        print(f"  [Subscriber] ← X-Ray received  {patient_id}")

        self.xray_buffer[patient_id] = data

        # If ECG already arrived for this patient, process now
        if patient_id in self.ecg_buffer:
            self.process_patient(patient_id)


# =============================================================
# PART 3: MAIN — Run publisher + subscriber together
# =============================================================

async def main():
    print("=" * 60)
    print("  PRANA — NATS Real-Time Pipeline")
    print("  Sensors → NATS → AI → Database")
    print("=" * 60)

    # ── Load ECG model (from prana_database.py) ───────────
    print("\n[Step 1] Loading ECG model...")
    ecg_model = load_ecg_model()

    # ── Setup database ─────────────────────────────────────
    print("\n[Step 2] Setting up database...")
    init_database()

    # ── Connect to NATS ────────────────────────────────────
    print(f"\n[Step 3] Connecting to NATS at {NATS_URL}...")
    try:
        nc = await nats.connect(
            NATS_URL,
            max_reconnect_attempts=0,   # fail immediately, no retries
            connect_timeout=2           # give up after 2 seconds
        )
        print(f"  Connected!")
    except Exception as e:
        print(f"\n  ✗ Could not connect to NATS: {type(e).__name__}")
        print("\n  ── To fix this ──────────────────────────────────────")
        print("  1. Download NATS server: https://nats.io/download")
        print("  2. Run in a separate terminal: nats-server")
        print("  3. Then run this script again")
        print("  ─────────────────────────────────────────────────────")
        print("\n  Running in OFFLINE SIMULATION MODE instead...")
        await run_offline_simulation(ecg_model)
        return

    # ── Setup subscriber ───────────────────────────────────
    print("\n[Step 4] Starting subscriber (AI processing side)...")
    subscriber = PranaSubscriber(ecg_model)

    await nc.subscribe(ECG_TOPIC,  cb=subscriber.on_ecg)
    await nc.subscribe(XRAY_TOPIC, cb=subscriber.on_xray)
    print(f"  Listening on: {ECG_TOPIC}")
    print(f"  Listening on: {XRAY_TOPIC}")

    # ── Run publisher ──────────────────────────────────────
    print("\n[Step 5] Starting publisher (sensor simulation side)...")
    await publisher(nc, n_patients=N_PATIENTS)

    # Wait for all messages to be processed
    await asyncio.sleep(2)

    # ── Print final stats ──────────────────────────────────
    print(f"\n[Step 6] Pipeline complete — {subscriber.reports_done} reports processed")
    print("\n  Clinic Summary:")
    stats = get_stats()
    total = sum(stats.values())
    for level, count in sorted(stats.items()):
        icon = {"CRITICAL":"⛔","HIGH":"🔴","MODERATE":"🟡","NORMAL":"🟢"}.get(level,"⬜")
        bar  = "█" * count
        print(f"    {icon} {level:<10}: {bar}  ({count}/{total})")

    await nc.close()
    print("\n[Done] NATS pipeline complete.")


async def run_offline_simulation(ecg_model):
    """
    Runs the same pipeline without NATS.
    Used when NATS server is not running.
    Useful for testing the AI + DB logic in isolation.
    """
    print("\n" + "=" * 60)
    print("  OFFLINE SIMULATION MODE")
    print("  (Same logic, no NATS server needed)")
    print("=" * 60)

    init_database()
    subscriber = PranaSubscriber(ecg_model)

    print(f"\n  Simulating {N_PATIENTS} patients...\n")

    for i in range(1, N_PATIENTS + 1):
        patient_id = f"SIM{i:03d}"

        # Simulate ECG message
        ecg_msg_data  = simulate_ecg_data(patient_id)
        xray_msg_data = simulate_xray_data(patient_id)

        # Process directly (bypass NATS)
        subscriber.ecg_buffer[patient_id]  = ecg_msg_data
        subscriber.xray_buffer[patient_id] = xray_msg_data
        subscriber.process_patient(patient_id)

        await asyncio.sleep(0.1)

    print(f"\n  {subscriber.reports_done} reports saved to database.")
    print("\n  Clinic Summary:")
    stats = get_stats()
    total = sum(stats.values())
    for level, count in sorted(stats.items()):
        icon = {"CRITICAL":"⛔","HIGH":"🔴","MODERATE":"🟡","NORMAL":"🟢"}.get(level,"⬜")
        bar  = "█" * count
        print(f"    {icon} {level:<10}: {bar}  ({count}/{total})")

    print("\n[Done] Offline simulation complete.")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())