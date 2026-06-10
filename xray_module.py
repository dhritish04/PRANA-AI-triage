# =============================================================
#  PRANA PROJECT — xray_module.py
#  Step 7: Chest X-Ray Analysis using TorchXRayVision
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms

USE_DEMO_IMAGE   = True
IMAGE_PATH       = "xray.png"
ALERT_THRESHOLD  = 0.3

ALL_CONDITIONS = [
    'Atelectasis','Consolidation','Infiltration','Pneumothorax',
    'Edema','Emphysema','Fibrosis','Effusion','Pneumonia',
    'Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia',
    'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum'
]

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

def load_and_preprocess(image_path=None, use_demo=True):
    if use_demo:
        print("  Using synthetic demo X-ray")
        img = create_demo_xray()
    else:
        import skimage.io
        print(f"  Loading: {image_path}")
        img = skimage.io.imread(image_path)
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        img = img.astype(np.float32)

    img = xrv.datasets.normalize(img, img.max())
    img = img[None, :]
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    tensor = torch.from_numpy(img).unsqueeze(0)
    return img, tensor

def load_model():
    print("  Loading DenseNet121 model...")
    print("  (First run downloads ~100MB — takes 1-2 minutes)")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    print("  Model loaded!")
    return model

def run_inference(model, tensor):
    with torch.no_grad():
        predictions = model(tensor)
    return predictions.cpu().numpy()[0]

def generate_triage_report(scores, pathologies, patient_id="PAT001"):
    results  = dict(zip(pathologies, scores))
    flagged  = {k: v for k, v in results.items() if v >= ALERT_THRESHOLD}
    critical = {k: v for k, v in flagged.items() if v >= 0.5}
    status   = "CRITICAL — Refer immediately" if critical else \
               "ATTENTION — Monitor closely"  if flagged else \
               "NORMAL — No significant findings"

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║         PRANA X-RAY TRIAGE REPORT               ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Patient : {patient_id:<39s}║")
    print(f"  ║  Model   : DenseNet121 (18 conditions)          ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  Condition               Score   Flag            ║")
    print("  ╠══════════════════════════════════════════════════╣")

    for condition, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar  = "█" * int(score * 10)
        flag = "  ⚠ HIGH" if score >= 0.5 else \
               "  △ MED " if score >= ALERT_THRESHOLD else ""
        print(f"  ║  {condition:<24s} {score:.2f}  {bar:<10s}{flag:<8s}║")

    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Status: {status:<41s}║")
    print("  ╚══════════════════════════════════════════════════╝")
    return results, flagged, status

def save_visualisation(img_array, results, status, patient_id="PAT001"):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"PRANA X-Ray Report — {patient_id}", fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_array[0], cmap='gray')
    ax1.set_title("Chest X-Ray", fontsize=12)
    ax1.axis('off')
    color = '#e74c3c' if 'CRITICAL' in status else \
            '#e67e22' if 'ATTENTION' in status else '#27ae60'
    ax1.text(0.5, 0.05, status.split('—')[0].strip(),
             transform=ax1.transAxes, ha='center', fontsize=11,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.85))

    ax2 = fig.add_subplot(gs[0, 1])
    conditions = list(results.keys())
    scores     = list(results.values())
    bar_colors = ['#e74c3c' if s>=0.5 else '#e67e22' if s>=ALERT_THRESHOLD else '#2ecc71'
                  for s in scores]
    bars = ax2.barh(conditions, scores, color=bar_colors, alpha=0.85)
    ax2.axvline(x=ALERT_THRESHOLD, color='orange', linestyle='--',
                linewidth=1.2, label=f'Alert ({ALERT_THRESHOLD})')
    ax2.axvline(x=0.5, color='red', linestyle='--',
                linewidth=1.2, label='Critical (0.5)')
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Probability Score", fontsize=11)
    ax2.set_title("Condition Probability Scores", fontsize=12)
    ax2.legend(fontsize=9)
    for bar, score in zip(bars, scores):
        ax2.text(min(score+0.02, 0.92), bar.get_y()+bar.get_height()/2,
                 f'{score:.2f}', va='center', fontsize=8)

    plt.savefig("prana_xray_report.png", dpi=150, bbox_inches='tight', facecolor='white')
    print("\n  Plot saved as prana_xray_report.png")

# ── Main ─────────────────────────────────────────────────────
print("=" * 58)
print("  PRANA — X-Ray Analysis Module")
print("=" * 58)

print("\n[Step 1] Loading pre-trained model...")
model = load_model()

print("\n[Step 2] Loading X-ray image...")
img_array, tensor = load_and_preprocess(image_path=IMAGE_PATH, use_demo=USE_DEMO_IMAGE)
print(f"  Image shape: {img_array.shape}")

print("\n[Step 3] Running AI analysis...")
scores = run_inference(model, tensor)
print(f"  Analysis complete — {len(scores)} conditions evaluated")

print("\n[Step 4] Generating triage report...")
results, flagged, status = generate_triage_report(scores, model.pathologies, patient_id="PAT001")

print("\n[Step 5] Saving visualisation...")
save_visualisation(img_array, results, status, patient_id="PAT001")

print("\n[Done] X-Ray module complete.")
print(f"\nFlagged conditions ({len(flagged)}):")
for condition, score in sorted(flagged.items(), key=lambda x: x[1], reverse=True):
    print(f"  {condition}: {score:.3f}")