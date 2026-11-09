"""
Interactive 2-Line (Train vs Validation) Curve Viewer
Loaded directly from real training phase history (training_history.csv).
Rule: plt.show() only, NEVER plt.savefig().
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("scratch/gemastik/Bp revisi/training_history.csv")
if not csv_path.exists():
    csv_path = Path("scratch/gemastik/remote_jobs/pulsedb_calbased/output_v3/output/training_history.csv")

df = pd.read_csv(csv_path)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: SBP MAE (Train vs Validation)
ax1.plot(df['epoch'], df['train_sbp_mae'], color='#3498db', linewidth=2, label='Train SBP MAE')
ax1.plot(df['epoch'], df['val_sbp_mae'], color='#e74c3c', linewidth=2, linestyle='-', label='Val SBP MAE')
ax1.axhline(5.0, color='#27ae60', linestyle='--', linewidth=1.5, label='BHS Grade A Bound (5.0 mmHg)')
ax1.set_title('Systolic BP (SBP): Train vs Validation MAE', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('MAE (mmHg)', fontsize=11)
ax1.set_ylim(2.0, 14.0)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=10)

# Panel 2: DBP MAE (Train vs Validation)
ax2.plot(df['epoch'], df['train_dbp_mae'], color='#2ecc71', linewidth=2, label='Train DBP MAE')
ax2.plot(df['epoch'], df['val_dbp_mae'], color='#9b59b6', linewidth=2, linestyle='-', label='Val DBP MAE')
ax2.axhline(5.0, color='#27ae60', linestyle='--', linewidth=1.5, label='BHS Grade A Bound (5.0 mmHg)')
ax2.set_title('Diastolic BP (DBP): Train vs Validation MAE', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('MAE (mmHg)', fontsize=11)
ax2.set_ylim(1.5, 9.0)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=10)

# Panel 3: Total Loss (Train vs Validation)
ax3.plot(df['epoch'], df['train_loss'], color='#f39c12', linewidth=2, label='Train Loss')
ax3.plot(df['epoch'], df['val_loss'], color='#d35400', linewidth=2, linestyle='-', label='Val Loss')
ax3.set_title('Physics-BHS Huber Loss: Train vs Validation', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Loss', fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(fontsize=10)

plt.suptitle('CalBPNet (CNN-BiGRU) v3: Empirical Train vs. Validation Learning Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
