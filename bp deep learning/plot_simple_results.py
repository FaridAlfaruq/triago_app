#!/usr/bin/env python3
"""
Simple Scatter (R^2) and Bland-Altman Plot for Transfer Learning BP Estimation.
Renders directly via plt.show().
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Robust path resolution (works regardless of current working directory)
SCRIPT_DIR = Path(__file__).resolve().parent
candidates = [
    SCRIPT_DIR / "Bp revisi" / "predictions_tl_v2.npz",
    Path("scratch/gemastik/Bp revisi/predictions_tl_v2.npz"),
    Path("Bp revisi/predictions_tl_v2.npz"),
]
data_path = next((p for p in candidates if p.exists()), None)
if data_path is None:
    raise FileNotFoundError(f"Could not find 'predictions_tl_v2.npz'. Checked: {[str(p) for p in candidates]}")

# 1. Load calibrated predictions (N=415 disjoint segments)
data = np.load(data_path)
ys_true = data["ys_true"]
yd_true = data["yd_true"]
ys_pred = data["ys_pred"]
yd_pred = data["yd_pred"]

# 2. Compute statistics
# SBP
err_s = ys_pred - ys_true
bias_s = np.mean(err_s)
std_s = np.std(err_s)
r2_s = np.corrcoef(ys_true, ys_pred)[0, 1] ** 2

# DBP
err_d = yd_pred - yd_true
bias_d = np.mean(err_d)
std_d = np.std(err_d)
r2_d = np.corrcoef(yd_true, yd_pred)[0, 1] ** 2

# 3. Create 2x2 Simple Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# --- [A] SBP Scatter Plot (R^2) ---
ax = axes[0, 0]
ax.scatter(ys_true, ys_pred, alpha=0.5, edgecolors="none", s=25, c="#1f77b4")
lims_s = [min(ys_true.min(), ys_pred.min()) - 5, max(ys_true.max(), ys_pred.max()) + 5]
ax.plot(lims_s, lims_s, "k--", alpha=0.7, label="Identity (y=x)")
# Fit line
p_s = np.polyfit(ys_true, ys_pred, 1)
ax.plot(lims_s, np.polyval(p_s, lims_s), "r-", label=f"Fit (R² = {r2_s:.3f})")
ax.set_title(f"SBP Correlation (R² = {r2_s:.3f}, MAE = {np.mean(np.abs(err_s)):.2f})")
ax.set_xlabel("Reference SBP (mmHg)")
ax.set_ylabel("Predicted SBP (mmHg)")
ax.set_xlim(lims_s)
ax.set_ylim(lims_s)
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

# --- [B] DBP Scatter Plot (R^2) ---
ax = axes[0, 1]
ax.scatter(yd_true, yd_pred, alpha=0.5, edgecolors="none", s=25, c="#2ca02c")
lims_d = [min(yd_true.min(), yd_pred.min()) - 5, max(yd_true.max(), yd_pred.max()) + 5]
ax.plot(lims_d, lims_d, "k--", alpha=0.7, label="Identity (y=x)")
# Fit line
p_d = np.polyfit(yd_true, yd_pred, 1)
ax.plot(lims_d, np.polyval(p_d, lims_d), "r-", label=f"Fit (R² = {r2_d:.3f})")
ax.set_title(f"DBP Correlation (R² = {r2_d:.3f}, MAE = {np.mean(np.abs(err_d)):.2f})")
ax.set_xlabel("Reference DBP (mmHg)")
ax.set_ylabel("Predicted DBP (mmHg)")
ax.set_xlim(lims_d)
ax.set_ylim(lims_d)
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

# --- [C] SBP Bland-Altman Plot ---
ax = axes[1, 0]
mean_s = (ys_true + ys_pred) / 2.0
ax.scatter(mean_s, err_s, alpha=0.5, edgecolors="none", s=25, c="#1f77b4")
ax.axhline(bias_s, color="blue", linestyle="-", label=f"Bias: {bias_s:+.2f}")
ax.axhline(bias_s + 1.96 * std_s, color="red", linestyle="--", label=f"+1.96 SD: {bias_s + 1.96*std_s:.2f}")
ax.axhline(bias_s - 1.96 * std_s, color="red", linestyle="--", label=f"-1.96 SD: {bias_s - 1.96*std_s:.2f}")
ax.set_title(f"SBP Bland-Altman (Bias: {bias_s:+.2f} ± {std_s:.2f})")
ax.set_xlabel("Mean (Ref + Pred)/2 (mmHg)")
ax.set_ylabel("Difference (Pred - Ref) (mmHg)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.6)

# --- [D] DBP Bland-Altman Plot ---
ax = axes[1, 1]
mean_d = (yd_true + yd_pred) / 2.0
ax.scatter(mean_d, err_d, alpha=0.5, edgecolors="none", s=25, c="#2ca02c")
ax.axhline(bias_d, color="green", linestyle="-", label=f"Bias: {bias_d:+.2f}")
ax.axhline(bias_d + 1.96 * std_d, color="red", linestyle="--", label=f"+1.96 SD: {bias_d + 1.96*std_d:.2f}")
ax.axhline(bias_d - 1.96 * std_d, color="red", linestyle="--", label=f"-1.96 SD: {bias_d - 1.96*std_d:.2f}")
ax.set_title(f"DBP Bland-Altman (Bias: {bias_d:+.2f} ± {std_d:.2f})")
ax.set_xlabel("Mean (Ref + Pred)/2 (mmHg)")
ax.set_ylabel("Difference (Pred - Ref) (mmHg)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
print("Displaying simple Bland-Altman and Correlation plots via plt.show()...")
plt.show()
