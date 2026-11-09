import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path("scratch/gemastik/remote_jobs/pulsedb_calbased").resolve()))
from train_calbased import CalBPNet, compute_bhs_grades

data = np.load("scratch/gemastik/Bp revisi/dataset_7ch.npz")
X = data["X"]  # (732, 1250, 7)
ys = data["y_sbp"]
yd = data["y_dbp"]
groups = data["groups"]

unique_groups = np.unique(groups)
print(f"Loaded TriaGO Dataset: {len(X)} segments across {len(unique_groups)} unique patients")

# 1. Direct Forward Pass using PulseDB-trained Weights
ckpt_path = Path("scratch/gemastik/remote_jobs/pulsedb_calbased/output_v3/output/model_best.pt")
model = CalBPNet(in_channels=7)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()

s_mean_p, s_std_p = 118.27, 18.00
d_mean_p, d_std_p = 64.22, 12.27

with torch.no_grad():
    tx = torch.from_numpy(X).float()
    ps_z, pd_z = model(tx)
    ps = ps_z.numpy() * s_std_p + s_mean_p
    pd = pd_z.numpy() * d_std_p + d_mean_p

print("\n--- 1. Raw Forward Pass (Zero-Shot Cross-Domain: PulseDB -> TriaGO) ---")
err_raw_s = ps - ys
err_raw_d = pd - yd
print(f"SBP Raw: MAE={np.mean(np.abs(err_raw_s)):.2f} mmHg, STD={np.std(err_raw_s):.2f} mmHg")
print(f"DBP Raw: MAE={np.mean(np.abs(err_raw_d)):.2f} mmHg, STD={np.std(err_raw_d):.2f} mmHg")

# 2. 1-Point Baseline Calibration on TriaGO
cal_err_s, cal_err_d = [], []
eval_count = 0

for p in unique_groups:
    pts = np.where(groups == p)[0]
    if len(pts) <= 1:
        continue
    # Anchor is Segment 0 of this patient
    idx0 = pts[0]
    off_s = ys[idx0] - ps[idx0]
    off_d = yd[idx0] - pd[idx0]
    
    # Evaluate subsequent segments
    eval_pts = pts[1:]
    cal_s = ps[eval_pts] + off_s
    cal_d = pd[eval_pts] + off_d
    
    cal_err_s.extend(cal_s - ys[eval_pts])
    cal_err_d.extend(cal_d - yd[eval_pts])
    eval_count += len(eval_pts)

cal_s_arr = np.array(cal_err_s)
cal_d_arr = np.array(cal_err_d)

res_s = compute_bhs_grades(cal_s_arr)
res_d = compute_bhs_grades(cal_d_arr)

print(f"\n--- 2. 1-Point Calibrated Forward Pass (N={eval_count} segments across {len(unique_groups)} patients) ---")
print(f"SBP: MAE={res_s['mae']} mmHg, STD={res_s['std']} mmHg | <=5:{res_s['pct_le_5']}%, <=10:{res_s['pct_le_10']}%, <=15:{res_s['pct_le_15']}% | Grade: {res_s['bhs_grade']}")
print(f"DBP: MAE={res_d['mae']} mmHg, STD={res_d['std']} mmHg | <=5:{res_d['pct_le_5']}%, <=10:{res_d['pct_le_10']}%, <=15:{res_d['pct_le_15']}% | Grade: {res_d['bhs_grade']}")
