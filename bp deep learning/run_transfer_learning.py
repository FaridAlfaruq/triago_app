#!/usr/bin/env python3
"""
TASK-TL-KIM-BAEK-02: Modified Transfer Learning on TriaGO PPG Dataset
Backbone-Frozen + L2-SP Regularization + Calibration-Consistent Head Adaptation.
Methodology: Kim & Baek (Scientific Reports 2026, PMCID: PMC13039200)
Architecture: CalBPNet (Pure-Waveform CNN-BiGRU-MHSA)

Strict Execution Protocol:
- Frozen backbone context extraction in a single torch.no_grad() pass (~1.4s).
- Lightweight regression head fine-tuning (<1s total execution).
- Strict zero-leakage 1-point baseline calibration evaluation (N=415, anchor excluded).
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

PULSEDB_DIR = Path("scratch/gemastik/remote_jobs/pulsedb_calbased")
sys.path.append(str(PULSEDB_DIR.resolve()))
from train_calbased import CalBPNet, apply_tight_sqi, compute_bhs_grades  # noqa: E402

PDB_SBP_MEAN = 118.2714
PDB_SBP_STD = 17.9963
PDB_DBP_MEAN = 64.2229
PDB_DBP_STD = 12.2722


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_reg_head() -> nn.Sequential:
    """Standard lightweight regression head (256 -> 96 -> 1)."""
    return nn.Sequential(
        nn.Linear(256, 96),
        nn.BatchNorm1d(96),
        nn.SiLU(),
        nn.Dropout(0.15),
        nn.Linear(96, 1),
    )


@torch.no_grad()
def extract_backbone_context(model: CalBPNet, x: torch.Tensor) -> torch.Tensor:
    """Extract frozen 256-dim dual-pooling context before regression heads."""
    model.eval()
    bx = x.transpose(1, 2) if x.shape[1] == 1250 else x
    bx = model.stem(bx)
    bx = model.drop1(m_pool1 := model.pool1(model.inc1(bx)))
    bx = model.drop2(model.pool2(model.inc2(bx)))
    bx = model.drop3(model.pool3(model.inc3(bx)))
    bx = bx.transpose(1, 2)
    gru_out, _ = model.bigru(bx)
    attn_out, _ = model.mhsa(gru_out, gru_out, gru_out)
    bx = model.ln(gru_out + attn_out)
    avg_p = bx.mean(dim=1)
    max_p = bx.max(dim=1).values
    return torch.cat([avg_p, max_p], dim=1)


def compact_metric(r: dict) -> dict:
    return {
        "mae": round(float(r["mae"]), 2),
        "std": round(float(r["std"]), 2),
        "bhs": r["bhs_grade"],
        "pct": [round(float(p), 1) for p in [r["pct_le_5"], r["pct_le_10"], r["pct_le_15"]]],
    }


def main() -> None:
    t_start = time.time()
    set_seed(42)
    torch.set_num_threads(os.cpu_count() or 6)

    # 1. Load target TriaGO dataset & filter with tight SQI
    target_path = Path("scratch/gemastik/Bp revisi/dataset_7ch.npz")
    data = np.load(target_path)
    X_raw = data["X"]
    y_sbp_raw = data["y_sbp"]
    y_dbp_raw = data["y_dbp"]
    groups_raw = data["groups"]

    sqi_mask = apply_tight_sqi(X_raw, fs=125)
    X_clean = torch.from_numpy(X_raw[sqi_mask]).float()
    y_sbp = y_sbp_raw[sqi_mask]
    y_dbp = y_dbp_raw[sqi_mask]
    groups = groups_raw[sqi_mask]

    n_samples = len(X_clean)
    n_subjects = len(np.unique(groups))
    assert n_samples == 472 and n_subjects == 57

    # 2. Frozen Backbone Context Extraction (Single forward pass)
    weights_path = PULSEDB_DIR / "output_v3" / "output" / "model_best.pt"
    m_base = CalBPNet()
    m_base.load_state_dict(torch.load(weights_path, map_location="cpu"))
    m_base.eval()

    ctx = extract_backbone_context(m_base, X_clean)  # [472, 256]

    preds = {
        "scratch": {"s": np.zeros(n_samples), "d": np.zeros(n_samples)},
        "zero_shot": {"s": np.zeros(n_samples), "d": np.zeros(n_samples)},
        "transfer_learning": {"s": np.zeros(n_samples), "d": np.zeros(n_samples)},
    }
    cal_preds = {
        "scratch": {"s": np.zeros(n_samples), "d": np.zeros(n_samples)},
        "zero_shot": {"s": np.zeros(n_samples), "d": np.zeros(n_samples)},
        "transfer_learning": {"s": np.zeros(n_samples), "d": np.zeros(n_samples)},
    }
    eval_mask = np.zeros(n_samples, dtype=bool)

    # 3. Subject-Disjoint 3-Fold Cross-Validation
    gkf = GroupKFold(n_splits=3)
    huber = nn.SmoothL1Loss()

    # Pass 1: Modified Transfer Learning (Pretrained Head + L2-SP + Calibration-Consistent Loss)
    set_seed(42)
    for trn_idx, tst_idx in gkf.split(ctx, y_sbp, groups):
        assert len(set(groups[trn_idx]).intersection(set(groups[tst_idx]))) == 0
        s_mean, s_std = float(y_sbp[trn_idx].mean()), float(y_sbp[trn_idx].std())
        d_mean, d_std = float(y_dbp[trn_idx].mean()), float(y_dbp[trn_idx].std())

        x_trn = ctx[trn_idx]
        ys_trn_z = torch.from_numpy((y_sbp[trn_idx] - s_mean) / s_std).float()
        yd_trn_z = torch.from_numpy((y_dbp[trn_idx] - d_mean) / d_std).float()
        x_tst = ctx[tst_idx]

        # Zero-Shot
        with torch.no_grad():
            preds["zero_shot"]["s"][tst_idx] = m_base.sbp_head(x_tst).squeeze(-1).numpy() * PDB_SBP_STD + PDB_SBP_MEAN
            preds["zero_shot"]["d"][tst_idx] = m_base.dbp_head(x_tst).squeeze(-1).numpy() * PDB_DBP_STD + PDB_DBP_MEAN

        # Modified Transfer Learning
        sbp_tl = copy.deepcopy(m_base.sbp_head)
        dbp_tl = copy.deepcopy(m_base.dbp_head)
        sbp_init = {k: p.detach().clone() for k, p in sbp_tl.named_parameters()}
        dbp_init = {k: p.detach().clone() for k, p in dbp_tl.named_parameters()}

        trn_grps = groups[trn_idx]
        anc_map = torch.tensor([np.where(trn_grps == g)[0][0] for g in trn_grps], dtype=torch.long)

        opt_tl = torch.optim.Adam(list(sbp_tl.parameters()) + list(dbp_tl.parameters()), lr=5e-4)
        sbp_tl.train()
        dbp_tl.train()
        for _ in range(25):
            opt_tl.zero_grad()
            ps_z = sbp_tl(x_trn).squeeze(-1)
            pd_z = dbp_tl(x_trn).squeeze(-1)
            loss_abs = huber(ps_z, ys_trn_z) + huber(pd_z, yd_trn_z)
            loss_cal = huber(ps_z - ps_z[anc_map], ys_trn_z - ys_trn_z[anc_map]) + huber(
                pd_z - pd_z[anc_map], yd_trn_z - yd_trn_z[anc_map]
            )
            l2_sp = sum(torch.sum((p - sbp_init[k]) ** 2) for k, p in sbp_tl.named_parameters()) + sum(
                torch.sum((p - dbp_init[k]) ** 2) for k, p in dbp_tl.named_parameters()
            )
            loss = loss_abs + 1.5 * loss_cal + 0.05 * l2_sp
            loss.backward()
            opt_tl.step()

        sbp_tl.eval()
        dbp_tl.eval()
        with torch.no_grad():
            preds["transfer_learning"]["s"][tst_idx] = sbp_tl(x_tst).squeeze(-1).numpy() * s_std + s_mean
            preds["transfer_learning"]["d"][tst_idx] = dbp_tl(x_tst).squeeze(-1).numpy() * d_std + d_mean

    # Pass 2: Scratch Head (Fresh randomly initialized heads without L2-SP or Cal-Loss)
    set_seed(42)
    for trn_idx, tst_idx in gkf.split(ctx, y_sbp, groups):
        s_mean, s_std = float(y_sbp[trn_idx].mean()), float(y_sbp[trn_idx].std())
        d_mean, d_std = float(y_dbp[trn_idx].mean()), float(y_dbp[trn_idx].std())
        x_trn = ctx[trn_idx]
        ys_trn_z = torch.from_numpy((y_sbp[trn_idx] - s_mean) / s_std).float()
        yd_trn_z = torch.from_numpy((y_dbp[trn_idx] - d_mean) / d_std).float()
        x_tst = ctx[tst_idx]

        sbp_sc = make_reg_head()
        dbp_sc = make_reg_head()
        opt_sc = torch.optim.Adam(list(sbp_sc.parameters()) + list(dbp_sc.parameters()), lr=1e-3)
        sbp_sc.train()
        dbp_sc.train()
        for _ in range(25):
            opt_sc.zero_grad()
            ps = sbp_sc(x_trn).squeeze(-1)
            pd = dbp_sc(x_trn).squeeze(-1)
            loss = huber(ps, ys_trn_z) + huber(pd, yd_trn_z)
            loss.backward()
            opt_sc.step()

        sbp_sc.eval()
        dbp_sc.eval()
        with torch.no_grad():
            preds["scratch"]["s"][tst_idx] = sbp_sc(x_tst).squeeze(-1).numpy() * s_std + s_mean
            preds["scratch"]["d"][tst_idx] = dbp_sc(x_tst).squeeze(-1).numpy() * d_std + d_mean

        # 4. Strict Zero-Leakage 1-Point Calibration: Anchor segment gi[0] excluded from evaluation
        tst_grps = groups[tst_idx]
        for g in np.unique(tst_grps):
            gi = np.where(tst_grps == g)[0]
            if len(gi) > 1:
                anc_idx = tst_idx[gi[0]]
                eval_indices = tst_idx[gi[1:]]
                eval_mask[eval_indices] = True
                for cond in ["scratch", "zero_shot", "transfer_learning"]:
                    off_s = y_sbp[anc_idx] - preds[cond]["s"][anc_idx]
                    off_d = y_dbp[anc_idx] - preds[cond]["d"][anc_idx]
                    cal_preds[cond]["s"][eval_indices] = preds[cond]["s"][eval_indices] + off_s
                    cal_preds[cond]["d"][eval_indices] = preds[cond]["d"][eval_indices] + off_d

    n_cal_eval = int(np.sum(eval_mask))
    assert n_cal_eval == 415

    # 5. Compute Aggregate Out-Of-Fold Metrics
    results = {}
    for cond in ["scratch", "zero_shot", "transfer_learning"]:
        results[cond] = {
            "uncal_s": compute_bhs_grades(preds[cond]["s"] - y_sbp),
            "uncal_d": compute_bhs_grades(preds[cond]["d"] - y_dbp),
            "cal_s": compute_bhs_grades(cal_preds[cond]["s"][eval_mask] - y_sbp[eval_mask]),
            "cal_d": compute_bhs_grades(cal_preds[cond]["d"][eval_mask] - y_dbp[eval_mask]),
        }

    # 6. Build Compact Summary JSON (< 1 KB)
    summary = {
        "task_id": "TASK-TL-KIM-BAEK-02",
        "folds": 3,
        "n_samples": int(n_samples),
        "n_cal_samples": int(n_cal_eval),
        "conditions": {
            cond: {
                "uncalibrated": {
                    "sbp": compact_metric(results[cond]["uncal_s"]),
                    "dbp": compact_metric(results[cond]["uncal_d"]),
                },
                "calibrated": {
                    "sbp": compact_metric(results[cond]["cal_s"]),
                    "dbp": compact_metric(results[cond]["cal_d"]),
                },
            }
            for cond in ["scratch", "zero_shot", "transfer_learning"]
        },
    }

    out_json_path = Path("scratch/gemastik/Bp revisi/transfer_learning_summary.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, separators=(",", ":"))

    json_size = out_json_path.stat().st_size
    assert json_size < 1024, f"Summary JSON exceeded 1 KB ({json_size} bytes)"

    elapsed = time.time() - t_start

    def fmt_row(name: str, sm: dict, dm: dict) -> str:
        aami = "PASS" if (sm["aami_pass"] and dm["aami_pass"]) else "FAIL"
        return f"{name:<21} | {sm['mae']:5.2f} ({sm['std']:5.2f}) |  {sm['bhs_grade']}  | {dm['mae']:5.2f} ({dm['std']:5.2f}) |  {dm['bhs_grade']}  | {aami}"

    print("=" * 80)
    print("TASK-TL-KIM-BAEK-02: BACKBONE-FROZEN TRANSFER LEARNING BENCHMARK")
    print("=" * 80)
    print(f"TriaGO: {n_samples} seg ({n_subjects} subj) | Eval: {n_samples} uncal, {n_cal_eval} cal (anc excluded) | {elapsed:.2f}s")
    print("-" * 80)
    print("Condition             | SBP MAE (Std)  | BHS | DBP MAE (Std)  | BHS | AAMI")
    print("-" * 80)
    print(fmt_row("Scratch (Uncal)", results["scratch"]["uncal_s"], results["scratch"]["uncal_d"]))
    print(fmt_row("Scratch (Cal-1pt)", results["scratch"]["cal_s"], results["scratch"]["cal_d"]))
    print(fmt_row("Zero-Shot (Uncal)", results["zero_shot"]["uncal_s"], results["zero_shot"]["uncal_d"]))
    print(fmt_row("Zero-Shot (Cal-1pt)", results["zero_shot"]["cal_s"], results["zero_shot"]["cal_d"]))
    print(fmt_row("Transfer (Uncal)", results["transfer_learning"]["uncal_s"], results["transfer_learning"]["uncal_d"]))
    print(fmt_row("Transfer (Cal-1pt)", results["transfer_learning"]["cal_s"], results["transfer_learning"]["cal_d"]))
    print("=" * 80)
    print(f"Summary JSON saved: {out_json_path} ({json_size} bytes)")


if __name__ == "__main__":
    main()
