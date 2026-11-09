"""
exp_1min_loso.py - T_core formulas that use ONLY one 60 s TriaGo acquisition window.

Each sample = 6 consecutive 10 s epochs (window_idx contiguous, same segment).
Features come from inside the window only (mean skin, mean ambient, mean PI, OLS skin slope).
Label = mean T_core of the same window. No EMA, no history before the window.

Protocol (same cohort + same 80/20 subject split, seed 2026, as exp_loso_improve.py):
  Stage A: LOSO inside the 80 train subjects -> pick the design (pass count, tie-break MAE).
  Stage B: LOSO over all 100 subjects; report the 20 original test subjects + all 100.
           Also the fixed 80 -> 20 fit.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(DIR, "cohort100_epochs_realistic_amb.parquet"))
df = df.sort_values(["caseid", "seg_id", "window_idx"]).reset_index(drop=True)
df["blk"] = df["window_idx"] // 6
df["pos"] = df["window_idx"] % 6

# Canonical split: shuffle over all 100 cases BEFORE any filtering.
cases = sorted(df["caseid"].unique())
assert len(cases) == 100
rng = np.random.default_rng(2026)
sh = list(cases)
rng.shuffle(sh)
TRAIN, TEST = sorted(sh[:80]), sorted(sh[80:])
assert not set(TRAIN) & set(TEST)

T = np.arange(6) * 10.0
T = T - T.mean()


def window_row(g):
    return pd.Series({
        "ts": g["t_skin_meas"].mean(),
        "ta": g["t_amb"].mean(),
        "p": 0.10 * g["pi"].mean(),
        "slope": float(np.dot(T, g["t_skin_meas"].to_numpy()) / np.dot(T, T)),  # OLS, C/s
        "y": g["t_core"].mean(),
    })


# Keep only complete, time-contiguous 60 s windows (6 epochs, no SQI gaps).
full = df.groupby(["caseid", "seg_id", "blk"]).filter(
    lambda g: len(g) == 6 and g["window_idx"].max() - g["window_idx"].min() == 5)
W = full.groupby(["caseid", "seg_id", "blk"]).apply(window_row).reset_index()
W["p"] = np.clip(W["p"], 0.005, 10.0)
CID = W["caseid"].to_numpy()
print(f"windows: {len(W)} from {W['caseid'].nunique()} subjects "
      f"(test subjects with windows: {len(set(TEST) & set(CID))}/20)")
assert set(TEST) <= set(CID), "a test subject has no complete 60 s window"

X = {c: W[c].to_numpy() for c in ["ts", "ta", "p", "slope"]}
Y = W["y"].to_numpy()


def bio(ts, ta, p, a, b):
    return (ts - ta) / np.maximum(a + b * p, 0.1)


M = {
    "S0 Linear Rational + c0 (static)": (
        lambda x, a, b, c: x["ts"] + bio(x["ts"], x["ta"], x["p"], a, b) + c, [0.85, 1.0, 1.0]),
    "S1 Basic power-law (static)": (
        lambda x, a, b, al: x["ts"] + (x["ts"] - x["ta"]) / np.maximum(a + b * np.power(x["p"], al), 0.1),
        [0.6, 0.9, 0.75]),
    "W1 + 60 s slope (additive)": (
        lambda x, a, b, k, c: x["ts"] + bio(x["ts"], x["ta"], x["p"], a, b) + k * x["slope"] + c,
        [0.85, 0.85, 100.0, 0.8]),
    "W2 slope-corrected skin inside bioheat": (
        lambda x, a, b, k, c: (x["ts"] + k * x["slope"])
        + bio(x["ts"] + k * x["slope"], x["ta"], x["p"], a, b) + c,
        [0.85, 0.85, 100.0, 0.8]),
    "W3 + slope + slope x PI": (
        lambda x, a, b, k, m, c: x["ts"] + bio(x["ts"], x["ta"], x["p"], a, b)
        + (k + m * x["p"]) * x["slope"] + c, [0.85, 0.85, 100.0, 0.0, 0.8]),
}


def sub(idx):
    return {c: v[idx] for c, v in X.items()}


def fit(fn, p0, idx):
    f = lambda x, *q: np.clip(fn(x, *q), 30.0, 45.0)
    return curve_fit(f, sub(idx), Y[idx], p0=p0, method="lm", maxfev=20000)[0]


def pred(fn, q, idx):
    return np.clip(fn(sub(idx), *q), 30.0, 45.0)


def loso(fn, p0, subjects):
    pool = np.isin(CID, subjects)
    warm = fit(fn, p0, np.where(pool)[0])
    per, errs = {}, []
    for s in subjects:
        tr, te = np.where(pool & (CID != s))[0], np.where(CID == s)[0]
        assert s not in set(CID[tr])  # held-out subject never trains
        e = pred(fn, fit(fn, warm, tr), te) - Y[te]
        per[int(s)] = float(np.mean(np.abs(e)))
        errs.append(e)
    return per, np.concatenate(errs)


def summ(per, e, subj):
    m = np.array([per[int(s)] for s in subj])
    return {"pooled_mae": round(float(np.mean(np.abs(e))), 4), "macro_mae": round(float(m.mean()), 4),
            "bias": round(float(e.mean()), 4), "sd": round(float(e.std(ddof=1)), 4),
            "pass": int((m <= 0.35).sum()), "n": int(m.size), "worst": round(float(m.max()), 4)}


have = sorted(set(CID))
train_h = [int(s) for s in TRAIN if s in set(have)]
out = {
    "n_windows": int(len(W)),
    "train_subjects": [int(s) for s in TRAIN],
    "test_subjects": [int(s) for s in TEST]
}

print("\nSTAGE A - LOSO inside train subjects (design selection)")
sa = {}
for n, (fn, p0) in M.items():
    per, e = loso(fn, p0, train_h)
    sa[n] = summ(per, e, train_h)
    print(f"{n:<42} | MAE {sa[n]['pooled_mae']:.4f} | pass {sa[n]['pass']}/{sa[n]['n']}")
best = max(sa, key=lambda k: (sa[k]["pass"], -sa[k]["pooled_mae"]))
print("Selected by Stage A:", best)
out["stage_a"], out["selected"] = sa, best

print("\nSTAGE B - LOSO over all subjects; test-20 subgroup; fixed 80->20")
tr_i, te_i = np.where(np.isin(CID, train_h))[0], np.where(np.isin(CID, TEST))[0]
sb = {}
for n, (fn, p0) in M.items():
    per, e = loso(fn, p0, have)
    q = fit(fn, p0, tr_i)
    ef = pred(fn, q, te_i) - Y[te_i]
    pf = pd.Series(np.abs(ef)).groupby(CID[te_i]).mean()
    m20 = np.array([per[int(s)] for s in TEST])
    sb[n] = {"loso_all": summ(per, e, have),
             "loso_test20": {"macro_mae": round(float(m20.mean()), 4), "pass": int((m20 <= 0.35).sum()),
                             "worst": round(float(m20.max()), 4)},
             "fixed_80_20": {"pooled_mae": round(float(np.abs(ef).mean()), 4), "pass": int((pf <= 0.35).sum()),
                             "params": [round(float(v), 5) for v in q]}}
    b = sb[n]
    print(f"{n:<42} | LOSO-all MAE {b['loso_all']['pooled_mae']:.4f} pass {b['loso_all']['pass']}/{b['loso_all']['n']}"
          f" | LOSO test-20 pass {b['loso_test20']['pass']}/20 (worst {b['loso_test20']['worst']:.3f})"
          f" | fixed 80/20 MAE {b['fixed_80_20']['pooled_mae']:.4f} pass {b['fixed_80_20']['pass']}/20")
out["stage_b"] = sb
json.dump(
    out, 
    open(os.path.join(DIR, "loso_1min_summary.json"), "w"), 
    indent=2, 
    default=lambda x: int(x) if isinstance(x, (np.int64, np.integer)) else float(x) if isinstance(x, (np.float64, np.floating)) else str(x)
)
print(f"\nSuccessfully saved summary JSON to: {os.path.join(DIR, 'loso_1min_summary.json')}")
