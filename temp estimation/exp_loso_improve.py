"""
exp_loso_improve.py - LOSO search for a formula that passes >= 16/20 test subjects.

Protocol (subject-independent, no test snooping):
  Stage A (design selection): LOSO inside the 80 train subjects only.
          Each fold fits on 79 subjects, scores the held-out one.
          Variant is chosen by pass count (MAE <= 0.35 C), tie-break pooled MAE.
  Stage B (final report): LOSO over all 100 subjects (fit on 99, score 1).
          Pass count is reported for the 20 original test subjects and all 100.
          The fixed 80 -> 20 fit is reported too.

Same cohort and same 80/20 split (seed 2026) as exp_linear_rational_c0_80_20.py.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(DIR, "cohort100_epochs.parquet"))
df = df.sort_values(["caseid", "seg_id", "window_idx"]).reset_index(drop=True)
df["p"] = np.clip(0.10 * df["pi"], 0.005, 10.0)
g = df.groupby(["caseid", "seg_id"])


def ema(col, span):
    # Causal per-segment EMA (adjust=False -> recursive form, past samples only).
    return g[col].transform(lambda s: s.ewm(span=span, adjust=False).mean())


for n in (6, 12, 18, 30):
    df[f"e{n}"] = ema("t_skin_meas", n)
df["ta_s"] = ema("t_amb", 18)   # smoothed ambient (~3 min)
df["p_s"] = ema("p", 6)          # smoothed perfusion (~1 min)

COLS = ["t_skin_meas", "t_amb", "p", "ta_s", "p_s", "e6", "e12", "e18", "e30"]
X_ALL = {c: df[c].to_numpy() for c in COLS}
Y_ALL = df["t_core"].to_numpy()
CID = df["caseid"].to_numpy()


def lin_rat(ts, ta, p, a, b, c):
    return ts + (ts - ta) / np.maximum(a + b * p, 0.1) + c


def corrected(n):
    # Lead-corrected skin goes INSIDE the bioheat term (inverse of first-order skin lag).
    def f(x, a, b, k, c):
        tc = x["t_skin_meas"] + k * (x["t_skin_meas"] - x[f"e{n}"])
        return lin_rat(tc, x["ta_s"], x["p_s"], a, b, c)
    return f, [0.84, 1.02, 3.0, 1.15]


V = {
    "R   Linear Rational + c0 (ref)": (
        lambda x, a, b, c: lin_rat(x["t_skin_meas"], x["t_amb"], x["p"], a, b, c), [0.85, 1.07, 1.2]),
    "L6  Lead filter span6 (previous)": (
        lambda x, a, b, k, c: lin_rat(x["t_skin_meas"], x["t_amb"], x["p"], a, b, c)
        + k * (x["t_skin_meas"] - x["e6"]), [0.84, 1.02, 1.5, 1.15]),
    "L18 Lead span18": (
        lambda x, a, b, k, c: lin_rat(x["t_skin_meas"], x["t_amb"], x["p"], a, b, c)
        + k * (x["t_skin_meas"] - x["e18"]), [0.84, 1.02, 1.5, 1.15]),
    "S18 Lead span18 + smoothed Tamb/PI": (
        lambda x, a, b, k, c: lin_rat(x["t_skin_meas"], x["ta_s"], x["p_s"], a, b, c)
        + k * (x["t_skin_meas"] - x["e18"]), [0.84, 1.02, 1.5, 1.15]),
    "C12 Corrected-skin span12 + smoothed": corrected(12),
    "C18 Corrected-skin span18 + smoothed": corrected(18),
    "C30 Corrected-skin span30 + smoothed": corrected(30),
}


def sub(idx):
    return {c: v[idx] for c, v in X_ALL.items()}


def fit(fn, p0, idx):
    f = lambda x, *q: np.clip(fn(x, *q), 30.0, 45.0)
    popt, _ = curve_fit(f, sub(idx), Y_ALL[idx], p0=p0, method="lm", maxfev=20000)
    return popt


def predict(fn, popt, idx):
    return np.clip(fn(sub(idx), *popt), 30.0, 45.0)


def loso(fn, p0, subjects):
    """Fit on all-but-one of `subjects`, predict the held-out one. Returns per-subject MAE and errors."""
    mask_pool = np.isin(CID, subjects)
    warm = fit(fn, p0, np.where(mask_pool)[0])  # warm start only; not used for scoring
    per, errs = {}, []
    for s in subjects:
        tr = np.where(mask_pool & (CID != s))[0]
        te = np.where(CID == s)[0]
        assert not np.isin(s, CID[tr])  # held-out subject never in training
        popt = fit(fn, warm, tr)
        e = predict(fn, popt, te) - Y_ALL[te]
        per[int(s)] = float(np.mean(np.abs(e)))
        errs.append(e)
    e = np.concatenate(errs)
    return per, e


def summary(per, e, subjects):
    m = np.array([per[int(s)] for s in subjects])
    return {
        "pooled_mae": round(float(np.mean(np.abs(e))), 4),
        "macro_mae": round(float(m.mean()), 4),
        "bias": round(float(e.mean()), 4),
        "sd": round(float(e.std(ddof=1)), 4),
        "pass": int((m <= 0.35).sum()),
        "n": int(m.size),
    }


cases = sorted(np.unique(CID))
rng = np.random.default_rng(2026)
sh = list(cases)
rng.shuffle(sh)
TRAIN, TEST = sorted(sh[:80]), sorted(sh[80:])

t0 = time.time()
out = {"train_subjects": [int(c) for c in TRAIN], "test_subjects": [int(c) for c in TEST]}

print("STAGE A - LOSO inside 80 train subjects (design selection)")
print(f"{'Variant':<38} | {'LOSO MAE':<9} | {'Macro':<7} | Pass/80")
stage_a = {}
for name, (fn, p0) in V.items():
    per, e = loso(fn, p0, TRAIN)
    s = summary(per, e, TRAIN)
    stage_a[name] = s
    print(f"{name:<38} | {s['pooled_mae']:.4f} C | {s['macro_mae']:.4f} | {s['pass']}/80")
best = max(stage_a, key=lambda k: (stage_a[k]["pass"], -stage_a[k]["pooled_mae"]))
print(f"Selected by Stage A: {best}")
out["stage_a_loso_80"] = stage_a
out["selected"] = best

print("\nSTAGE B - LOSO over all 100 subjects + fixed 80->20 fit")
print(f"{'Variant':<38} | {'LOSO-100 MAE':<12} | Pass/100 | {'LOSO test-20 MAE':<16} | Pass/20 | Fixed 80->20 MAE | Pass/20")
stage_b = {}
tr_idx = np.where(np.isin(CID, TRAIN))[0]
te_idx = np.where(np.isin(CID, TEST))[0]
for name, (fn, p0) in V.items():
    per, e = loso(fn, p0, cases)
    s100 = summary(per, e, cases)
    m20 = np.array([per[int(s)] for s in TEST])
    popt = fit(fn, p0, tr_idx)
    ef = predict(fn, popt, te_idx) - Y_ALL[te_idx]
    perf = pd.Series(np.abs(ef)).groupby(CID[te_idx]).mean()
    stage_b[name] = {
        "loso_100": s100,
        "loso_test20": {"macro_mae": round(float(m20.mean()), 4), "pass": int((m20 <= 0.35).sum()),
                        "per_subject_mae": {int(s): round(per[int(s)], 4) for s in TEST}},
        "fixed_80_20": {"pooled_mae": round(float(np.mean(np.abs(ef))), 4),
                        "pass": int((perf <= 0.35).sum()),
                        "params": [round(float(v), 5) for v in popt]},
    }
    b = stage_b[name]
    print(f"{name:<38} | {s100['pooled_mae']:.4f} C     | {s100['pass']}/100   | "
          f"{b['loso_test20']['macro_mae']:.4f} C (macro) | {b['loso_test20']['pass']}/20   | "
          f"{b['fixed_80_20']['pooled_mae']:.4f} C         | {b['fixed_80_20']['pass']}/20")
out["stage_b"] = stage_b
out["runtime_sec"] = round(time.time() - t0, 1)

with open(os.path.join(DIR, "loso_improve_summary.json"), "w") as fh:
    json.dump(out, fh, indent=2)

# Self-check: reference fixed 80->20 must reproduce the earlier reported test MAE.
assert abs(stage_b["R   Linear Rational + c0 (ref)"]["fixed_80_20"]["pooled_mae"] - 0.3213) < 2e-3
print(f"\nDone in {out['runtime_sec']} s")
