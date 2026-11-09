"""
exp_unconventional_formulas.py - Screen unconventional simple T_core formulas.

Same 100-case cohort and same 80/20 patient split (seed 2026) as
exp_linear_rational_c0_80_20.py, so results are directly comparable.
Each formula is fitted on 80 train patients (TRF curve_fit) and scored on
20 unseen test patients.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(DIR, "cohort100_epochs.parquet"))
df["p"] = np.clip(0.10 * df["pi"], 0.005, 10.0)
df["d"] = df["t_skin_meas"] - df["t_amb"]
# Causal EMA of skin temp per segment (span 6 epochs = ~60 s), used by the lead filter.
df["ema"] = df.groupby(["caseid", "seg_id"])["t_skin_meas"].transform(
    lambda s: s.ewm(span=6, adjust=False).mean())
# Causal running max over 180 s (18 epochs), used by the max-hold formula.
df["mx"] = df.groupby(["caseid", "seg_id"])["t_skin_meas"].transform(
    lambda s: s.rolling(18, min_periods=1).max())
# Fixed PI bin edges (tertiles of the whole cohort PI_eff; no label used).
EDGES = np.quantile(df["p"], [1 / 3, 2 / 3])
df["bin"] = np.digitize(df["p"], EDGES)

cases = sorted(df["caseid"].unique())
rng = np.random.default_rng(2026)
sh = cases.copy()
rng.shuffle(sh)
tr = df[df["caseid"].isin(sh[:80])].reset_index(drop=True)
te = df[df["caseid"].isin(sh[80:])].reset_index(drop=True)

S, A, P, D = "t_skin_meas", "t_amb", "p", "d"

# name -> (fn(df, *params), p0, idea)
F = {
    "0. Linear Rational + c0 (reference)": (
        lambda x, a, b, c: x[S] + x[D] / (a + b * x[P]) + c, [0.85, 1.07, 1.2],
        "Current pick"),
    "U1. Skin + constant": (
        lambda x, c: x[S] + c, [3.0],
        "Ignore ambient and PI entirely"),
    "U2. Mirror (no PI)": (
        lambda x, k, c: x[S] + k * x[D] + c, [0.3, 1.0],
        "Reflect ambient across skin, fixed gain"),
    "U3. Inverse-PI offset": (
        lambda x, a, b: x[S] + a + b / (x[P] + 0.1), [2.0, 0.2],
        "Offset grows when perfusion drops; ambient ignored"),
    "U4. Exponential perfusion gate": (
        lambda x, a, b, k: x[S] + x[D] * (a + b * np.exp(-k * x[P])), [0.3, 0.5, 1.0],
        "Gain decays exponentially with PI"),
    "U5. Kelvin power law": (
        lambda x, k, c: (x[S] + 273.15) * ((x[S] + 273.15) / (x[A] + 273.15)) ** k - 273.15 + c,
        [1.0, 0.0], "Multiplicative ratio in absolute temperature"),
    "U6. PI tertile lookup table": (
        lambda x, c0, c1, c2: x[S] + np.choose(x["bin"].values, [c0, c1, c2]), [3, 2, 1],
        "3 constants picked by PI bin, no physics"),
    "U7. Lead filter (anti-lag)": (
        lambda x, a, b, k, c: x[S] + x[D] / (a + b * x[P]) + k * (x[S] - x["ema"]) + c,
        [0.85, 1.07, 3.0, 1.2], "Linear Rational + 1 EMA lead term"),
    "U8. Max-hold 180 s": (
        lambda x, a, b, c: x["mx"] + (x["mx"] - x[A]) / (a + b * x[P]) + c, [0.85, 1.07, 1.2],
        "Use running max of skin instead of current skin"),
}


def score(y, yp, sub):
    e = yp - y
    per = sub.assign(e=np.abs(e)).groupby("caseid")["e"].mean()
    return {
        "mae": round(float(np.mean(np.abs(e))), 4),
        "bias": round(float(np.mean(e)), 4),
        "sd": round(float(np.std(e, ddof=1)), 4),
        "pass": int((per <= 0.35).sum()),
        "n": int(per.size),
    }


out = {}
print(f"{'Formula':<38} | {'k':<2} | {'Test MAE':<9} | {'Bias':<8} | {'SD':<7} | {'Pass':<9} | Params")
print("-" * 115)
for name, (fn, p0, idea) in F.items():
    f = lambda X, *q, fn=fn: np.clip(fn(X, *q), 30.0, 45.0)
    popt, _ = curve_fit(f, tr, tr["t_core"].values, p0=p0, method="trf", maxfev=20000)
    m = score(te["t_core"].values, f(te, *popt), te)
    out[name] = {"idea": idea, "params": [round(float(v), 5) for v in popt], "test": m}
    print(f"{name:<38} | {len(popt):<2} | {m['mae']:.4f} C | {m['bias']:+.4f} | {m['sd']:.4f} | "
          f"{m['pass']}/{m['n']} ({m['pass'] / m['n'] * 100:.0f}%) | {[round(v, 4) for v in popt]}")

with open(os.path.join(DIR, "unconventional_formulas_summary.json"), "w") as fh:
    json.dump({"pi_tertile_edges": [float(v) for v in EDGES], "results": out}, fh, indent=2)

# Self-check: reference row must reproduce the previously reported test MAE.
assert abs(out["0. Linear Rational + c0 (reference)"]["test"]["mae"] - 0.3213) < 2e-3
