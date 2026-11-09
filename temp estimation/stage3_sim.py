"""
stage3_sim.py - Runner Script for Stage 3 Bio-Heat Dynamic Simulation
Author: Alex (Lead Orchestrator)
Master Reference: scratch/temp estimation/PRD_VitalDB_Temp_Validation.md

Simulates:
1. Dynamic skin temperature T_skin across all 10 VitalDB surgical cases
2. Melexis MLX90614 medical infrared thermopile sensor noise and 17-bit quantization (T_skin_meas)
3. Outputs case-by-case and aggregate Min/Max/Mean/Std values for T_skin and T_skin_meas
4. Stops cleanly before Stage 4 calibration
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Import Stage 3 simulation core
from stage3 import process_case_stage3


def run_stage3_simulation(case_ids=None):
    if case_ids is None:
        case_ids = list(range(1, 11))

    print("=" * 115)
    print("STAGE 3: DYNAMIC SKIN TEMPERATURE & MLX90614 SENSOR NOISE SIMULATION")
    print("Biophysical Model: Stolwijk-Hardy / Pennes ODE (tau=180s) | Sensor: MLX90614 (NETD=0.05C, 17-bit 0.02C)")
    print("=" * 115)

    all_case_stats = []
    all_sim_dfs = []

    for cid in case_ids:
        print(f"Simulating Case {cid:2d}/10 ... ", end="", flush=True)
        res = process_case_stage3(cid)
        if "stats" in res and res["stats"].get("status") == "OK":
            st = res["stats"]
            all_case_stats.append(st)
            all_sim_dfs.append(res["df_sim"])
            print(
                f"OK | Epochs: {st['n_epochs']:>4} ({st['valid_duration_min']:5.1f}m) | "
                f"T_core: [{st['t_core_min']:4.1f}, {st['t_core_max']:4.1f}] | "
                f"T_skin: [{st['t_skin_min']:4.2f}, {st['t_skin_max']:4.2f}] C | "
                f"T_meas: [{st['t_meas_min']:4.2f}, {st['t_meas_max']:4.2f}] C | "
                f"Delta(Core-Skin): {st['delta_core_skin_mean']:4.2f} C"
            )
        else:
            print(f"FAILED: {res.get('status')}")

    if not all_case_stats:
        print("ERROR: No cases simulated successfully.")
        return

    # Combine all epochs across cohort
    df_cohort = pd.concat(all_sim_dfs, ignore_index=True)

    cohort_t_skin = df_cohort["t_skin"].values
    cohort_t_meas = df_cohort["t_skin_meas"].values
    cohort_t_core = df_cohort["t_core"].values
    cohort_pi = df_cohort["pi"].values

    cohort_summary = {
        "total_cases": len(all_case_stats),
        "total_epochs": len(df_cohort),
        "total_duration_hours": round(len(df_cohort) * 10.0 / 3600.0, 2),
        "t_core_min": round(float(np.min(cohort_t_core)), 2),
        "t_core_max": round(float(np.max(cohort_t_core)), 2),
        "t_core_mean": round(float(np.mean(cohort_t_core)), 2),
        "t_skin_min": round(float(np.min(cohort_t_skin)), 2),
        "t_skin_max": round(float(np.max(cohort_t_skin)), 2),
        "t_skin_mean": round(float(np.mean(cohort_t_skin)), 2),
        "t_skin_std": round(float(np.std(cohort_t_skin)), 2),
        "t_meas_min": round(float(np.min(cohort_t_meas)), 2),
        "t_meas_max": round(float(np.max(cohort_t_meas)), 2),
        "t_meas_mean": round(float(np.mean(cohort_t_meas)), 2),
        "t_meas_std": round(float(np.std(cohort_t_meas)), 2),
        "delta_core_skin_mean": round(float(np.mean(cohort_t_core - cohort_t_skin)), 2),
        "delta_core_skin_min": round(float(np.min(cohort_t_core - cohort_t_skin)), 2),
        "delta_core_skin_max": round(float(np.max(cohort_t_core - cohort_t_skin)), 2),
        "pi_mean": round(float(np.mean(cohort_pi)), 2),
        "pi_min": round(float(np.min(cohort_pi)), 2),
        "pi_max": round(float(np.max(cohort_pi)), 2)
    }

    # Format into detailed audit table
    df_table = pd.DataFrame(all_case_stats)[[
        "caseid", "n_epochs", "valid_duration_min",
        "t_core_min", "t_core_max", "t_core_mean",
        "t_skin_min", "t_skin_max", "t_skin_mean",
        "t_meas_min", "t_meas_max", "t_meas_mean",
        "delta_core_skin_mean"
    ]]
    df_table.columns = [
        "Case", "Epochs", "Dur(m)",
        "T_core Min", "T_core Max", "T_core Mean",
        "T_skin Min", "T_skin Max", "T_skin Mean",
        "T_meas Min", "T_meas Max", "T_meas Mean",
        "Core-Skin"
    ]

    print("\n" + "=" * 115)
    print("STAGE 3 SIMULATION AUDIT TABLE: MIN / MAX / MEAN VALUES (10 SURGICAL CASES)")
    print("=" * 115)
    print(df_table.to_string(index=False))
    print("-" * 115)
    print(
        f"COHORT AGGREGATE ({cohort_summary['total_cases']} Cases | "
        f"{cohort_summary['total_epochs']:,} Epochs | {cohort_summary['total_duration_hours']:.2f} Hours):\n"
        f"  * T_core range:      [{cohort_summary['t_core_min']:.2f}, {cohort_summary['t_core_max']:.2f}] degC  (Mean: {cohort_summary['t_core_mean']:.2f} degC)\n"
        f"  * T_skin range:      [{cohort_summary['t_skin_min']:.2f}, {cohort_summary['t_skin_max']:.2f}] degC  (Mean: {cohort_summary['t_skin_mean']:.2f} degC, Std: {cohort_summary['t_skin_std']:.2f} degC)\n"
        f"  * T_skin_meas range: [{cohort_summary['t_meas_min']:.2f}, {cohort_summary['t_meas_max']:.2f}] degC  (Mean: {cohort_summary['t_meas_mean']:.2f} degC, Std: {cohort_summary['t_meas_std']:.2f} degC)\n"
        f"  * Core - Skin Delta: [{cohort_summary['delta_core_skin_min']:.2f}, {cohort_summary['delta_core_skin_max']:.2f}] degC  (Mean: {cohort_summary['delta_core_skin_mean']:.2f} degC)"
    )
    print("=" * 115)

    # Save summary JSON
    out_json = os.path.join(os.path.dirname(__file__), "stage3_sim_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "cohort_summary": cohort_summary,
            "case_stats": all_case_stats
        }, f, indent=2)
    print(f"Saved Stage 3 simulation summary to: {out_json}")
    print("Execution complete. STOPPING HERE as instructed.")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 Bio-Heat Dynamic Simulation")
    parser.add_argument("--cases", type=int, nargs="+", default=None, help="Case IDs to simulate (default: 1..10)")
    args = parser.parse_args()
    run_stage3_simulation(case_ids=args.cases)


if __name__ == "__main__":
    main()
