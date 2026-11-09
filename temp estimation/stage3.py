"""
stage3.py - Stage 3: Bio-Heat Dynamic Skin Temperature & MLX90614 Sensor Simulation
Author: Alex (Lead Orchestrator)
Master Reference: scratch/temp estimation/PRD_VitalDB_Temp_Validation.md

Physics & Formulation:
1. Two-Node Core-to-Skin Equilibrium (Pennes 1948, Stolwijk-Hardy 1966, Gagge 1971):
   Q_core_to_skin = (K_tissue + c_perf * PI^alpha) * (T_core - T_skin)
   Q_skin_to_amb  = h_total * (T_skin - T_amb)
   At steady-state equilibrium:
   T_eq = ((a + b * PI_eff^alpha) * T_core + T_amb) / (1 + a + b * PI_eff^alpha)
   where:
     a = K_tissue / h_total = 5.28 / 9.2 = 0.5739 (~0.574)
     b = c_perf / h_total = 0.850
     alpha = 0.800
     PI_eff = PI / 10.0 (maps VitalDB photodiode ratio to clinical 0.1 - 10.0% range)

2. Dynamic Thermal Inertia / First-Order Lag (Wissler 1985):
   dT_skin/dt = (1 / tau) * (T_eq(t) - T_skin(t)), tau = 180.0 s
   Discrete exponential integration:
   T_skin[k] = T_skin[k-1] * exp(-dt / tau) + T_eq[k] * (1 - exp(-dt / tau))

3. MLX90614 Medical-Grade Infrared Sensor Realism:
   - Ambient hospital temperature: T_amb ~ N(21.0, 0.3^2) degC
   - Noise Equivalent Temperature Difference (NETD): epsilon_sensor ~ N(0, 0.05^2) degC
   - Digital resolution: 17-bit on-chip ADC quantization step of 0.02 degC:
     T_skin_meas = round((T_skin + epsilon_sensor) / 0.02) * 0.02
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Link to Stage 2 DSP pipeline
from stage2 import process_case_dsp

# -----------------------------------------------------------------------------
# Biophysical & Sensor Default Constants (PRD Sections 2 & 3)
# -----------------------------------------------------------------------------
A_DEFAULT = 0.574          # Conductance ratio: K_tissue / h_total (5.28 / 9.2)
B_DEFAULT = 0.850          # Perfusion coefficient ratio: c_perf / h_total
ALPHA_DEFAULT = 0.800      # Nonlinear perfusion scaling power
TAU_DEFAULT = 180.0        # Thermal lag time constant in seconds
PI_SCALE_DEFAULT = 0.10    # Maps VitalDB optical counts ratio to clinical %
T_AMB_MEAN_DEFAULT = 21.0  # Mean hospital operating room temperature (degC)
T_AMB_STD_DEFAULT = 0.30   # Hospital ambient temperature fluctuation (degC)
SENSOR_NETD_DEFAULT = 0.05 # MLX90614 NETD Gaussian standard deviation (degC)
ADC_RESOLUTION_DEFAULT = 0.02 # MLX90614 17-bit ADC step (degC)


def compute_equilibrium_temperature(
    t_core: np.ndarray,
    pi: np.ndarray,
    t_amb: np.ndarray,
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    pi_scale: float = PI_SCALE_DEFAULT
) -> np.ndarray:
    """
    Computes steady-state equilibrium skin temperature T_eq from T_core, PI, and T_amb.
    Formula:
      G_cs = a + b * (PI * pi_scale)^alpha
      T_eq = (G_cs * T_core + T_amb) / (1.0 + G_cs)
    """
    pi_eff = np.maximum(pi * pi_scale, 1e-4)
    g_cs = a + b * np.power(pi_eff, alpha)
    t_eq = (g_cs * t_core + t_amb) / (1.0 + g_cs)
    return t_eq


def simulate_skin_temperature_ode(
    t_sec: np.ndarray,
    t_eq: np.ndarray,
    tau: float = TAU_DEFAULT,
    dt_max: float = 60.0
) -> np.ndarray:
    """
    Simulates dynamic skin temperature by solving first-order ODE with thermal time constant tau:
      dT_skin/dt = (1 / tau) * (T_eq(t) - T_skin(t))
    Using exact exponential step:
      T_skin[k] = T_skin[k-1] * exp(-dt / tau) + T_eq[k] * (1 - exp(-dt / tau))
    """
    n = len(t_sec)
    t_skin = np.zeros(n, dtype=np.float64)
    if n == 0:
        return t_skin

    t_skin[0] = t_eq[0]
    for k in range(1, n):
        dt = t_sec[k] - t_sec[k - 1]
        if dt <= 0.0 or dt > dt_max:
            # New epoch or large time gap: reset to steady-state equilibrium
            t_skin[k] = t_eq[k]
        else:
            decay = np.exp(-dt / tau)
            t_skin[k] = t_skin[k - 1] * decay + t_eq[k] * (1.0 - decay)

    return t_skin


def apply_mlx90614_noise(
    t_skin: np.ndarray,
    sigma_sensor: float = SENSOR_NETD_DEFAULT,
    adc_step: float = ADC_RESOLUTION_DEFAULT,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Applies Melexis MLX90614 medical infrared thermopile sensor physics:
    1. Noise Equivalent Temperature Difference (NETD): Gaussian noise ~ N(0, sigma_sensor^2)
    2. 17-bit on-chip ADC digital quantization: step size = 0.02 degC
    """
    if rng is None:
        rng = np.random.default_rng(42)

    noise = rng.normal(loc=0.0, scale=sigma_sensor, size=len(t_skin))
    t_continuous = t_skin + noise
    t_quantized = np.round(t_continuous / adc_step) * adc_step
    return t_quantized


def process_case_stage3(
    caseid: int,
    a: float = A_DEFAULT,
    b: float = B_DEFAULT,
    alpha: float = ALPHA_DEFAULT,
    tau: float = TAU_DEFAULT,
    pi_scale: float = PI_SCALE_DEFAULT,
    t_amb_mean: float = T_AMB_MEAN_DEFAULT,
    t_amb_std: float = T_AMB_STD_DEFAULT,
    sigma_sensor: float = SENSOR_NETD_DEFAULT,
    adc_step: float = ADC_RESOLUTION_DEFAULT,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs Stage 3 dynamic skin temperature synthesis on a single VitalDB surgical case.
    1. Loads DSP epochs from Stage 2.
    2. Filters for valid SQI and physiological T_core.
    3. Solves dynamic bio-heat ODE segment-by-segment.
    4. Simulates MLX90614 sensor noise and digital quantization.
    """
    rng = np.random.default_rng(seed + caseid)

    # 1. Load Stage 2 DSP epochs
    dsp_res = process_case_dsp(caseid)
    if dsp_res.get("status") != "OK":
        return {
            "caseid": caseid,
            "status": f"STAGE2_ERROR: {dsp_res.get('status')}",
            "df_sim": pd.DataFrame()
        }

    df_epochs = dsp_res["df_epochs"]
    valid_mask = (
        df_epochs["sqi_passed"] &
        df_epochs["t_core"].notna() &
        df_epochs["pi"].notna() &
        (df_epochs["t_core"] >= 33.0) &
        (df_epochs["t_core"] <= 45.0) &
        (df_epochs["pi"] > 0.0)
    )

    df_valid = df_epochs[valid_mask].copy().reset_index(drop=True)
    if len(df_valid) == 0:
        return {
            "caseid": caseid,
            "status": "NO_VALID_EPOCHS",
            "df_sim": pd.DataFrame()
        }

    # 2. Process segment by segment
    sim_rows = []
    for seg_id, group in df_valid.groupby("seg_id"):
        group = group.sort_values("t_center_sec").reset_index(drop=True)
        t_sec = group["t_center_sec"].values
        t_core = group["t_core"].values
        pi_vals = group["pi"].values
        n_epochs = len(group)

        # Ambient temperature time-series
        t_amb = rng.normal(loc=t_amb_mean, scale=t_amb_std, size=n_epochs)

        # Steady-state equilibrium temperature
        t_eq = compute_equilibrium_temperature(
            t_core=t_core,
            pi=pi_vals,
            t_amb=t_amb,
            a=a,
            b=b,
            alpha=alpha,
            pi_scale=pi_scale
        )

        # Dynamic thermal lag ODE
        t_skin = simulate_skin_temperature_ode(
            t_sec=t_sec,
            t_eq=t_eq,
            tau=tau,
            dt_max=60.0
        )

        # MLX90614 sensor measurement simulation
        t_skin_meas = apply_mlx90614_noise(
            t_skin=t_skin,
            sigma_sensor=sigma_sensor,
            adc_step=adc_step,
            rng=rng
        )

        for i in range(n_epochs):
            sim_rows.append({
                "caseid": caseid,
                "seg_id": seg_id,
                "window_idx": int(group["window_idx"].iloc[i]),
                "t_center_sec": round(float(t_sec[i]), 2),
                "t_center_min": round(float(t_sec[i]) / 60.0, 2),
                "t_core": round(float(t_core[i]), 2),
                "ac": round(float(group["ac"].iloc[i]), 3),
                "dc": round(float(group["dc"].iloc[i]), 3),
                "pi": round(float(pi_vals[i]), 2),
                "t_amb": round(float(t_amb[i]), 3),
                "t_eq": round(float(t_eq[i]), 3),
                "t_skin": round(float(t_skin[i]), 3),
                "t_skin_meas": round(float(t_skin_meas[i]), 2)
            })

    df_sim = pd.DataFrame(sim_rows)

    # Calculate statistics
    t_skin_arr = df_sim["t_skin"].values
    t_meas_arr = df_sim["t_skin_meas"].values
    t_core_arr = df_sim["t_core"].values
    pi_arr = df_sim["pi"].values

    stats = {
        "caseid": caseid,
        "status": "OK",
        "n_epochs": len(df_sim),
        "valid_duration_min": round(len(df_sim) * 10.0 / 60.0, 2),
        "t_core_mean": round(float(np.mean(t_core_arr)), 2),
        "t_core_min": round(float(np.min(t_core_arr)), 2),
        "t_core_max": round(float(np.max(t_core_arr)), 2),
        "t_skin_mean": round(float(np.mean(t_skin_arr)), 2),
        "t_skin_min": round(float(np.min(t_skin_arr)), 2),
        "t_skin_max": round(float(np.max(t_skin_arr)), 2),
        "t_skin_std": round(float(np.std(t_skin_arr)), 2),
        "t_meas_mean": round(float(np.mean(t_meas_arr)), 2),
        "t_meas_min": round(float(np.min(t_meas_arr)), 2),
        "t_meas_max": round(float(np.max(t_meas_arr)), 2),
        "t_meas_std": round(float(np.std(t_meas_arr)), 2),
        "delta_core_skin_mean": round(float(np.mean(t_core_arr - t_skin_arr)), 2),
        "pi_mean": round(float(np.mean(pi_arr)), 2)
    }

    return {
        "stats": stats,
        "df_sim": df_sim
    }
