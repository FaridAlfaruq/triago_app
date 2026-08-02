import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import CubicSpline

DATA_DIR = "draft_filter/bidmc"   # folder tempat file bidmc_XX_*.csv berada
OUTPUT_DIR = "draft_filter/output"

NUM_SUBJECTS = 52          # berapa banyak subjek yang mau diproses (10-20, bebas)
SUBJECT_START = 1          # subjek dimulai dari bidmc_01
SUBJECT_IDS = list(range(SUBJECT_START, SUBJECT_START + NUM_SUBJECTS))
# Atau tentukan manual, misal subjek tertentu saja:
# SUBJECT_IDS = [1, 3, 7, 12, 15]

FS = 125.0                 # Hz, tetap untuk seluruh dataset BIDMC
RR_TOLERANCE = 2.0         # bpm, toleransi dianggap "akurat" untuk RR
HR_TOLERANCE = 5.0         # bpm, toleransi dianggap "akurat" untuk HR

# bandpass
def bandpass(x, low_cut, high_cut, fs=FS, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    return filtfilt(b, a, x)

# r peak
def calculate_mean_slope(signal, peak_idx, window_samples):
    start_idx = max(0, peak_idx - window_samples)
    segment = signal[start_idx: peak_idx + 1]
    if len(segment) < 2:
        return 0.0
    return np.mean(np.abs(np.diff(segment)))


def calculate_meansb(r_peaks, all_peaks, current_peak_idx, signal):
    prev_qrs_vals = [signal[idx] for idx in r_peaks[-3:]] if r_peaks else []
    future_peaks = [idx for idx in all_peaks if idx > current_peak_idx]
    next_peak_vals = [signal[idx] for idx in future_peaks[:3]]
    combined_vals = prev_qrs_vals + [signal[current_peak_idx]] + next_peak_vals
    return np.mean(combined_vals) if combined_vals else signal[current_peak_idx]


def rpeak(signal, fs=FS):
    ecg = bandpass(signal, 5.0, 18.0, fs=fs)

    T = 1.0 / fs
    kernel = np.array([1, 2, 0, -2, -1]) * (1.0 / (8 * T))
    ecg_diff = np.convolve(ecg, kernel, mode='same')
    ecg_squared = ecg_diff ** 2

    win_duration = 0.060
    n_smooth = int(np.round(win_duration * fs))
    if n_smooth % 2 == 0:
        n_smooth += 1
    n = np.arange(n_smooth)
    psi = (2 * np.pi * n) / n_smooth
    a0, a1, a2, a3, a4 = (0.2155789, 0.4166316, 0.27726316, 0.08357895, 0.00694737)
    flatop_win = (a0 - a1 * np.cos(psi) + a2 * np.cos(2 * psi)
                  - a3 * np.cos(3 * psi) + a4 * np.cos(4 * psi))
    flatop_win = flatop_win / np.sum(flatop_win)
    ecg_smoothed = np.convolve(ecg_squared, flatop_win, mode='same')

    mwi_win = 0.150
    n_mwi = int(np.round(mwi_win * fs))
    kernel_mwi = np.ones(n_mwi) / n_mwi
    ecg_mwi = np.convolve(ecg_smoothed, kernel_mwi, mode='same')

    refractory_period = 0.231
    min_distance = int(np.round(refractory_period * fs))
    r_peaks_index, _ = find_peaks(ecg_mwi, distance=min_distance)

    init_duration = 2.0
    init_sample = int(np.round(init_duration * fs))
    mwi_init = ecg_mwi[:init_sample]

    MAXF = np.max(mwi_init) if len(mwi_init) > 0 else np.max(ecg_mwi)
    MEANF = np.mean(mwi_init) if len(mwi_init) > 0 else np.mean(ecg_mwi)

    Threshold1 = MAXF / 3.0
    Threshold2 = 0.5 * MEANF

    spk = Threshold1
    npk = Threshold2
    th1 = Threshold1
    th2 = Threshold2

    N_70 = int(np.round(0.070 * fs))

    r_peaks = []
    noise_peaks = []

    for peak_idx in r_peaks_index:
        peak_val = ecg_mwi[peak_idx]

        if len(r_peaks) > 0:
            last_r_idx = r_peaks[-1]
            current_rr_sec = (peak_idx - last_r_idx) / fs
            if len(r_peaks) >= 2:
                recent_r = r_peaks[-8:]
                mean_rr_sec = np.mean(np.diff(recent_r) / fs)
            else:
                mean_rr_sec = current_rr_sec
        else:
            current_rr_sec = 0.0
            mean_rr_sec = 0.0

        is_classified = False

        if peak_val > th1:
            is_r_peak = True
            if len(r_peaks) > 0 and (current_rr_sec < 0.360 or current_rr_sec < (0.5 * mean_rr_sec)):
                slope_curr = calculate_mean_slope(ecg_mwi, peak_idx, N_70)
                slope_prev = calculate_mean_slope(ecg_mwi, last_r_idx, N_70)
                if slope_curr < 0.60 * slope_prev:
                    is_r_peak = False

            if is_r_peak:
                r_peaks.append(peak_idx)
                spk = 0.125 * peak_val + 0.875 * spk
                is_classified = True
            else:
                noise_peaks.append(peak_idx)
                npk = 0.125 * peak_val + 0.875 * npk
                is_classified = True

        elif len(r_peaks) > 0 and (current_rr_sec > 1.0 or current_rr_sec > (1.66 * mean_rr_sec)):
            meansb = calculate_meansb(r_peaks, r_peaks_index, peak_idx, ecg_mwi)
            th3 = 0.5 * th2 + 0.5 * meansb
            if peak_val > th3:
                r_peaks.append(peak_idx)
                spk = 0.75 * peak_val + 0.25 * spk
                is_classified = True

        if not is_classified and len(r_peaks) > 0 and current_rr_sec > 1.4:
            if peak_val > (0.2 * th2):
                r_peaks.append(peak_idx)
                spk = 0.75 * peak_val + 0.25 * spk
                is_classified = True

        if not is_classified:
            noise_peaks.append(peak_idx)
            npk = 0.125 * peak_val + 0.875 * npk

        th1 = npk + 0.25 * (spk - npk)
        th2 = 0.4 * th1

    return r_peaks, noise_peaks


# rr
def rr(ecg, r_peaks, fs=FS):
    n_samples = len(ecg)
    total_seconds = n_samples / fs
    r_peaks = np.array(r_peaks)

    if len(r_peaks) < 4:
        return 0.0, np.zeros(n_samples), np.array([])

    T = 1.0 / fs
    kernel_diff = np.array([1, 2, 0, -2, -1]) * (1.0 / (8 * T))
    ecg_diff = np.convolve(ecg, kernel_diff, mode='same')
    win_samples = int(np.round(0.040 * fs))
    edr_samples = []

    for idx in r_peaks:
        t0 = max(0, idx - win_samples)
        t1 = min(n_samples, idx + win_samples + 1)
        max_slope = np.max(np.abs(ecg_diff[t0:t1]))
        edr_samples.append(max_slope)

    edr_samples = np.array(edr_samples)
    xm = 0.0
    xd = 1.0
    xc = 0

    r_signals = []
    for x in edr_samples:
        d = x - xm
        if xc < 500:
            xc += 1
            dn = d / xc
        else:
            dn = d / xc
            xdmax = 3.0 * xd / xc
            dn = np.clip(dn, -xdmax, xdmax)

        xm += dn
        xd += abs(dn) - xd / xc
        if xd < 1e-3:
            xd = 1e-3

        r = d / xd
        r_signals.append(r)

    r_signals = np.array(r_signals)
    cs = CubicSpline(r_peaks, r_signals, extrapolate=False)

    interp_indices = np.arange(r_peaks[0], r_peaks[-1] + 1)
    resp_signal = cs(interp_indices)

    resp_signal_full = np.zeros(n_samples)
    resp_signal_full[r_peaks[0]:r_peaks[-1] + 1] = resp_signal
    resp_signal_full[:r_peaks[0]] = resp_signal[0]
    resp_signal_full[r_peaks[-1] + 1:] = resp_signal[-1]

    min_distance_resp = int(np.round(1.6 * fs))
    resp_peaks, _ = find_peaks(
        resp_signal_full,
        distance=min_distance_resp,
        height=np.mean(resp_signal_full)
    )

    b = len(resp_peaks)
    resp_rate = (b / (total_seconds / 60.0)) if total_seconds > 0 else 0.0

    return resp_rate, resp_signal_full, resp_peaks


# load data + proses
def load_subject(subject_id, data_dir=DATA_DIR):
    sid = f"{subject_id:02d}"
    sig_path = os.path.join(data_dir, f"bidmc_{sid}_Signals.csv")
    num_path = os.path.join(data_dir, f"bidmc_{sid}_Numerics.csv")

    if not (os.path.exists(sig_path) and os.path.exists(num_path)):
        return None, None

    sig = pd.read_csv(sig_path, skipinitialspace=True)
    num = pd.read_csv(num_path, skipinitialspace=True)
    return sig, num


def process_subject(subject_id, data_dir=DATA_DIR, fs=FS):
    sig, num = load_subject(subject_id, data_dir)
    if sig is None:
        return None

    ecg = sig['II'].values.astype(float)

    r_peaks, noise_peaks = rpeak(ecg, fs)

    if len(r_peaks) < 4:
        hr_est, rr_est = np.nan, np.nan
    else:
        hr_est = 60.0 / np.mean(np.diff(r_peaks) / fs)
        rr_est, _, _ = rr(ecg, r_peaks, fs)

    # -- Ground truth (rata-rata sepanjang rekaman) --
    hr_gt = num['HR'].mean()
    rr_gt = num['RESP'].mean()

    return {
        "Subject": f"bidmc_{subject_id:02d}",
        "n_Rpeaks": len(r_peaks),
        "HR_est": hr_est,
        "HR_gt": hr_gt,
        "HR_err": (hr_est - hr_gt) if not np.isnan(hr_est) else np.nan,
        "RR_est": rr_est,
        "RR_gt": rr_gt,
        "RR_err": (rr_est - rr_gt) if not np.isnan(rr_est) else np.nan,
    }


# loop
def evaluate_all(subject_ids=SUBJECT_IDS, data_dir=DATA_DIR):
    results = []
    skipped = []

    for sid in subject_ids:
        row = process_subject(sid, data_dir)
        if row is None:
            skipped.append(f"bidmc_{sid:02d}")
            continue
        results.append(row)

    if skipped:
        print(f"[!] File tidak ditemukan, dilewati: {', '.join(skipped)}")
        print(f"    (pastikan bidmc_XX_Signals.csv & bidmc_XX_Numerics.csv ada di {data_dir})\n")

    if not results:
        print("Tidak ada subjek yang berhasil diproses.")
        return None

    df = pd.DataFrame(results)

    # kolom persentase error & flag akurat/tidak
    df["RR_abs_err"] = df["RR_err"].abs()
    df["HR_abs_err"] = df["HR_err"].abs()
    df["RR_pct_err"] = 100 * df["RR_abs_err"] / df["RR_gt"]
    df["HR_pct_err"] = 100 * df["HR_abs_err"] / df["HR_gt"]
    df["RR_ok"] = df["RR_abs_err"] <= RR_TOLERANCE
    df["HR_ok"] = df["HR_abs_err"] <= HR_TOLERANCE

    # akurasi per-subjek (100% - error relatif), diclip biar gak minus kalau errornya gede
    df["RR_acc_pct"] = (100 - df["RR_pct_err"]).clip(lower=0)
    df["HR_acc_pct"] = (100 - df["HR_pct_err"]).clip(lower=0)

    return df


def print_report(df):
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")

    print("=" * 92)
    print(" TABEL PERBANDINGAN RESPIRATION RATE (RR) & HEART RATE (HR)")
    print("=" * 92)
    table = df[["Subject", "n_Rpeaks",
                "HR_est", "HR_gt", "HR_abs_err", "HR_acc_pct",
                "RR_est", "RR_gt", "RR_abs_err", "RR_acc_pct"]].copy()
    table = table.rename(columns={
        "HR_abs_err": "HR_MAE", "HR_acc_pct": "HR_Akurasi(%)",
        "RR_abs_err": "RR_MAE", "RR_acc_pct": "RR_Akurasi(%)",
    })
    print(table.to_string(index=False))
    print("-" * 92)

    n = len(df)
    rr_mae = df["RR_abs_err"].mean()
    rr_rmse = np.sqrt((df["RR_err"] ** 2).mean())
    rr_mape = df["RR_pct_err"].mean()
    rr_acc = 100 - rr_mape
    rr_within_tol = 100 * df["RR_ok"].sum() / n

    hr_mae = df["HR_abs_err"].mean()
    hr_rmse = np.sqrt((df["HR_err"] ** 2).mean())
    hr_mape = df["HR_pct_err"].mean()
    hr_acc = 100 - hr_mape
    hr_within_tol = 100 * df["HR_ok"].sum() / n

    print(f"Jumlah subjek dievaluasi : {n}")
    print()
    print(f"RESPIRATION RATE (RR)")
    print(f"  MAE                    : {rr_mae:.2f} bpm")
    print(f"  RMSE                   : {rr_rmse:.2f} bpm")
    print(f"  MAPE                   : {rr_mape:.2f} %")
    print(f"  Akurasi (100-MAPE)     : {rr_acc:.2f} %")
    print(f"  % subjek dlm toleransi (\u00b1{RR_TOLERANCE} bpm) : {rr_within_tol:.1f} %")
    print()
    print(f"HEART RATE (HR)")
    print(f"  MAE                    : {hr_mae:.2f} bpm")
    print(f"  RMSE                   : {hr_rmse:.2f} bpm")
    print(f"  MAPE                   : {hr_mape:.2f} %")
    print(f"  Akurasi (100-MAPE)     : {hr_acc:.2f} %")
    print(f"  % subjek dlm toleransi (\u00b1{HR_TOLERANCE} bpm) : {hr_within_tol:.1f} %")
    print("=" * 92)


if __name__ == "__main__":
    df_results = evaluate_all(SUBJECT_IDS, DATA_DIR)
    if df_results is not None:
        print_report(df_results)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "bidmc_rr_hr_results.csv")
        df_results.to_csv(out_path, index=False)
        print(f"\nHasil lengkap disimpan di: {out_path}")
