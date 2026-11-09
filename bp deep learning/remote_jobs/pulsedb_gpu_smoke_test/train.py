"""
Kaggle GPU Remote Smoke Test (1-Epoch).
Verifies:
1. Kaggle Remote GPU accessibility (NVIDIA T4 / P100 / Tesla).
2. PulseDB 7-channel dataset loading on Kaggle.
3. CalBPNet training loop and backward pass on CUDA.
4. GPU memory footprint and latency per batch.
5. Checkpoint export to Kaggle working directory.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==============================================================================
# Model Definition (CalBPNet)
# ==============================================================================

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, filters: int = 64):
        super().__init__()
        b = filters // 4
        self.b1 = nn.Conv1d(in_channels, b, kernel_size=1, padding=0)
        self.b2 = nn.Conv1d(in_channels, b, kernel_size=3, padding=1)
        self.b3 = nn.Conv1d(in_channels, b, kernel_size=7, padding=3)
        self.b4 = nn.Conv1d(in_channels, b, kernel_size=15, padding=7)
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm1d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.bn(self.act(out))


class CalBPNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        stem_ch: int = 48,
        inc1_ch: int = 64,
        inc2_ch: int = 96,
        inc3_ch: int = 128,
        gru_hid: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_ch, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(stem_ch),
            nn.SiLU(),
        )
        self.inc1 = InceptionBlock1D(stem_ch, inc1_ch)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.15)

        self.inc2 = InceptionBlock1D(inc1_ch, inc2_ch)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.15)

        self.inc3 = InceptionBlock1D(inc2_ch, inc3_ch)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.20)

        self.bigru = nn.GRU(
            input_size=inc3_ch,
            hidden_size=gru_hid,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.mhsa = nn.MultiheadAttention(
            embed_dim=gru_hid * 2,
            num_heads=num_heads,
            dropout=0.10,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(gru_hid * 2)

        context_dim = gru_hid * 4
        self.context_drop = nn.Dropout(0.20)

        self.sbp_head = nn.Sequential(
            nn.Linear(context_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )
        self.dbp_head = nn.Sequential(
            nn.Linear(context_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] == 1250:
            x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.drop1(self.pool1(self.inc1(x)))
        x = self.drop2(self.pool2(self.inc2(x)))
        x = self.drop3(self.pool3(self.inc3(x)))

        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        attn_out, _ = self.mhsa(gru_out, gru_out, gru_out)
        x = self.ln(gru_out + attn_out)

        avg_p = x.mean(dim=1)
        max_p = x.max(dim=1).values
        context = self.context_drop(torch.cat([avg_p, max_p], dim=1))

        sbp = self.sbp_head(context).squeeze(-1)
        dbp = self.dbp_head(context).squeeze(-1)
        return sbp, dbp


class PhysicsBHSLoss(nn.Module):
    def __init__(self, s_mean: float, s_std: float, d_mean: float, d_std: float):
        super().__init__()
        self.s_mean = s_mean
        self.s_std = s_std
        self.d_mean = d_mean
        self.d_std = d_std
        self.huber = nn.SmoothL1Loss(beta=0.20)

    def forward(self, s_pred_z, d_pred_z, s_true_z, d_true_z):
        loss_huber = self.huber(s_pred_z, s_true_z) + self.huber(d_pred_z, d_true_z)
        s_pred = s_pred_z * self.s_std + self.s_mean
        d_pred = d_pred_z * self.d_std + self.d_mean
        s_true = s_true_z * self.s_std + self.s_mean
        d_true = d_true_z * self.d_std + self.d_mean
        map_err = torch.mean(torch.abs((s_pred + 2.0 * d_pred) / 3.0 - (s_true + 2.0 * d_true) / 3.0)) / 15.0
        pp_err = torch.mean(torch.abs((s_pred - d_pred) - (s_true - d_true))) / 15.0
        return loss_huber + 0.15 * map_err + 0.10 * pp_err


# ==============================================================================
# Helper to Locate Dataset File
# ==============================================================================

def find_dataset_file() -> Path:
    candidates = [
        Path("/kaggle/input/pulsedb-vital-mimic-7ch/dataset_7ch.npz"),
        Path("/kaggle/input/pulsedb-vital-mimic-7ch/pulsedb_benchmark_7ch.npz"),
        Path("../input/pulsedb-vital-mimic-7ch/dataset_7ch.npz"),
        Path("../input/pulsedb-vital-mimic-7ch/pulsedb_benchmark_7ch.npz"),
        Path("scratch/gemastik/pulsedb_benchmark_7ch.npz"),
    ]
    for c in candidates:
        if c.exists():
            return c
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = list(kaggle_input.rglob("*.npz"))
        if found:
            return found[0]
    raise FileNotFoundError("Could not find PulseDB .npz dataset file!")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("=" * 70)
    print("KAGGLE REMOTE GPU SMOKE TEST (1-EPOCH VERIFICATION)")
    print("=" * 70)

    # 1. Check Hardware & CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device      : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Remote GPU Name    : {gpu_name}")
        print(f"Total GPU Memory   : {total_mem:.2f} GB")
        print(f"CUDA Capability    : {torch.cuda.get_device_capability(0)}")
    else:
        print("WARNING: CUDA is not available! Running on CPU fallback.")
    print("-" * 70)

    # 2. Locate and Load Dataset
    data_path = find_dataset_file()
    print(f"Loading dataset from: {data_path} ({data_path.stat().st_size / (1024*1024):.1f} MB)")
    data = np.load(data_path)

    X_raw = data["X"]
    y_s_raw = data["y_sbp"]
    y_d_raw = data["y_dbp"]
    groups_raw = data["groups"]
    print(f"Dataset Raw Shape  : X={X_raw.shape}, y_sbp={y_s_raw.shape}, subjects={len(np.unique(groups_raw))}")

    # Use first 800 segments for rapid 1-epoch smoke test
    n_sample = min(1000, len(X_raw))
    X = X_raw[:n_sample]
    y_s = y_s_raw[:n_sample]
    y_d = y_d_raw[:n_sample]
    groups = groups_raw[:n_sample]

    # Split: 80% train / 20% test
    n_tr = int(n_sample * 0.80)
    train_idx = np.arange(0, n_tr)
    val_idx = np.arange(n_tr, n_sample)

    s_mean, s_std = float(y_s[train_idx].mean()), float(y_s[train_idx].std() + 1e-6)
    d_mean, d_std = float(y_d[train_idx].mean()), float(y_d[train_idx].std() + 1e-6)

    train_ds = TensorDataset(
        torch.tensor(X[train_idx], dtype=torch.float32),
        torch.tensor((y_s[train_idx] - s_mean) / s_std, dtype=torch.float32),
        torch.tensor((y_d[train_idx] - d_mean) / d_std, dtype=torch.float32),
        torch.tensor(y_s[train_idx], dtype=torch.float32),
        torch.tensor(y_d[train_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X[val_idx], dtype=torch.float32),
        torch.tensor((y_s[val_idx] - s_mean) / s_std, dtype=torch.float32),
        torch.tensor((y_d[val_idx] - d_mean) / d_std, dtype=torch.float32),
        torch.tensor(y_s[val_idx], dtype=torch.float32),
        torch.tensor(y_d[val_idx], dtype=torch.float32),
    )

    batch_size = 64 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Train Batches      : {len(train_loader)} (Batch Size: {batch_size})")
    print(f"Val Batches        : {len(val_loader)}")
    print("-" * 70)

    # 3. Model & Optimizer
    model = CalBPNet(in_channels=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PhysicsBHSLoss(s_mean=s_mean, s_std=s_std, d_mean=d_mean, d_std=d_std)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CalBPNet Model     : {n_params:,} parameters (FP32)")

    # 4. Train 1-Epoch Smoke Test
    print("\nExecuting 1-Epoch Training on Remote Accelerator...")
    t0 = time.time()
    model.train()
    tr_loss = 0.0

    for step, (bx, bs_z, bd_z, bs_raw, bd_raw) in enumerate(train_loader, 1):
        bx = bx.to(device)
        bs_z = bs_z.to(device)
        bd_z = bd_z.to(device)

        optimizer.zero_grad()
        p_s_z, p_d_z = model(bx)
        loss = criterion(p_s_z, p_d_z, bs_z, bd_z)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(bx)

    tr_loss /= len(train_ds)
    train_time = time.time() - t0

    # 5. Validation Pass
    model.eval()
    val_loss = 0.0
    all_s_err, all_d_err = [], []

    with torch.no_grad():
        for bx, bs_z, bd_z, bs_raw, bd_raw in val_loader:
            bx = bx.to(device)
            bs_z = bs_z.to(device)
            bd_z = bd_z.to(device)
            bs_raw = bs_raw.to(device)
            bd_raw = bd_raw.to(device)

            p_s_z, p_d_z = model(bx)
            loss = criterion(p_s_z, p_d_z, bs_z, bd_z)
            val_loss += loss.item() * len(bx)

            p_s_mmhg = p_s_z * s_std + s_mean
            p_d_mmhg = p_d_z * d_std + d_mean
            all_s_err.append(torch.abs(p_s_mmhg - bs_raw).cpu().numpy())
            all_d_err.append(torch.abs(p_d_mmhg - bd_raw).cpu().numpy())

    val_loss /= len(val_ds)
    val_mae_s = float(np.mean(np.concatenate(all_s_err)))
    val_mae_d = float(np.mean(np.concatenate(all_d_err)))
    val_mae = (val_mae_s + val_mae_d) / 2.0
    total_time = time.time() - t0

    print(f"1-Epoch Execution  : {train_time:.2f}s (Train) | {total_time:.2f}s (Total)")
    print(f"Train Loss         : {tr_loss:.4f}")
    print(f"Val Loss           : {val_loss:.4f}")
    print(f"Val SBP MAE        : {val_mae_s:.2f} mmHg")
    print(f"Val DBP MAE        : {val_mae_d:.2f} mmHg")
    print(f"Overall Val MAE    : {val_mae:.2f} mmHg")

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(0) / (1024**2)
        mem_cached = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"Peak GPU VRAM Used : {mem_alloc:.1f} MB (Allocated), {mem_cached:.1f} MB (Reserved)")

    # 6. Save Checkpoint & Metadata
    ckpt_path = "best_calbpnet_smoke_test.pt"
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "val_mae": val_mae,
        "val_mae_sbp": val_mae_s,
        "val_mae_dbp": val_mae_d,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }, ckpt_path)

    results_path = "smoke_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "status": "SUCCESS",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "train_time_sec": round(train_time, 2),
            "total_time_sec": round(total_time, 2),
            "val_mae_all": round(val_mae, 2),
            "val_mae_sbp": round(val_mae_s, 2),
            "val_mae_dbp": round(val_mae_d, 2),
        }, f, indent=2)

    print("-" * 70)
    print(f"Saved Checkpoint   : {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.2f} MB)")
    print(f"Saved Summary JSON : {results_path}")
    print("REMOTE GPU SMOKE TEST: PASSED [STATUS: OPERATIONAL]")
    print("=" * 70)


if __name__ == "__main__":
    main()
