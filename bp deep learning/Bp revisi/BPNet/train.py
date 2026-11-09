"""
train.py
Training module for CalBPNet with dual partitioning protocols (CalFree & CalBased).

Features:
1. Two Split Protocols:
   - CalFree (Subject-Wise): Unseen subjects in validation (pure generalization).
   - CalBased (Within-Subject): Within-subject 90/10 split with calibration anchors.
2. Metrics Tracking (Saved every epoch for Train & Val):
   - Total Loss
   - Overall MAE (mmHg)
   - SBP MAE (mmHg)
   - DBP MAE (mmHg)
3. Best Checkpoint Saving:
   - Automatically saves model weights when validation MAE improves.
4. History Export:
   - Saves complete training history to JSON.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from load import load_raw_subject, list_available_subjects
from signal_processing import preprocess_subject_signals
from data_split import split_dataset
from models import CalBPNet, PhysicsBHSLoss


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def load_cohort_dataset(
    num_subjects: int = 5,
    max_segments_per_subj: int = 10,
    source: str = "PulseDB_Vital",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw PulseDB subjects and preprocess through Tight SQI filter.
    Returns:
        X: Preprocessed 7-channel tensors, shape (N_total, 1250, 7)
        y_sbp: Ground truth SBP (mmHg), shape (N_total,)
        y_dbp: Ground truth DBP (mmHg), shape (N_total,)
        groups: Subject identifier per segment, shape (N_total,)
    """
    avail = list_available_subjects(source=source)[:num_subjects]
    all_X, all_sbp, all_dbp, all_g = [], [], [], []

    for s in avail:
        e, p, a, sbp, dbp = load_raw_subject(s, max_segments=max_segments_per_subj, source=source)
        Xc, sc, dc, mask = preprocess_subject_signals(e, p, sbp, dbp, fs=125)
        if len(Xc) > 0:
            all_X.append(Xc)
            all_sbp.append(sc)
            all_dbp.append(dc)
            all_g.extend([s] * len(Xc))

    if len(all_X) == 0:
        raise ValueError("No clean segments found across loaded subjects.")

    X = np.concatenate(all_X, axis=0)
    y_sbp = np.concatenate(all_sbp, axis=0)
    y_dbp = np.concatenate(all_dbp, axis=0)
    groups = np.array(all_g)
    return X, y_sbp, y_dbp, groups


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    s_mean: float,
    s_std: float,
    d_mean: float,
    d_std: float,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Train for a single epoch. Returns (loss, mae_all, mae_sbp, mae_dbp)."""
    model.train()
    total_loss = 0.0
    all_sbp_err, all_dbp_err = [], []

    for bx, bs_z, bd_z, bs_raw, bd_raw in loader:
        bx = bx.to(device)
        bs_z = bs_z.to(device)
        bd_z = bd_z.to(device)
        bs_raw = bs_raw.to(device)
        bd_raw = bd_raw.to(device)

        optimizer.zero_grad()
        p_s_z, p_d_z = model(bx)
        loss = criterion(p_s_z, p_d_z, bs_z, bd_z)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(bx)

        # Denormalize predictions to mmHg for MAE calculation
        p_sbp_mmhg = p_s_z * s_std + s_mean
        p_dbp_mmhg = p_d_z * d_std + d_mean

        all_sbp_err.append(torch.abs(p_sbp_mmhg - bs_raw).detach().cpu().numpy())
        all_dbp_err.append(torch.abs(p_dbp_mmhg - bd_raw).detach().cpu().numpy())

    n_samples = len(loader.dataset)
    epoch_loss = total_loss / n_samples
    mae_sbp = float(np.mean(np.concatenate(all_sbp_err)))
    mae_dbp = float(np.mean(np.concatenate(all_dbp_err)))
    mae_all = (mae_sbp + mae_dbp) / 2.0

    return epoch_loss, mae_all, mae_sbp, mae_dbp


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    s_mean: float,
    s_std: float,
    d_mean: float,
    d_std: float,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluate on validation set. Returns (loss, mae_all, mae_sbp, mae_dbp)."""
    model.eval()
    total_loss = 0.0
    all_sbp_err, all_dbp_err = [], []

    with torch.no_grad():
        for bx, bs_z, bd_z, bs_raw, bd_raw in loader:
            bx = bx.to(device)
            bs_z = bs_z.to(device)
            bd_z = bd_z.to(device)
            bs_raw = bs_raw.to(device)
            bd_raw = bd_raw.to(device)

            p_s_z, p_d_z = model(bx)
            loss = criterion(p_s_z, p_d_z, bs_z, bd_z)
            total_loss += loss.item() * len(bx)

            p_sbp_mmhg = p_s_z * s_std + s_mean
            p_dbp_mmhg = p_d_z * d_std + d_mean

            all_sbp_err.append(torch.abs(p_sbp_mmhg - bs_raw).cpu().numpy())
            all_dbp_err.append(torch.abs(p_dbp_mmhg - bd_raw).cpu().numpy())

    n_samples = len(loader.dataset)
    val_loss = total_loss / n_samples
    mae_sbp = float(np.mean(np.concatenate(all_sbp_err)))
    mae_dbp = float(np.mean(np.concatenate(all_dbp_err)))
    mae_all = (mae_sbp + mae_dbp) / 2.0

    return val_loss, mae_all, mae_sbp, mae_dbp


def run_training(
    protocol: str = "calfree",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_subjects: int = 5,
    source: str = "PulseDB_Vital",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute training loop under specified protocol (CalFree or CalBased).
    Saves best model weights and complete epoch-by-epoch training history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"best_calbpnet_{protocol}.pt"
    history_path = CHECKPOINT_DIR / f"history_{protocol}.json"

    print("=" * 70)
    print(f"TRAINING CALBPNET | PROTOCOL: {protocol.upper()} | DEVICE: {device}")
    print("=" * 70)

    # 1. Load Data
    X, y_sbp, y_dbp, groups = load_cohort_dataset(
        num_subjects=num_subjects, source=source
    )

    # 2. Partition
    train_idx, val_idx, meta = split_dataset(groups, protocol=protocol, seed=seed)
    n_train = len(train_idx)
    n_val = len(val_idx)

    print(f"Data Cohort : {len(np.unique(groups))} subjects | {len(groups)} total clean segments")
    print(f"Train Set   : {len(meta['train_subjects'])} subjects | {n_train} segments ({n_train/len(groups)*100:.1f}%)")
    print(f"Val Set     : {len(meta['test_subjects'])} subjects | {n_val} segments ({n_val/len(groups)*100:.1f}%)")
    if meta.get("anchors"):
        print(f"Anchors     : {len(meta['anchors'])} calibration anchors (CalBased mode)")
    print("-" * 70)

    # 3. Label Normalization Statistics (Computed strictly on Train set)
    s_mean = float(y_sbp[train_idx].mean())
    s_std = float(y_sbp[train_idx].std() + 1e-6)
    d_mean = float(y_dbp[train_idx].mean())
    d_std = float(y_dbp[train_idx].std() + 1e-6)

    # 4. PyTorch DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X[train_idx], dtype=torch.float32),
        torch.tensor((y_sbp[train_idx] - s_mean) / s_std, dtype=torch.float32),
        torch.tensor((y_dbp[train_idx] - d_mean) / d_std, dtype=torch.float32),
        torch.tensor(y_sbp[train_idx], dtype=torch.float32),
        torch.tensor(y_dbp[train_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X[val_idx], dtype=torch.float32),
        torch.tensor((y_sbp[val_idx] - s_mean) / s_std, dtype=torch.float32),
        torch.tensor((y_dbp[val_idx] - d_mean) / d_std, dtype=torch.float32),
        torch.tensor(y_sbp[val_idx], dtype=torch.float32),
        torch.tensor(y_dbp[val_idx], dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Model, Optimizer, Criterion
    model = CalBPNet(in_channels=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = PhysicsBHSLoss(s_mean=s_mean, s_std=s_std, d_mean=d_mean, d_std=d_std)

    # 6. Training Loop
    history: List[Dict[str, Any]] = []
    best_val_mae = float("inf")

    print(f"{'Epoch':^7s} | {'Train Loss':^10s} | {'Tr MAE (All/S/D)':^18s} | {'Val Loss':^10s} | {'Val MAE (All/S/D)':^18s} | {'Checkpoint'}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae, tr_s_mae, tr_d_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, s_mean, s_std, d_mean, d_std, device
        )
        vl_loss, vl_mae, vl_s_mae, vl_d_mae = evaluate_validation(
            model, val_loader, criterion, s_mean, s_std, d_mean, d_std, device
        )
        elapsed = time.time() - t0

        # Best model checkpoint logic
        is_best = vl_mae < best_val_mae
        ckpt_tag = ""
        if is_best:
            best_val_mae = vl_mae
            ckpt_tag = "[BEST SAVED]"
            torch.save({
                "epoch": epoch,
                "protocol": protocol,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae_all": vl_mae,
                "val_mae_sbp": vl_s_mae,
                "val_mae_dbp": vl_d_mae,
                "s_mean": s_mean,
                "s_std": s_std,
                "d_mean": d_mean,
                "d_std": d_std,
                "train_subjects": meta["train_subjects"],
                "test_subjects": meta["test_subjects"],
            }, ckpt_path)

        epoch_record = {
            "epoch": epoch,
            "time_sec": round(elapsed, 2),
            "train_loss": round(tr_loss, 4),
            "train_mae_all": round(tr_mae, 2),
            "train_mae_sbp": round(tr_s_mae, 2),
            "train_mae_dbp": round(tr_d_mae, 2),
            "val_loss": round(vl_loss, 4),
            "val_mae_all": round(vl_mae, 2),
            "val_mae_sbp": round(vl_s_mae, 2),
            "val_mae_dbp": round(vl_d_mae, 2),
            "is_best": is_best,
        }
        history.append(epoch_record)

        print(
            f"{epoch:^7d} | {tr_loss:^10.4f} | {tr_mae:5.2f} / {tr_s_mae:4.1f} / {tr_d_mae:4.1f} | "
            f"{vl_loss:^10.4f} | {vl_mae:5.2f} / {vl_s_mae:4.1f} / {vl_d_mae:4.1f} | {ckpt_tag}"
        )

    # 7. Save History to JSON
    with open(history_path, "w") as f:
        json.dump({
            "protocol": protocol,
            "epochs": epochs,
            "best_val_mae": round(best_val_mae, 2),
            "checkpoint_path": str(ckpt_path),
            "history": history,
        }, f, indent=2)

    print("-" * 75)
    print(f"Training Complete! Saved best checkpoint to: {ckpt_path.name}")
    print(f"Training history saved to: {history_path.name}\n")

    return {
        "best_val_mae": best_val_mae,
        "checkpoint": str(ckpt_path),
        "history_file": str(history_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CalBPNet with CalFree or CalBased protocols.")
    parser.add_argument("--protocol", type=str, default="calfree", choices=["calfree", "calbased"])
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train (default: 1)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--subjects", type=int, default=5, help="Number of PulseDB subjects to train on (default: 5)")
    parser.add_argument("--source", type=str, default="PulseDB_Vital", choices=["PulseDB_Vital", "PulseDB_MIMIC"])
    args = parser.parse_args()

    run_training(
        protocol=args.protocol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_subjects=args.subjects,
        source=args.source,
    )
