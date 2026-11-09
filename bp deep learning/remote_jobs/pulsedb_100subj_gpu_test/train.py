"""
train.py
PulseDB 100-Subject GPU Multi-Epoch Training Pipeline.
CalBPNet (408,354 parameters) with Dual Protocol Support:
1. CalFree (Subject-Wise, Zero-Shot Generalization, 80/20 subject split)
2. CalBased (Within-Subject, 90/10 split with 1-point baseline calibration)
Includes early stopping (patience=25), BHS standard grading, and AAMI standard compliance.
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==============================================================================
# Determinism & Dataset Discovery
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def find_dataset_file(data_path: Optional[str] = None) -> Path:
    if data_path and Path(data_path).is_file():
        return Path(data_path)

    # 1. Check Kaggle input mount (rglob *.npz)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        found = [p for p in kaggle_input.rglob("*.npz") if p.is_file()]
        if found:
            return found[0]

    # 2. Local fallback candidates
    candidates = [
        Path("scratch/gemastik/pulsedb_100subj_7ch.npz"),
        Path("pulsedb_100subj_7ch.npz"),
        Path(__file__).resolve().parents[2] / "scratch/gemastik/pulsedb_100subj_7ch.npz",
        Path(__file__).resolve().parents[3] / "scratch/gemastik/pulsedb_100subj_7ch.npz",
        Path("scratch/gemastik/pulsedb_benchmark_7ch.npz"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError("Could not find PulseDB .npz dataset file! Specify with --data-path.")


# ==============================================================================
# Biomedical Partitioning Protocols (CalFree & CalBased)
# ==============================================================================

def split_calfree(
    groups: np.ndarray,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Subject-wise partition with 0 subject overlap (Generalization test)."""
    rng = np.random.RandomState(seed)
    unique_subjs = np.unique(groups)

    perm_subjs = rng.permutation(unique_subjs)
    n_test_subjs = max(1, int(len(unique_subjs) * test_ratio))
    test_subjs = sorted(perm_subjs[:n_test_subjs].tolist())
    train_subjs = sorted(perm_subjs[n_test_subjs:].tolist())

    train_idx = np.where(np.isin(groups, train_subjs))[0]
    test_idx = np.where(np.isin(groups, test_subjs))[0]

    overlap = set(train_subjs).intersection(set(test_subjs))
    assert len(overlap) == 0, f"Subject leakage detected: {overlap}"
    return train_idx, test_idx, train_subjs, test_subjs


def split_calbased(
    groups: np.ndarray,
    test_ratio: float = 0.10,
    min_segments_per_subj: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Within-subject partition (90/10) with Segment 0 of train as calibration anchor."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    subj_anchors = {}

    unique_subjs = np.unique(groups)
    for s in unique_subjs:
        pts = np.where(groups == s)[0]
        if len(pts) < min_segments_per_subj:
            continue

        perm = rng.permutation(pts)
        n_test = max(1, int(len(pts) * test_ratio))
        p_test = perm[:n_test]
        p_train = perm[n_test:]

        train_idx.extend(p_train)
        test_idx.extend(p_test)
        subj_anchors[s] = int(p_train[0])

    return np.array(train_idx), np.array(test_idx), subj_anchors


# ==============================================================================
# Evaluation Metrics (BHS, AAMI, Pearson, R2, MAE, RMSE)
# ==============================================================================

def compute_bhs_grades(errors: np.ndarray) -> Dict[str, Any]:
    abs_err = np.abs(errors)
    p5 = float(np.mean(abs_err <= 5.0) * 100.0) if len(errors) > 0 else 0.0
    p10 = float(np.mean(abs_err <= 10.0) * 100.0) if len(errors) > 0 else 0.0
    p15 = float(np.mean(abs_err <= 15.0) * 100.0) if len(errors) > 0 else 0.0

    if p5 >= 60.0 and p10 >= 85.0 and p15 >= 95.0:
        grade = "A"
    elif p5 >= 50.0 and p10 >= 75.0 and p15 >= 90.0:
        grade = "B"
    elif p5 >= 40.0 and p10 >= 65.0 and p15 >= 85.0:
        grade = "C"
    else:
        grade = "D"

    return {
        "pct_le_5": round(p5, 1),
        "pct_le_10": round(p10, 1),
        "pct_le_15": round(p15, 1),
        "bhs_grade": grade,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "mae": 0.0, "rmse": 0.0, "me": 0.0, "std": 0.0,
            "r2": 0.0, "pearson_r": 0.0, "aami_pass": False,
            "bhs_grade": "D", "pct_le_5": 0.0, "pct_le_10": 0.0, "pct_le_15": 0.0
        }
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    me = float(np.mean(errors))
    std = float(np.std(errors))

    aami_pass = bool(abs(me) <= 5.0 and std <= 8.0)
    bhs = compute_bhs_grades(errors)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - float(ss_res / (ss_tot + 1e-8))

    if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "me": round(me, 2),
        "std": round(std, 2),
        "r2": round(r2, 3),
        "pearson_r": round(r, 3),
        "aami_pass": aami_pass,
        "bhs_grade": bhs["bhs_grade"],
        "pct_le_5": bhs["pct_le_5"],
        "pct_le_10": bhs["pct_le_10"],
        "pct_le_15": bhs["pct_le_15"],
    }


# ==============================================================================
# Model Architecture (CalBPNet: 408,354 parameters) & Loss
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
        if x.dim() == 3 and x.shape[-1] == 7:
            x = x.transpose(1, 2)
        elif x.shape[1] == 1250:
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

        sbp_z = self.sbp_head(context).squeeze(-1)
        dbp_z = self.dbp_head(context).squeeze(-1)
        return sbp_z, dbp_z


class PhysicsBHSLoss(nn.Module):
    def __init__(
        self,
        s_mean: float = 117.94,
        s_std: float = 20.40,
        d_mean: float = 62.68,
        d_std: float = 11.93,
        lambda_bhs: float = 0.05,
        lambda_phys: float = 0.02,
    ):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.s_mean = s_mean
        self.s_std = s_std
        self.d_mean = d_mean
        self.d_std = d_std
        self.lambda_bhs = lambda_bhs
        self.lambda_phys = lambda_phys

    def forward(
        self,
        p_s_z: torch.Tensor,
        p_d_z: torch.Tensor,
        t_s_z: torch.Tensor,
        t_d_z: torch.Tensor,
    ) -> torch.Tensor:
        l_base = 0.6 * self.huber(p_s_z, t_s_z) + 0.4 * self.huber(p_d_z, t_d_z)

        p_s_mmhg = p_s_z * self.s_std + self.s_mean
        p_d_mmhg = p_d_z * self.d_std + self.d_mean
        t_s_mmhg = t_s_z * self.s_std + self.s_mean
        t_d_mmhg = t_d_z * self.d_std + self.d_mean

        err_s = torch.abs(p_s_mmhg - t_s_mmhg)
        err_d = torch.abs(p_d_mmhg - t_d_mmhg)
        l_bhs = torch.relu(err_s - 5.0).mean() + torch.relu(err_d - 5.0).mean()

        pp_pred = p_s_mmhg - p_d_mmhg
        pp_viol = torch.relu(15.0 - pp_pred) + torch.relu(pp_pred - 120.0)
        l_phys = pp_viol.mean()

        return l_base + (self.lambda_bhs * l_bhs / self.s_std) + (self.lambda_phys * l_phys / self.s_std)


class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# ==============================================================================
# Protocol Training Runner
# ==============================================================================

def train_protocol(
    protocol: str,
    X: np.ndarray,
    y_sbp: np.ndarray,
    y_dbp: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # 1. Dataset Split
    if protocol == "calfree":
        train_idx, test_idx, train_subjs, test_subjs = split_calfree(
            groups, test_ratio=0.20, seed=args.seed
        )
        anchors: Dict[str, int] = {}
    elif protocol == "calbased":
        train_idx, test_idx, anchors = split_calbased(
            groups, test_ratio=0.10, min_segments_per_subj=5, seed=args.seed
        )
        train_subjs = sorted(np.unique(groups[train_idx]).tolist())
        test_subjs = sorted(np.unique(groups[test_idx]).tolist())
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    # 2. Target Normalization (Computed strictly on Training split)
    s_mean = float(y_sbp[train_idx].mean())
    s_std = float(y_sbp[train_idx].std() + 1e-6)
    d_mean = float(y_dbp[train_idx].mean())
    d_std = float(y_dbp[train_idx].std() + 1e-6)

    # 3. DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X[train_idx], dtype=torch.float32),
        torch.tensor((y_sbp[train_idx] - s_mean) / s_std, dtype=torch.float32),
        torch.tensor((y_dbp[train_idx] - d_mean) / d_std, dtype=torch.float32),
        torch.tensor(y_sbp[train_idx], dtype=torch.float32),
        torch.tensor(y_dbp[train_idx], dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X[test_idx], dtype=torch.float32),
        torch.tensor((y_sbp[test_idx] - s_mean) / s_std, dtype=torch.float32),
        torch.tensor((y_dbp[test_idx] - d_mean) / d_std, dtype=torch.float32),
        torch.tensor(y_sbp[test_idx], dtype=torch.float32),
        torch.tensor(y_dbp[test_idx], dtype=torch.float32),
        torch.tensor(np.arange(len(test_idx)), dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Model, Optimizer, Scheduler, Loss, EarlyStopping
    set_seed(args.seed)
    model = CalBPNet(in_channels=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = PhysicsBHSLoss(s_mean=s_mean, s_std=s_std, d_mean=d_mean, d_std=d_std)
    early_stopping = EarlyStopping(patience=args.patience)

    def evaluate_model(curr_model: nn.Module) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        curr_model.eval()
        val_loss_sum = 0.0
        val_count = 0
        pred_s_z_list, pred_d_z_list = [], []
        true_s_list, true_d_list = [], []
        eval_pos_list = []

        with torch.no_grad():
            for step, (bx, bs_z, bd_z, bs_raw, bd_raw, bpos) in enumerate(test_loader, 1):
                bx, bs_z, bd_z = bx.to(device), bs_z.to(device), bd_z.to(device)
                p_s_z, p_d_z = curr_model(bx)
                loss = criterion(p_s_z, p_d_z, bs_z, bd_z)
                val_loss_sum += loss.item() * len(bx)
                val_count += len(bx)

                pred_s_z_list.append(p_s_z.cpu())
                pred_d_z_list.append(p_d_z.cpu())
                true_s_list.append(bs_raw.cpu())
                true_d_list.append(bd_raw.cpu())
                eval_pos_list.append(bpos.cpu())

                if args.max_batches and step >= args.max_batches:
                    break

        raw_pred_s = torch.cat(pred_s_z_list).numpy() * s_std + s_mean
        raw_pred_d = torch.cat(pred_d_z_list).numpy() * d_std + d_mean
        t_s = torch.cat(true_s_list).numpy()
        t_d = torch.cat(true_d_list).numpy()
        eval_pos = torch.cat(eval_pos_list).numpy()
        eval_subjs = groups[test_idx[eval_pos]]

        if protocol == "calbased":
            unique_subjs = np.unique(eval_subjs)
            anchor_idx_list = [anchors[s] for s in unique_subjs]
            anc_x = torch.tensor(X[anchor_idx_list], dtype=torch.float32).to(device)
            with torch.no_grad():
                anc_s_z, anc_d_z = curr_model(anc_x)
            anc_pred_s = anc_s_z.cpu().numpy() * s_std + s_mean
            anc_pred_d = anc_d_z.cpu().numpy() * d_std + d_mean

            delta_s = {s: y_sbp[idx] - ps for s, idx, ps in zip(unique_subjs, anchor_idx_list, anc_pred_s)}
            delta_d = {s: y_dbp[idx] - pd for s, idx, pd in zip(unique_subjs, anchor_idx_list, anc_pred_d)}

            p_s = raw_pred_s + np.array([delta_s[s] for s in eval_subjs])
            p_d = raw_pred_d + np.array([delta_d[s] for s in eval_subjs])
        else:
            p_s = raw_pred_s
            p_d = raw_pred_d

        v_loss = val_loss_sum / max(1, val_count)
        mae_s = float(np.mean(np.abs(p_s - t_s)))
        mae_d = float(np.mean(np.abs(p_d - t_d)))
        mae_all = round((mae_s + mae_d) / 2.0, 2)
        return v_loss, mae_all, round(mae_s, 2), round(mae_d, 2), p_s, p_d, t_s, t_d

    best_val_mae = float("inf")
    best_epoch = 1
    best_state = None
    stopped_epoch = args.epochs
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss_sum = 0.0
        tr_count = 0
        tr_err_s, tr_err_d = [], []

        for step, (bx, bs_z, bd_z, bs_raw, bd_raw) in enumerate(train_loader, 1):
            bx, bs_z, bd_z = bx.to(device), bs_z.to(device), bd_z.to(device)
            bs_raw, bd_raw = bs_raw.to(device), bd_raw.to(device)

            optimizer.zero_grad()
            p_s_z, p_d_z = model(bx)
            loss = criterion(p_s_z, p_d_z, bs_z, bd_z)
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * len(bx)
            tr_count += len(bx)

            p_s_mmhg = p_s_z.detach() * s_std + s_mean
            p_d_mmhg = p_d_z.detach() * d_std + d_mean
            tr_err_s.append(torch.abs(p_s_mmhg - bs_raw).cpu().numpy())
            tr_err_d.append(torch.abs(p_d_mmhg - bd_raw).cpu().numpy())

            if args.max_batches and step >= args.max_batches:
                break

        scheduler.step()
        tr_loss = tr_loss_sum / max(1, tr_count)
        tr_mae_s = float(np.mean(np.concatenate(tr_err_s))) if tr_err_s else 0.0
        tr_mae_d = float(np.mean(np.concatenate(tr_err_d))) if tr_err_d else 0.0
        tr_mae_all = round((tr_mae_s + tr_mae_d) / 2.0, 2)

        vl_loss, vl_mae_all, vl_mae_s, vl_mae_d, _, _, _, _ = evaluate_model(model)
        elapsed = time.time() - t0

        is_best = early_stopping.step(vl_mae_all)
        if is_best:
            best_val_mae = vl_mae_all
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_rec = {
            "epoch": epoch,
            "time_sec": round(elapsed, 2),
            "train_loss": round(tr_loss, 4),
            "train_mae_all": tr_mae_all,
            "train_mae_sbp": round(tr_mae_s, 2),
            "train_mae_dbp": round(tr_mae_d, 2),
            "val_loss": round(vl_loss, 4),
            "val_mae_all": vl_mae_all,
            "val_mae_sbp": vl_mae_s,
            "val_mae_dbp": vl_mae_d,
            "lr": float(scheduler.get_last_lr()[0]),
            "is_best": is_best,
        }
        history.append(epoch_rec)

        # Stdout Suppression Rule: print epoch 1, every 25 epochs, early stopping, or final epoch
        should_print = (
            epoch == 1 or
            epoch % 25 == 0 or
            epoch == args.epochs or
            early_stopping.early_stop
        )
        if should_print:
            best_tag = " [BEST]" if is_best else ""
            print(
                f"[{protocol.upper():<8s}] Ep {epoch:03d}/{args.epochs:03d} ({elapsed:.1f}s) | "
                f"TrLoss: {tr_loss:.4f} | "
                f"ValMAE: {vl_mae_all:.2f} (S:{vl_mae_s:.2f}, D:{vl_mae_d:.2f}){best_tag}"
            )

        if early_stopping.early_stop:
            stopped_epoch = epoch
            print(f"[{protocol.upper()}] Early stopped at epoch {epoch:03d} (Best Epoch: {best_epoch:03d}, Val MAE: {best_val_mae:.2f} mmHg)")
            break

    # 5. Final Evaluation on Best Model State
    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, _, _, best_p_s, best_p_d, best_t_s, best_t_d = evaluate_model(model)
    s_metrics = compute_metrics(best_t_s, best_p_s)
    d_metrics = compute_metrics(best_t_d, best_p_d)
    overall_mae = round((s_metrics["mae"] + d_metrics["mae"]) / 2.0, 2)
    overall_rmse = round((s_metrics["rmse"] + d_metrics["rmse"]) / 2.0, 2)
    aami_pass = bool(s_metrics["aami_pass"] and d_metrics["aami_pass"])

    eval_metrics = {
        "protocol": protocol,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "aami_pass": aami_pass,
        "sbp": s_metrics,
        "dbp": d_metrics,
        "test_segments": len(best_p_s),
        "test_subjects": test_subjs,
    }

    # 6. Save Protocol Artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"best_calbpnet_{protocol}.pt")
    torch.save({
        "epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "protocol": protocol,
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_mae": overall_mae,
        "val_mae_sbp": s_metrics["mae"],
        "val_mae_dbp": d_metrics["mae"],
        "s_mean": s_mean,
        "s_std": s_std,
        "d_mean": d_mean,
        "d_std": d_std,
        "train_subjects": train_subjs,
        "test_subjects": test_subjs,
        "metrics": eval_metrics,
    }, ckpt_path)

    history_path = os.path.join(args.output_dir, f"history_{protocol}.json")
    with open(history_path, "w") as f:
        json.dump({
            "protocol": protocol,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_mae": best_val_mae,
            "history": history,
        }, f, indent=2)

    eval_path = os.path.join(args.output_dir, f"eval_metrics_{protocol}.json")
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    return eval_metrics, history


# ==============================================================================
# Summary Display & Comparison Exporter
# ==============================================================================

def print_comparison_table(cf: Dict[str, Any], cb: Dict[str, Any]) -> None:
    cf_s, cf_d = cf["sbp"], cf["dbp"]
    cb_s, cb_d = cb["sbp"], cb["dbp"]
    cf_aami = "PASS" if cf["aami_pass"] else "FAIL"
    cb_aami = "PASS" if cb["aami_pass"] else "FAIL"

    print("=" * 76)
    print("RESULTS COMPARISON: CalFree (Zero-Shot) vs CalBased (1-Point Calibration)")
    print("-" * 76)
    print(f"{'Metric':<24s} | {'CalFree (Subject-Wise)':<22s} | {'CalBased (1-Pt Cal)':<22s}")
    print("-" * 76)
    print(f"{'MAE SBP / DBP (Overall)':<24s} | {cf_s['mae']:5.2f} / {cf_d['mae']:5.2f} ({cf['overall_mae']:5.2f}) | {cb_s['mae']:5.2f} / {cb_d['mae']:5.2f} ({cb['overall_mae']:5.2f})")
    print(f"{'RMSE SBP / DBP':<24s} | {cf_s['rmse']:5.2f} / {cf_d['rmse']:5.2f}         | {cb_s['rmse']:5.2f} / {cb_d['rmse']:5.2f}")
    print(f"{'ME +/- STD SBP (mmHg)':<24s} | {cf_s['me']:+5.2f} +/- {cf_s['std']:<5.2f}         | {cb_s['me']:+5.2f} +/- {cb_s['std']:<5.2f}")
    print(f"{'ME +/- STD DBP (mmHg)':<24s} | {cf_d['me']:+5.2f} +/- {cf_d['std']:<5.2f}         | {cb_d['me']:+5.2f} +/- {cb_d['std']:<5.2f}")
    print(f"{'R2 SBP / DBP':<24s} | {cf_s['r2']:5.3f} / {cf_d['r2']:5.3f}         | {cb_s['r2']:5.3f} / {cb_d['r2']:5.3f}")
    print(f"{'Pearson r SBP / DBP':<24s} | {cf_s['pearson_r']:5.3f} / {cf_d['pearson_r']:5.3f}         | {cb_s['pearson_r']:5.3f} / {cb_d['pearson_r']:5.3f}")
    print(f"{'BHS Grade SBP / DBP':<24s} |   {cf_s['bhs_grade']:^3s}   /   {cf_d['bhs_grade']:^3s}         |   {cb_s['bhs_grade']:^3s}   /   {cb_d['bhs_grade']:^3s}")
    print(f"{'AAMI Standard Pass':<24s} | {cf_aami:^22s} | {cb_aami:^22s}")
    print(f"{'Best / Stopped Epoch':<24s} |  {cf['best_epoch']:3d}   /   {cf['stopped_epoch']:<3d}         |  {cb['best_epoch']:3d}   /   {cb['stopped_epoch']:<3d}")
    print("=" * 76)


def save_comparison_contract(cf: Dict[str, Any], cb: Dict[str, Any], output_dir: str) -> str:
    cf_s, cf_d = cf["sbp"], cf["dbp"]
    cb_s, cb_d = cb["sbp"], cb["dbp"]
    comparison = {
        "calfree": cf,
        "calbased": cb,
        "comparison_summary": {
            "mae_sbp": {
                "calfree": cf_s["mae"],
                "calbased": cb_s["mae"],
                "improvement_mmhg": round(cf_s["mae"] - cb_s["mae"], 2),
            },
            "mae_dbp": {
                "calfree": cf_d["mae"],
                "calbased": cb_d["mae"],
                "improvement_mmhg": round(cf_d["mae"] - cb_d["mae"], 2),
            },
            "mae_overall": {
                "calfree": cf["overall_mae"],
                "calbased": cb["overall_mae"],
                "improvement_mmhg": round(cf["overall_mae"] - cb["overall_mae"], 2),
            },
            "rmse_sbp": {
                "calfree": cf_s["rmse"],
                "calbased": cb_s["rmse"],
                "improvement_mmhg": round(cf_s["rmse"] - cb_s["rmse"], 2),
            },
            "rmse_dbp": {
                "calfree": cf_d["rmse"],
                "calbased": cb_d["rmse"],
                "improvement_mmhg": round(cf_d["rmse"] - cb_d["rmse"], 2),
            },
            "me_std_sbp": {
                "calfree": f"{cf_s['me']:+.2f} +/- {cf_s['std']:.2f}",
                "calbased": f"{cb_s['me']:+.2f} +/- {cb_s['std']:.2f}",
            },
            "me_std_dbp": {
                "calfree": f"{cf_d['me']:+.2f} +/- {cf_d['std']:.2f}",
                "calbased": f"{cb_d['me']:+.2f} +/- {cb_d['std']:.2f}",
            },
            "r2_sbp": {"calfree": cf_s["r2"], "calbased": cb_s["r2"]},
            "r2_dbp": {"calfree": cf_d["r2"], "calbased": cb_d["r2"]},
            "pearson_r_sbp": {"calfree": cf_s["pearson_r"], "calbased": cb_s["pearson_r"]},
            "pearson_r_dbp": {"calfree": cf_d["pearson_r"], "calbased": cb_d["pearson_r"]},
            "bhs_grade_sbp": {"calfree": cf_s["bhs_grade"], "calbased": cb_s["bhs_grade"]},
            "bhs_grade_dbp": {"calfree": cf_d["bhs_grade"], "calbased": cb_d["bhs_grade"]},
            "aami_pass_sbp": {"calfree": cf_s["aami_pass"], "calbased": cb_s["aami_pass"]},
            "aami_pass_dbp": {"calfree": cf_d["aami_pass"], "calbased": cb_d["aami_pass"]},
            "aami_pass_overall": {"calfree": cf["aami_pass"], "calbased": cb["aami_pass"]},
            "best_epoch": {"calfree": cf["best_epoch"], "calbased": cb["best_epoch"]},
            "stopped_epoch": {"calfree": cf["stopped_epoch"], "calbased": cb["stopped_epoch"]},
        },
    }
    out_path = os.path.join(output_dir, "results_comparison_500epochs.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    return out_path


# ==============================================================================
# Main Entrypoint
# ==============================================================================

def main():
    default_bs = 64 if torch.cuda.is_available() else 16
    parser = argparse.ArgumentParser(description="PulseDB 100-Subject Dual-Protocol Training (CalFree & CalBased)")
    parser.add_argument("--protocol", type=str, default="both", choices=["both", "calfree", "calbased"], help="Protocol to run (default: both)")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs per protocol (default: 500)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience in epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=default_bs, help=f"Batch size (default: {default_bs})")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches for smoke testing")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save checkpoints and JSON results")
    parser.add_argument("--data-path", type=str, default=None, help="Explicit path to dataset npz file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args, _ = parser.parse_known_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    data_path = find_dataset_file(args.data_path)
    data = np.load(data_path)
    X = data["X"]
    y_sbp = data["y_sbp"]
    y_dbp = data["y_dbp"]
    groups = data["groups"]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"PulseDB Dual Training | Device: {device} ({gpu_name}) | Total: {len(np.unique(groups))} subjs ({len(X):,} segs) | Mode: {args.protocol}")

    eval_cf, eval_cb = None, None

    # Load previously existing eval metrics if available
    cf_path = os.path.join(args.output_dir, "eval_metrics_calfree.json")
    cb_path = os.path.join(args.output_dir, "eval_metrics_calbased.json")
    if os.path.exists(cf_path):
        try:
            with open(cf_path) as f:
                eval_cf = json.load(f)
        except Exception:
            pass
    if os.path.exists(cb_path):
        try:
            with open(cb_path) as f:
                eval_cb = json.load(f)
        except Exception:
            pass

    # Protocol 1: CalFree
    if args.protocol in ("both", "calfree"):
        eval_cf, _ = train_protocol("calfree", X, y_sbp, y_dbp, groups, args, device)

    # Protocol 2: CalBased
    if args.protocol in ("both", "calbased"):
        eval_cb, _ = train_protocol("calbased", X, y_sbp, y_dbp, groups, args, device)

    # Combined Comparison Contract
    if eval_cf is not None and eval_cb is not None:
        save_comparison_contract(eval_cf, eval_cb, args.output_dir)
        print_comparison_table(eval_cf, eval_cb)
    elif eval_cf is not None:
        s, d = eval_cf["sbp"], eval_cf["dbp"]
        print(f"[CALFREE DONE] Overall MAE: {eval_cf['overall_mae']:.2f} mmHg (SBP: {s['mae']:.2f} [{s['bhs_grade']}], DBP: {d['mae']:.2f} [{d['bhs_grade']}]) | AAMI: {'PASS' if eval_cf['aami_pass'] else 'FAIL'}")
    elif eval_cb is not None:
        s, d = eval_cb["sbp"], eval_cb["dbp"]
        print(f"[CALBASED DONE] Overall MAE: {eval_cb['overall_mae']:.2f} mmHg (SBP: {s['mae']:.2f} [{s['bhs_grade']}], DBP: {d['mae']:.2f} [{d['bhs_grade']}]) | AAMI: {'PASS' if eval_cb['aami_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
