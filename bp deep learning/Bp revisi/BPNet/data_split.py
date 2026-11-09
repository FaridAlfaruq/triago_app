"""
data_split.py
Module for partitioning PulseDB data under two distinct biomedical protocols:
1. CalBased Protocol (Wang et al., 2023):
   - Within-subject split (90% train / 10% test per subject).
   - Calibration anchor: Segment 0 of each subject in Train.
   - Tests 1-point cuff calibration tracking for known subjects.
2. CalFree Protocol (Subject-Wise / Subject-Independent):
   - Between-subject split (80% train subjects / 20% test subjects).
   - Test subjects are 100% unseen in training. Zero subject overlap.
   - Tests true zero-shot cross-subject clinical generalization.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from load import load_raw_subject, list_available_subjects
from signal_processing import preprocess_subject_signals


def split_calbased(
    groups: np.ndarray,
    test_ratio: float = 0.10,
    min_segments_per_subj: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Protocol 1: CalBased Partition (Within-Subject Split).
    - 90% of each subject's clean segments -> Train.
    - 10% disjoint segments of each subject -> Test.
    - Segment 0 of each subject in Train -> Calibration Anchor.

    Returns:
        train_idx: Array of training segment indices.
        test_idx: Array of testing segment indices.
        subj_anchors: Mapping {subject_id: anchor_segment_index}.
    """
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


def split_calfree(
    groups: np.ndarray,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Protocol 2: CalFree Partition (Subject-Wise Split, Zero Leakage).
    - Entire subjects assigned exclusively to Train or Test.
    - Test subjects are 100% unseen during training.

    Returns:
        train_idx: Array of training segment indices.
        test_idx: Array of testing segment indices.
        train_subjs: List of subject IDs in training set.
        test_subjs: List of subject IDs in testing set.
    """
    rng = np.random.RandomState(seed)
    unique_subjs = np.unique(groups)

    perm_subjs = rng.permutation(unique_subjs)
    n_test_subjs = max(1, int(len(unique_subjs) * test_ratio))
    test_subjs = sorted(perm_subjs[:n_test_subjs].tolist())
    train_subjs = sorted(perm_subjs[n_test_subjs:].tolist())

    train_idx = np.where(np.isin(groups, train_subjs))[0]
    test_idx = np.where(np.isin(groups, test_subjs))[0]

    # Verify zero subject overlap
    overlap = set(train_subjs).intersection(set(test_subjs))
    assert len(overlap) == 0, f"Subject leakage detected between train and test: {overlap}"

    return train_idx, test_idx, train_subjs, test_subjs


def split_dataset(
    groups: np.ndarray,
    protocol: str = "calfree",
    test_ratio: Optional[float] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified partition dispatcher for both CalBased and CalFree protocols.
    """
    proto = protocol.lower().strip()
    if proto in ("calfree", "subject_wise", "cal_free"):
        ratio = test_ratio if test_ratio is not None else 0.20
        train_idx, test_idx, train_subjs, test_subjs = split_calfree(
            groups, test_ratio=ratio, seed=seed
        )
        meta = {
            "protocol": "CalFree (Subject-Wise)",
            "train_subjects": train_subjs,
            "test_subjects": test_subjs,
            "anchors": {},
        }
        return train_idx, test_idx, meta

    elif proto in ("calbased", "cal_based", "calibration"):
        ratio = test_ratio if test_ratio is not None else 0.10
        train_idx, test_idx, anchors = split_calbased(
            groups, test_ratio=ratio, seed=seed
        )
        meta = {
            "protocol": "CalBased (Within-Subject)",
            "train_subjects": np.unique(groups[train_idx]).tolist(),
            "test_subjects": np.unique(groups[test_idx]).tolist(),
            "anchors": anchors,
        }
        return train_idx, test_idx, meta

    else:
        raise ValueError(f"Unknown protocol '{protocol}'. Choose 'calfree' or 'calbased'.")


def plot_protocol_comparison(
    num_subjects: int = 5,
    segments_per_subj: int = 10,
    source: str = "PulseDB_Vital",
    seed: int = 42,
) -> None:
    """
    Visual comparison of CalBased vs CalFree partition protocols:
    - Row 0: CalBased (Within-Subject Shuffling + 1-Point Anchor per person).
    - Row 1: CalFree (Subject-Wise Separation, 100% Unseen Test Patients).
    """
    print(f"\nLoading {num_subjects} subjects from {source} for partition comparison...")
    avail_subjs = list_available_subjects(source=source)[:num_subjects]

    all_X, all_sbp, all_dbp, all_g = [], [], [], []
    for s in avail_subjs:
        e, p, a, sbp, dbp = load_raw_subject(s, max_segments=segments_per_subj, source=source)
        Xc, sc, dc, mask = preprocess_subject_signals(e, p, sbp, dbp, fs=125)
        if len(Xc) > 0:
            all_X.append(Xc)
            all_sbp.append(sc)
            all_dbp.append(dc)
            all_g.extend([s] * len(Xc))

    if len(all_g) == 0:
        raise ValueError("No clean segments found across selected subjects.")

    groups = np.array(all_g)
    sbp_arr = np.concatenate(all_sbp)
    dbp_arr = np.concatenate(all_dbp)
    n_total = len(groups)
    unique_subjs = np.unique(groups)

    # 1. CalBased Partition (Within-Subject)
    cb_train, cb_test, cb_anchors = split_calbased(groups, test_ratio=0.10, seed=seed)

    # 2. CalFree Partition (Subject-Wise)
    cf_train, cf_test, cf_train_subjs, cf_test_subjs = split_calfree(groups, test_ratio=0.25, seed=seed)

    # --------------------------------------------------------------------------
    # PRINT DETAILED EXPLANATION TO TERMINAL
    # --------------------------------------------------------------------------
    cb_train_subjs = np.unique(groups[cb_train])
    cb_test_subjs = np.unique(groups[cb_test])
    n_unique_subjs = len(unique_subjs)

    print("=" * 65)
    print("PULSEDB DATA PARTITION REPORT")
    print("=" * 65)
    print(f"Total Cohort          : {n_unique_subjs} subjects | {n_total} clean segments")
    print("-" * 65)
    print("PROTOCOL 1: CalBased (Within-Subject Partition)")
    print(f"  Train Subjects      : {len(cb_train_subjs)} / {n_unique_subjs} subjects (100.0%) -> {cb_train_subjs.tolist()}")
    print(f"  Test Subjects       : {len(cb_test_subjs)} / {n_unique_subjs} subjects (100.0%) -> {cb_test_subjs.tolist()}")
    print(f"  Train Segments      : {len(cb_train)} / {n_total} segments ({len(cb_train)/n_total*100:.1f}%)")
    print(f"  Test Segments       : {len(cb_test)} / {n_total} segments ({len(cb_test)/n_total*100:.1f}%)")
    print(f"  Calibration Anchors : {len(cb_anchors)} anchor segments (1 per subject)")
    print("  Note: Subject identity shared between train and test.")
    print("-" * 65)
    print("PROTOCOL 2: CalFree (Subject-Wise Partition, Zero Leakage)")
    print(f"  Train Subjects      : {len(cf_train_subjs)} / {n_unique_subjs} subjects ({len(cf_train_subjs)/n_unique_subjs*100:.1f}%) -> {cf_train_subjs}")
    print(f"  Test Subjects       : {len(cf_test_subjs)} / {n_unique_subjs} subjects ({len(cf_test_subjs)/n_unique_subjs*100:.1f}%) -> {cf_test_subjs}")
    print(f"  Train Segments      : {len(cf_train)} / {n_total} segments ({len(cf_train)/n_total*100:.1f}%)")
    print(f"  Test Segments       : {len(cf_test)} / {n_total} segments ({len(cf_test)/n_total*100:.1f}%)")
    print("  Note: Zero subject overlap. Evaluates true zero-shot generalization.")
    print("=" * 65 + "\n")

    # --------------------------------------------------------------------------
    # VISUALIZATION (Simple Titles, Clean Legends, Zero Text Clutter)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    plt.subplots_adjust(hspace=0.45)
    seg_indices = np.arange(n_total)

    # Detect subject boundary transitions for vertical divider lines
    change_idx = np.where(groups[:-1] != groups[1:])[0] + 1
    subj_boundaries = [0] + change_idx.tolist() + [n_total]

    # ROW 0: CALBASED PARTITION
    ax = axes[0]
    cb_colors = ["#1f77b4" if i in cb_train else "#ff7f0e" for i in seg_indices]
    bars_cb = ax.bar(seg_indices, np.ones(n_total), color=cb_colors, edgecolor="black", linewidth=0.5, width=0.8)

    # Highlight CalBased Anchors
    anchor_indices = list(cb_anchors.values())
    for a_idx in anchor_indices:
        bars_cb[a_idx].set_edgecolor("#d4af37")
        bars_cb[a_idx].set_linewidth(2.2)
        ax.scatter(a_idx, 1.15, color="#d4af37", marker="*", s=140, zorder=5)

    # Subject vertical division lines
    for b in subj_boundaries[1:-1]:
        ax.axvline(b - 0.5, color="red", linestyle=":", linewidth=1.0)

    # Label subject names at top
    for i in range(len(subj_boundaries) - 1):
        mid = (subj_boundaries[i] + subj_boundaries[i + 1] - 1) / 2.0
        s_name = groups[subj_boundaries[i]]
        ax.text(mid, 1.25, s_name, ha="center", va="center", fontsize=8, weight="bold")

    ax.set_title("CalBased Partition", fontsize=11, weight="bold")
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("Status")
    ax.set_ylim(0, 1.45)
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    train_patch = mpatches.Patch(color="#1f77b4", label="Train")
    test_patch = mpatches.Patch(color="#ff7f0e", label="Test")
    anchor_marker = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d4af37", markersize=11, label="Anchor")
    ax.legend(handles=[train_patch, test_patch, anchor_marker], loc="upper right", fontsize=8)

    # ROW 1: CALFREE PARTITION (SUBJECT-WISE)
    ax = axes[1]
    cf_colors = ["#1f77b4" if i in cf_train else "#ff7f0e" for i in seg_indices]
    ax.bar(seg_indices, np.ones(n_total), color=cf_colors, edgecolor="black", linewidth=0.5, width=0.8)

    for b in subj_boundaries[1:-1]:
        ax.axvline(b - 0.5, color="red", linestyle=":", linewidth=1.0)

    for i in range(len(subj_boundaries) - 1):
        mid = (subj_boundaries[i] + subj_boundaries[i + 1] - 1) / 2.0
        s_name = groups[subj_boundaries[i]]
        is_test = s_name in cf_test_subjs
        tag = f"{s_name}\n(Test)" if is_test else f"{s_name}\n(Train)"
        col = "#d95f02" if is_test else "#1b9e77"
        ax.text(mid, 1.25, tag, ha="center", va="center", fontsize=8, weight="bold", color=col)

    ax.set_title("CalFree Partition", fontsize=11, weight="bold")
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("Status")
    ax.set_ylim(0, 1.45)
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.legend(handles=[train_patch, test_patch], loc="upper right", fontsize=8)

    fig.suptitle("Data Partition: CalBased vs CalFree Comparison", fontsize=12, weight="bold")
    print("Displaying protocol comparison plot via plt.show()...")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PulseDB data partitioning (CalBased & CalFree).")
    parser.add_argument("--protocol", type=str, default="compare", choices=["compare", "calbased", "calfree"])
    parser.add_argument("--subjects", type=int, default=5, help="Number of subjects (default: 5)")
    parser.add_argument("--source", type=str, default="PulseDB_Vital", choices=["PulseDB_Vital", "PulseDB_MIMIC"])
    parser.add_argument("--no-plot", action="store_true", help="Print summary without opening plot window")
    args = parser.parse_args()

    if args.no_plot:
        avail_subjs = list_available_subjects(source=args.source)[:args.subjects]
        all_g = []
        for s in avail_subjs:
            e, p, a, sbp, dbp = load_raw_subject(s, max_segments=10, source=args.source)
            Xc, sc, dc, mask = preprocess_subject_signals(e, p, sbp, dbp, fs=125)
            all_g.extend([s] * len(Xc))

        groups = np.array(all_g)
        proto = args.protocol if args.protocol != "compare" else "calfree"
        train_idx, test_idx, meta = split_dataset(groups, protocol=proto)
        n_tot = len(groups)
        n_sub = len(np.unique(groups))
        print(f"\nProtocol: {meta['protocol']}")
        print(f"  Train: {len(meta['train_subjects'])} / {n_sub} subjects | {len(train_idx)} / {n_tot} segments ({len(train_idx)/n_tot*100:.1f}%)")
        print(f"  Test : {len(meta['test_subjects'])} / {n_sub} subjects | {len(test_idx)} / {n_tot} segments ({len(test_idx)/n_tot*100:.1f}%)")
    else:
        plot_protocol_comparison(
            num_subjects=args.subjects,
            source=args.source,
        )
