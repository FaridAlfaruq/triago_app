"""Build and check the first subject-disjoint revised BP deployment candidate."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from model.bpnet_inference import REVISED_CHANNELS, RevisedBPPredictor


class OrderedBP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveforms):
        return torch.stack(self.model(waveforms), dim=1)


def fit_first_fold(experiment_dir):
    sys.path.insert(0, str(experiment_dir))
    backbone_dir = experiment_dir / "remote_jobs" / "pulsedb_calbased"
    sys.path.insert(0, str(backbone_dir))
    from run_transfer_learning import CalBPNet, apply_tight_sqi, extract_backbone_context, set_seed

    with np.load(experiment_dir / "Bp revisi" / "dataset_7ch.npz") as dataset:
        selected = apply_tight_sqi(dataset["X"], fs=125)
        windows = torch.from_numpy(dataset["X"][selected]).float()
        targets = np.column_stack((dataset["y_sbp"][selected], dataset["y_dbp"][selected]))
        groups = dataset["groups"][selected]
    assert len(windows) == 472 and len(np.unique(groups)) == 57
    train, held_out = next(GroupKFold(n_splits=3).split(windows, targets[:, 0], groups))
    assert not set(groups[train]).intersection(groups[held_out])

    set_seed(42)
    backbone = CalBPNet()
    weights = backbone_dir / "output_v3" / "output" / "model_best.pt"
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    backbone.eval().requires_grad_(False)
    context = extract_backbone_context(backbone, windows[train])

    set_seed(42)
    heads = torch.nn.ModuleList([copy.deepcopy(backbone.sbp_head), copy.deepcopy(backbone.dbp_head)])
    heads.train().requires_grad_(True)
    initial_weights = [parameter.detach().clone() for parameter in heads.parameters()]
    optimizer = torch.optim.Adam(heads.parameters(), lr=5e-4)
    mean, std = targets[train].mean(axis=0), targets[train].std(axis=0)
    normalized = torch.from_numpy((targets[train] - mean) / std).float()
    train_groups = groups[train]
    anchors = torch.tensor([np.flatnonzero(train_groups == group)[0] for group in train_groups])
    huber = torch.nn.SmoothL1Loss()
    for _ in range(25):
        optimizer.zero_grad()
        sbp = heads[0](context).squeeze(-1)
        dbp = heads[1](context).squeeze(-1)
        sbp_target, dbp_target = normalized.unbind(dim=1)
        loss_abs = huber(sbp, sbp_target) + huber(dbp, dbp_target)
        loss_cal = huber(sbp - sbp[anchors], sbp_target - sbp_target[anchors]) + huber(
            dbp - dbp[anchors], dbp_target - dbp_target[anchors]
        )
        l2_sp = sum((parameter - initial).square().sum()
                    for parameter, initial in zip(heads.parameters(), initial_weights))
        (loss_abs + 1.5 * loss_cal + 0.05 * l2_sp).backward()
        optimizer.step()

    backbone.sbp_head, backbone.dbp_head = heads
    model = OrderedBP(backbone).eval()
    return model, windows[held_out], mean, std


def export_candidate(model, windows, mean, std, output_dir):
    torch.backends.mha.set_fastpath_enabled(False)
    sample = (windows[:1],)
    with torch.no_grad():
        expected = model(windows).numpy() * std + mean

    with TemporaryDirectory() as temp:
        temp = Path(temp)
        manifest_path = temp / "revised_bp_manifest.json"
        for suffix in ("tflite", "onnx"):
            artifact = temp / f"revised_bp.{suffix}"
            try:
                if suffix == "tflite":
                    import litert_torch

                    litert_torch.convert(model, sample).export(str(artifact))
                else:
                    torch.onnx.export(model, sample, str(artifact), input_names=["waveforms"],
                                      output_names=["bp"], opset_version=17, dynamo=False)
                manifest = {
                    "fold": 1,
                    "artifact": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "channels": list(REVISED_CHANNELS),
                    "input_shape": [1, 1250, 7],
                    "output_order": ["sbp", "dbp"],
                    "target_mean": mean.tolist(),
                    "target_std": std.tolist(),
                }
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                predictor = RevisedBPPredictor(manifest_path)
                actual = np.asarray([predictor.predict_segment(window.numpy()) for window in windows])
                max_error = np.max(np.abs(actual - expected), axis=0)
                if not np.isfinite(max_error).all() or np.any(max_error > 0.1):
                    raise ValueError(f"Conversion parity failed: {max_error} mmHg")
            except Exception as error:
                print(f"{suffix} candidate rejected: {error}")
                continue
            manifest["parity_windows"] = len(windows)
            manifest["max_conversion_error_mmhg"] = max_error.tolist()
            manifest["torch_version"] = str(torch.__version__)
            manifest["validation_runtime"] = predictor.interpreter_backend
            artifact_output = output_dir / artifact.name
            manifest_output = output_dir / manifest_path.name
            if artifact_output.exists() or manifest_output.exists():
                raise FileExistsError("Candidate output already exists")
            with artifact_output.open("xb") as stream:
                stream.write(artifact.read_bytes())
            with manifest_output.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
            print(f"{suffix} max SBP/DBP conversion errors over {len(windows)} windows: {max_error} mmHg")
            return
    raise RuntimeError("No export passed the 0.1 mmHg per-output parity gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="Directory containing run_transfer_learning.py")
    parser.add_argument("output_dir", type=Path, help="Existing directory for the verified candidate")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, windows, mean, std = fit_first_fold(args.experiment_dir.resolve())
    export_candidate(model, windows, mean, std, args.output_dir)


if __name__ == "__main__":
    main()
