"""
models.py
Deep learning model architectures for pure-waveform cuffless blood pressure estimation.

Architecture: CalBPNet (Pure-Waveform CNN-BiGRU-Attention)
Components:
1. Multi-scale 1D Inception Stem (Conv1D + Inception Blocks with parallel kernels: 1, 3, 7, 15).
2. Temporal BiGRU (2-layer Bidirectional GRU for cardiac cycle temporal dynamics).
3. Multi-Head Self-Attention (MHSA for inter-beat dependency modeling).
4. Dual Global Context Pooling (Concatenated Global Average + Global Max Pooling = 256-dim context).
5. Dual Regression Heads (Separate MLP branches for Systolic and Diastolic BP).
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class InceptionBlock1D(nn.Module):
    """
    Multi-scale 1D Inception module with parallel temporal kernel branches (1, 3, 7, 15).
    Captures fast QRS spikes and slower PPG pulse wave reflections simultaneously.
    """
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
    """
    Pure-Waveform CNN-BiGRU-Attention Architecture for Cuffless Blood Pressure Estimation.
    Takes 10s physiological waveform tensor of shape (batch, 1250, in_channels)
    and predicts continuous SBP and DBP.
    """
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
        self.in_channels = in_channels

        # Stage 1: Stem & Multi-Scale Inception Feature Extractor
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

        # Stage 2: Temporal Recurrent & Attention Modeling
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

        # Stage 3: Dual Global Context Pooling (Avg + Max = 256 dimensions)
        context_dim = gru_hid * 4
        self.context_drop = nn.Dropout(0.20)

        # Stage 4: SBP & DBP Regression Heads
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
        """
        Forward pass.
        Input:
            x: Tensor of shape (batch, 1250, in_channels) or (batch, in_channels, 1250)
        Returns:
            sbp_pred: Shape (batch,)
            dbp_pred: Shape (batch,)
        """
        if x.shape[1] == 1250:
            x = x.transpose(1, 2)

        # 1D Multi-scale CNN hierarchy
        x = self.stem(x)
        x = self.drop1(self.pool1(self.inc1(x)))
        x = self.drop2(self.pool2(self.inc2(x)))
        x = self.drop3(self.pool3(self.inc3(x)))

        # Temporal BiGRU + Self-Attention
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        attn_out, _ = self.mhsa(gru_out, gru_out, gru_out)
        x = self.ln(gru_out + attn_out)

        # Dual Global Context Pooling
        avg_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        context = self.context_drop(torch.cat([avg_pool, max_pool], dim=1))

        # BP heads
        sbp = self.sbp_head(context).squeeze(-1)
        dbp = self.dbp_head(context).squeeze(-1)
        return sbp, dbp


class PhysicsBHSLoss(nn.Module):
    """
    Physics-informed loss function for blood pressure estimation:
    Combines Smooth L1 Huber loss with Mean Arterial Pressure (MAP) and Pulse Pressure (PP) constraints.
    """
    def __init__(self, s_mean: float = 120.0, s_std: float = 20.0, d_mean: float = 70.0, d_std: float = 12.0):
        super().__init__()
        self.s_mean = s_mean
        self.s_std = s_std
        self.d_mean = d_mean
        self.d_std = d_std
        self.huber = nn.SmoothL1Loss(beta=0.20)

    def forward(
        self,
        s_pred: torch.Tensor,
        d_pred: torch.Tensor,
        s_true: torch.Tensor,
        d_true: torch.Tensor,
    ) -> torch.Tensor:
        loss_huber = self.huber(s_pred, s_true) + self.huber(d_pred, d_true)

        # Mean Arterial Pressure (MAP = (SBP + 2*DBP)/3)
        map_pred = (s_pred + 2.0 * d_pred) / 3.0
        map_true = (s_true + 2.0 * d_true) / 3.0
        map_err = torch.mean(torch.abs(map_pred - map_true)) / 15.0

        # Pulse Pressure (PP = SBP - DBP)
        pp_pred = s_pred - d_pred
        pp_true = s_true - d_true
        pp_err = torch.mean(torch.abs(pp_pred - pp_true)) / 15.0

        return loss_huber + 0.15 * map_err + 0.10 * pp_err


def build_model(in_channels: int = 7) -> CalBPNet:
    """Helper factory to instantiate CalBPNet."""
    return CalBPNet(in_channels=in_channels)


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = (2, 1250, 7)) -> None:
    """
    Print comprehensive parameter information, layer breakdown, and forward pass verification.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    param_size_mb = (total_params * 4) / (1024 * 1024)

    # Layer breakdown
    named_blocks = [
        ("Stem (Conv1D + BN + SiLU)", model.stem),
        ("Inception Block 1 (Kernels 1, 3, 7, 15)", model.inc1),
        ("Inception Block 2 (Kernels 1, 3, 7, 15)", model.inc2),
        ("Inception Block 3 (Kernels 1, 3, 7, 15)", model.inc3),
        ("BiGRU (2-Layer Bidirectional, 128 hidden)", model.bigru),
        ("MHSA (4-Head Self-Attention)", model.mhsa),
        ("LayerNorm (128-dim)", model.ln),
        ("SBP Regression Head (Linear-BN-SiLU-Linear)", model.sbp_head),
        ("DBP Regression Head (Linear-BN-SiLU-Linear)", model.dbp_head),
    ]

    print("=" * 70)
    print("CALBPNET ARCHITECTURE & PARAMETER SUMMARY")
    print("=" * 70)
    print(f"{'Component / Sub-Module':<45s} | {'Parameters':>12s} | {'% Total':>7s}")
    print("-" * 70)

    for name, block in named_blocks:
        count = sum(p.numel() for p in block.parameters())
        pct = (count / total_params) * 100.0
        print(f"{name:<45s} | {count:>12,d} | {pct:>6.2f}%")

    print("-" * 70)
    print(f"Total Parameters       : {total_params:,}")
    print(f"Trainable Parameters   : {trainable_params:,}")
    print(f"Non-Trainable Params   : {non_trainable:,}")
    print(f"Model Memory Size (FP32: {param_size_mb:.2f} MB")
    print("=" * 70)

    # Verification Forward Pass
    model.eval()
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        sbp_out, dbp_out = model(dummy_input)

    print("\nFORWARD PASS VERIFICATION:")
    print(f"  Input Tensor Shape   : {list(dummy_input.shape)} [Batch, Samples, Channels]")
    print(f"  Output SBP Shape     : {list(sbp_out.shape)} [Batch]")
    print(f"  Output DBP Shape     : {list(dbp_out.shape)} [Batch]")
    print(f"  Sample Prediction SBP: {sbp_out.numpy().round(2).tolist()} mmHg")
    print(f"  Sample Prediction DBP: {dbp_out.numpy().round(2).tolist()} mmHg")
    print("  Status               : PASSED (Forward graph fully functional)\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CalBPNet model definition & parameter info.")
    parser.add_argument("--channels", type=int, default=7, help="Input waveform channels (default: 7)")
    args = parser.parse_args()

    model = build_model(in_channels=args.channels)
    print_model_summary(model, input_shape=(2, 1250, args.channels))
