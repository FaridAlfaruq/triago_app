"""Pipeline Respiratory Rate TriaGo (ECG + PPG Multimodal Smart Fusion).

Berdasarkan:
- Peter H. Charlton et al., IEEE Reviews in Biomedical Engineering, 2018.
- Drew A. Birrenkott, Marco A. F. Pimentel, et al., IEEE TBME, 2018.
"""

from respiratory_rate.multimodal_pipeline import (
    MultimodalRespirationEstimator,
    ECGRespirationEstimator,
)

__all__ = ["MultimodalRespirationEstimator", "ECGRespirationEstimator"]
