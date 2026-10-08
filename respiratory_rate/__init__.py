"""Respiratory-rate estimation package."""

from respiratory_rate.multimodal_pipeline import (
    MultimodalRespirationEstimator,
    ECGRespirationEstimator,
)

__all__ = ["MultimodalRespirationEstimator", "ECGRespirationEstimator"]
