"""Scoring and statistical analysis."""

from ablation_harness.scoring.confidence import (
    analyze_sample_size,
    compute_confidence_interval,
)
from ablation_harness.scoring.metrics import compute_metrics, compute_span_iou
from ablation_harness.scoring.normalize import load_label_map, normalize_pii_type

__all__ = [
    "compute_metrics",
    "compute_span_iou",
    "normalize_pii_type",
    "load_label_map",
    "compute_confidence_interval",
    "analyze_sample_size",
]
