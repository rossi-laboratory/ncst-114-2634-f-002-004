"""Evaluation metrics for the four delivery goals."""
from evaluation.chamfer_distance import chamfer_distance
from evaluation.temporal_consistency import temporal_consistency
from evaluation.tracking_error import tracking_error

THRESHOLDS = {
    "tracking_success_rate": {"threshold": 0.90, "comparison": ">="},
    "temporal_error":        {"threshold": 0.10, "comparison": "<="},
    "id_switch_rate":        {"threshold": 0.08, "comparison": "<="},
    "chamfer_distance":      {"threshold": 0.05, "comparison": "<="},
}


def check_pass(metric_name: str, value: float) -> bool:
    """Return True iff ``value`` satisfies the threshold for ``metric_name``."""
    spec = THRESHOLDS[metric_name]
    if spec["comparison"] == ">=":
        return value >= spec["threshold"]
    return value <= spec["threshold"]


__all__ = [
    "chamfer_distance",
    "temporal_consistency",
    "tracking_error",
    "THRESHOLDS",
    "check_pass",
]
