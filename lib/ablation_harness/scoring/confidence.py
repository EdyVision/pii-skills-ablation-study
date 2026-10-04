"""Confidence interval and sample-size analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_interval(
    data: pd.Series, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Compute confidence interval for the mean.

    Returns ``(ci_low, ci_high, margin)``.
    """
    values = np.asarray(data, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Confidence intervals require finite observations")
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))

    mean = data.mean()
    se = stats.sem(data)
    if se <= 0 or np.isnan(se) or not np.isfinite(se):
        return (float(mean), float(mean), 0.0)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    margin = (ci[1] - ci[0]) / 2
    return (ci[0], ci[1], margin)


def analyze_sample_size(df: pd.DataFrame, target_margin: float = 0.05) -> pd.DataFrame:
    """Analyze if sample size is sufficient for each model × condition."""
    results = []

    for (model, condition), group in df.groupby(["model", "condition"]):
        ci_low, ci_high, margin = compute_confidence_interval(group["f1"])
        mean_f1 = group["f1"].mean()
        std_f1 = group["f1"].std()
        n = len(group)

        sufficient = n >= 2 and np.isfinite(margin) and margin <= target_margin

        if n >= 2 and np.isfinite(std_f1) and std_f1 > 0:
            t_crit = stats.t.ppf(0.975, df=n - 1)
            required_n = int(np.ceil((t_crit * std_f1 / target_margin) ** 2))
        else:
            required_n = n

        results.append(
            {
                "model": model,
                "condition": condition,
                "n": n,
                "mean_f1": round(mean_f1, 4),
                "std_f1": round(std_f1, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "margin": round(margin, 4),
                "sufficient": sufficient,
                "required_n": required_n,
            }
        )

    return pd.DataFrame(results)
