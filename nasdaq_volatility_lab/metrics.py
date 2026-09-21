"""Regression metrics in decimal units; presentation conversions are separate."""

import numpy as np


def regression_metrics(actual, predicted):
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("Metrics require aligned nonempty one-dimensional arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metric inputs must be finite.")
    errors = p - y
    mse = float(np.mean(errors**2))
    total = float(np.sum((y - y.mean()) ** 2))
    # Constant targets and single rows do not define informative R².
    r2 = (
        None if len(y) < 2 or np.ptp(y) == 0 or total == 0 else float(1 - np.sum(errors**2) / total)
    )
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }
