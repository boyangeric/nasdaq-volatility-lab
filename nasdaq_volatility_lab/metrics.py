"""Shared error metrics, in the same units as the regression target."""

import numpy as np


def error_metrics(errors):
    """Reject missing/non-finite errors instead of silently omitting rows."""
    values = np.asarray(errors, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Errors must be a nonempty, finite one-dimensional sequence.")
    return float(np.abs(values).mean()), float(np.sqrt(np.square(values).mean()))
