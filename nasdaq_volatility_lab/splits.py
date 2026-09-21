"""Chronological splits with explicit target-availability boundaries."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimeSplit:
    train: pd.DataFrame
    gap: pd.DataFrame
    evaluation: pd.DataFrame
    excluded_evaluation: pd.DataFrame

    @property
    def val(self) -> pd.DataFrame:
        """Alias for the eligible chronological validation rows."""
        return self.evaluation


def chronological_split(table, start, end, *, label_cutoff=None, gap_sessions=5):
    """Use [start, end) dates and optionally require labels before a cutoff.

    Count gap sessions BEFORE filtering labels. The fixed five-day target
    requires at least five gap sessions. Input must contain all trading rows.
    """
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex of feature dates.")
    if not table.index.is_unique or not table.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be sorted and unique.")
    if not isinstance(gap_sessions, int) or gap_sessions < 5:
        raise ValueError("The five-day target requires a gap of at least five sessions.")
    observed = table["target_vol_5d"].dropna()
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Observed volatility labels must be finite and nonnegative.")
    if not table["target_vol_5d"].isna().equals(table["label_end_date"].isna()):
        raise ValueError("Target and label-end availability must match.")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end:
        raise ValueError("Evaluation start must precede end.")
    candidates = table.loc[(table.index >= start) & (table.index < end)]
    if candidates.empty:
        raise ValueError("No evaluation dates in the requested interval.")
    evaluation_start = candidates.index[0]
    earlier = table.loc[table.index < evaluation_start]
    if len(earlier) <= gap_sessions:
        raise ValueError("Insufficient training history after the gap.")
    train = earlier.iloc[:-gap_sessions].copy()
    gap = earlier.iloc[-gap_sessions:].copy()
    if train[["target_vol_5d", "label_end_date"]].isna().any().any():
        raise ValueError("Training rows contain incomplete labels.")
    if not (train["label_end_date"] < evaluation_start).all():
        raise ValueError("Training labels reach the evaluation period.")
    if not (train["label_end_date"] > train.index).all():
        raise ValueError("Training labels must end after their feature dates.")
    eligible = candidates[["target_vol_5d", "label_end_date"]].notna().all(axis=1)
    if label_cutoff is not None:
        eligible &= candidates["label_end_date"] < pd.Timestamp(label_cutoff)
    evaluation = candidates.loc[eligible].copy()
    excluded = candidates.loc[~eligible].copy()
    if evaluation.empty:
        raise ValueError("No complete evaluation labels within the allowed boundary.")
    if not (evaluation["label_end_date"] > evaluation.index).all():
        raise ValueError("Evaluation labels must end after their feature dates.")
    return TimeSplit(train, gap, evaluation, excluded)


def describe_split(name, split):
    """Record dates and counts only; do not calculate model performance."""

    def dates(frame):
        return frame.index.strftime("%Y-%m-%d").tolist()

    return {
        "name": name,
        "train_rows": len(split.train),
        "train_start": dates(split.train)[0],
        "train_end": dates(split.train)[-1],
        "latest_training_label_end": str(split.train.label_end_date.max().date()),
        "gap_dates": dates(split.gap),
        "evaluation_rows": len(split.evaluation),
        "evaluation_start": dates(split.evaluation)[0],
        "evaluation_end": dates(split.evaluation)[-1],
        "latest_evaluation_label_end": str(split.evaluation.label_end_date.max().date()),
        "excluded_evaluation_dates": dates(split.excluded_evaluation),
    }
