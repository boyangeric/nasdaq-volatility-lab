"""Chronological splits and an audit of the fixed research protocol."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .paths import ROOT
from .storage import load_feature_table


@dataclass
class TimeSplit:
    train: pd.DataFrame
    gap: pd.DataFrame
    evaluation: pd.DataFrame
    excluded_evaluation: pd.DataFrame


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


def main():
    derived_dir = ROOT / "data" / "derived"
    table_path = derived_dir / "feature_table.parquet"
    table, schema = load_feature_table(table_path)
    digest = schema["table_sha256"]
    feature_columns = schema["feature_columns"]
    audit = {
        "table_sha256": digest, "gap_sessions": 5,
        "test_start": "2025-01-01", "feature_columns": feature_columns,
        "development_label_cutoff": "2025-01-01", "folds": [],
    }
    previous_train_dates = None
    for year in range(2017, 2025):
        outer = chronological_split(
            table, f"{year}-01-01", f"{year + 1}-01-01", label_cutoff="2025-01-01"
        )
        # Reserve the last calendar year INSIDE outer training for future tuning.
        inner = chronological_split(
            outer.train, f"{year - 1}-01-01", f"{year}-01-01",
            label_cutoff=outer.evaluation.index[0],
        )
        if not (inner.train.index.isin(outer.train.index).all()):
            raise ValueError('Inner training extends beyond outer training.')
        if not (inner.evaluation.index.isin(outer.train.index).all()):
            raise ValueError('Inner validation extends beyond outer training.')
        if previous_train_dates is not None:
            if not (previous_train_dates.isin(outer.train.index).all()):
                raise ValueError('Expanding-window training unexpectedly lost earlier dates.')
        previous_train_dates = outer.train.index
        record = describe_split(f"validation_{year}", outer)
        record["inner"] = describe_split(f"inner_{year - 1}", inner)
        audit["folds"].append(record)
    # Boundary audit only: no fitting, predictions, or test scoring.
    final = chronological_split(table, "2025-01-01", "2026-01-01")
    audit["final_test"] = describe_split("final_test_2025", final)
    audit["status"] = "boundaries_verified_no_models_trained_or_scored"
    output = derived_dir / "split_manifest.json"
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
