"""Evaluate the historical-volatility baseline on the 2017 development fold."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .metrics import error_metrics
from .paths import ROOT
from .split_data import chronological_split
from .storage import load_feature_table


def main():
    derived_dir = ROOT / "data" / "derived"
    table_path = derived_dir / "feature_table.parquet"
    table, schema = load_feature_table(table_path)
    split = chronological_split(table, "2017-01-01", "2018-01-01", label_cutoff="2025-01-01")

    # A fixed rule: no fitting, scaling, or tuning. Only X supplies predictions.
    X_valid = split.evaluation[schema["feature_columns"]]
    y_pred = X_valid["qqq_vol_20d"].rename("prediction")
    y_true = split.evaluation[schema["target_column"]].rename("actual")
    if not (y_pred.index.equals(y_true.index)):
        raise ValueError('Prediction and target dates differ.')
    if not (len(y_pred) > 0):
        raise ValueError('Evaluation requires at least one prediction.')
    if not (np.isfinite(y_pred).all() and np.isfinite(y_true).all()):
        raise ValueError('Predictions and targets must be finite.')

    predictions = pd.concat([y_pred, y_true], axis=1)
    predictions["error"] = y_pred - y_true
    errors = predictions["error"]

    mae, rmse = error_metrics(errors)
    if not (rmse >= mae):
        raise ValueError('RMSE must not be smaller than MAE.')

    output_dir = ROOT / "artifacts" / "baseline_2017"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions["label_end_date"] = split.evaluation["label_end_date"]
    prediction_path = output_dir / "predictions.parquet"
    predictions.to_parquet(prediction_path, engine="pyarrow")
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(prediction_path))
    report = {
        "model": "20-day historical volatility baseline",
        "prediction_rule": "prediction = qqq_vol_20d; no fitting",
        "validation_year": 2017,
        "observations": len(predictions),
        "feature_date_start": str(predictions.index.min().date()),
        "feature_date_end": str(predictions.index.max().date()),
        "mae": mae,
        "rmse": rmse,
        "units": "annualized volatility in decimal form; multiply errors by 100 for percentage points",
        "error_sign": "prediction minus actual",
        "gap_sessions": 5,
        "feature_table_sha256": schema["table_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": "One validation year; overlapping five-day targets; not a final-test result.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
