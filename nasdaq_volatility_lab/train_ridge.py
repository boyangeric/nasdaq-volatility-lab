"""Train and evaluate a fixed-alpha Ridge model on the 2017 development fold.

Inputs: the validated feature table and schema in data/derived/.
Outputs: daily predictions, a fitted Pipeline, and metrics in artifacts/ridge_2017/.
All volatility values use annualized decimal units: 0.25 means 25%.
Importing this module performs no training or file I/O.
"""

from pathlib import Path
import hashlib
import json
import sys

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
import sklearn

# Direct file execution has no package context. Add the checkout root so it
# resolves the same package as `python -m nasdaq_volatility_lab.train_ridge`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nasdaq_volatility_lab.metrics import error_metrics
from nasdaq_volatility_lab.paths import ROOT
from nasdaq_volatility_lab.split_data import chronological_split
from nasdaq_volatility_lab.storage import load_feature_table


def make_ridge_pipeline() -> Pipeline:
    """Build an unfitted model; fit the entire Pipeline on training rows only."""
    # StandardScaler learns per-feature mean/scale; Ridge learns coefficients
    # and an intercept. fit() must receive training rows only for both steps.
    # Alpha controls the L2 coefficient penalty; SVD solves the linear problem.
    # These settings are fixed before evaluation, with no hyperparameter search.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, solver="svd")),
    ])


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit preprocessing and regression, then verify the scaler's training scope."""
    # Each training row pairs historical features with its observed future label.
    # Pipeline.fit standardizes X first, then fits Ridge against unscaled y.
    model = make_ridge_pipeline()
    model.fit(X_train, y_train)
    scaler = model.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(scaler.var_, X_train.var(ddof=0).to_numpy(), rtol=1e-12, atol=1e-14)
    if not (scaler.n_samples_seen_ == len(X_train)):
        raise ValueError('Scaler sample count differs from the training row count.')
    return model


def predict_validation(model: Pipeline, X_valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and nonnegative predictions without refitting preprocessing."""
    scaler = model.named_steps["scaler"]
    # Validation uses the training statistics; it must not update them.
    saved_mean, saved_scale = scaler.mean_.copy(), scaler.scale_.copy()
    raw_predictions = np.asarray(model.predict(X_valid), dtype=float)
    if not (np.isfinite(raw_predictions).all()):
        raise ValueError('Ridge produced non-finite predictions.')
    # Nonnegative postprocessing is part of evaluation, not Pipeline.predict.
    # Keep raw predictions as well so clipping remains auditable.
    predictions = np.maximum(raw_predictions, 0.0)
    np.testing.assert_array_equal(scaler.mean_, saved_mean)
    np.testing.assert_array_equal(scaler.scale_, saved_scale)

    return raw_predictions, predictions


def evaluate_ridge(model: Pipeline, evaluation: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pair validation predictions with outcomes and a same-date baseline.

    Targets are used here only for evaluation. They never enter predict().
    The baseline predicts today's trailing 20-day volatility without fitting.
    """
    X_valid = evaluation[columns]
    y_valid = evaluation["target_vol_5d"]
    raw_predictions, predictions = predict_validation(model, X_valid)
    results = pd.DataFrame({
        "raw_prediction": raw_predictions, "prediction": predictions,
        "actual": y_valid, "baseline_prediction": X_valid["qqq_vol_20d"],
        "label_end_date": evaluation["label_end_date"],
    }, index=X_valid.index)
    return results


def build_report(
    model: Pipeline, X_train: pd.DataFrame, results: pd.DataFrame, digest: str,
) -> dict:
    """Summarize errors and provenance without writing files.

    MAE/RMSE remain in decimal volatility units; multiply by 100 to display
    percentage points. Both models are scored on exactly the same dates.
    """
    scaler = model.named_steps["scaler"]
    ridge_mae, ridge_rmse = error_metrics(results["prediction"] - results["actual"])
    baseline_mae, baseline_rmse = error_metrics(results["baseline_prediction"] - results["actual"])
    comparison = pd.DataFrame([
        {"model": "historical_volatility", "mae": baseline_mae, "rmse": baseline_rmse},
        {"model": "ridge_alpha_1", "mae": ridge_mae, "rmse": ridge_rmse},
    ])
    return {
        "validation_year": 2017, "alpha": 1.0, "solver": "svd",
        "training_rows": len(X_train), "validation_rows": len(results),
        "training_start": str(X_train.index.min().date()),
        "training_end": str(X_train.index.max().date()),
        "feature_columns": X_train.columns.tolist(), "negative_predictions_clipped": int((results["raw_prediction"] < 0).sum()),
        "metrics": comparison.to_dict(orient="records"),
        "metric_units": "annualized volatility in decimals; multiply by 100 for percentage points",
        "scaler_fit": "training only", "scaler_samples_seen": int(scaler.n_samples_seen_),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "feature_table_sha256": digest, "sklearn_version": sklearn.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": "Fixed-alpha single-fold development run; no tuning or 2025 scoring.",
    }


def save_outputs(
    output_dir: Path, model: Pipeline, results: pd.DataFrame,
    report: dict, X_valid: pd.DataFrame,
) -> None:
    """Save three artifacts and verify that the serialized model predicts identically.

    The joblib file contains StandardScaler and Ridge, not the clipping rule.
    Call predict_validation() to apply the same nonnegative postprocessing.
    Re-running this workflow replaces these development artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_dir / "predictions.parquet")
    joblib.dump(model, output_dir / "development_pipeline.joblib")
    loaded = joblib.load(output_dir / "development_pipeline.joblib")
    np.testing.assert_allclose(loaded.predict(X_valid), results["raw_prediction"].to_numpy())
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Load → split → fit → evaluate → save; remain silent on success."""
    table_path = ROOT / "data" / "derived" / "feature_table.parquet"
    table, schema = load_feature_table(table_path)

    # Reserve 2017 for validation. The splitter removes five preceding trading
    # rows and checks that training labels end before validation begins.
    # No 2025 final-test scoring or within-validation-year retraining occurs.
    split = chronological_split(
        table, "2017-01-01", "2018-01-01", label_cutoff="2025-01-01"
    )
    columns = schema["feature_columns"]
    X_train = split.train[columns]
    y_train = split.train[schema["target_column"]]

    model = fit_ridge(X_train, y_train)
    results = evaluate_ridge(model, split.evaluation, columns)
    report = build_report(model, X_train, results, schema["table_sha256"])
    save_outputs(
        ROOT / "artifacts" / "ridge_2017",
        model, results, report, split.evaluation[columns],
    )


if __name__ == "__main__":
    main()
