"""Render saved runs without fitting models or accessing market-data providers."""

import logging
import os
from pathlib import Path

import pandas as pd

from .storage import sha256

METRIC_LABELS = {"mse": "MSE (pp²)", "mae": "MAE (pp)", "rmse": "RMSE (pp)", "r2": "R²"}
METRIC_SCALES = {"mse": 10000, "mae": 100, "rmse": 100, "r2": 1}


def render_results(record, metric=None):
    metrics = [metric] if metric else list(METRIC_LABELS)
    models = record["models"]
    if metric:

        def order(item):
            value = item["metrics"]["val"][metric]
            return (value is None, -value if metric == "r2" and value is not None else value or 0)

        models = sorted(models, key=order)
    headers = ["Model", "Split", *[METRIC_LABELS[name] for name in metrics], "Train time (s)"]
    rows = []
    for item in models:
        for split in ("train", "val"):
            values = []
            for name in metrics:
                value = item["metrics"][split][name]
                values.append("N/A" if value is None else f"{value * METRIC_SCALES[name]:.4f}")
            rows.append([item["model"], split, *values, f"{item['training_seconds']:.4f}"])
    widths = [max(len(row[index]) for row in [headers, *rows]) for index in range(len(headers))]
    lines = [
        "  ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in [headers, *rows]
    ]
    return "\n".join(lines)


def plot_results(root: Path, record):
    """Compare the same validation dates; annualized volatility levels use %."""
    # Import plotting only on demand. Agg also works on headless machines.
    cache = root / "results/.matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    logger = logging.getLogger("matplotlib.font_manager")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        import matplotlib.pyplot as plt
    finally:
        logger.setLevel(old_level)

    path = root / record["predictions"]
    if not path.exists() or sha256(path) != record["predictions_sha256"]:
        raise ValueError(
            "Saved predictions are missing or changed. Rerun train to create a complete run."
        )
    predictions = pd.read_parquet(path)
    validation = predictions.loc[predictions["split"].eq("val")]
    actual = validation.drop_duplicates("feature_date").sort_values("feature_date")
    figure, axes = plt.subplots(figsize=(12, 5), layout="constrained")
    try:
        axes.plot(
            actual.feature_date, actual.actual * 100, color="black", linewidth=1.8, label="Actual"
        )
        for item in record["models"]:
            values = validation.loc[validation.model.eq(item["model"])].sort_values("feature_date")
            axes.plot(
                values.feature_date,
                values.predicted * 100,
                linewidth=1.1,
                alpha=0.85,
                label=item["model"],
            )
        axes.set(
            xlabel="Prediction date",
            ylabel="Annualized volatility (%)",
            title=f"Validation predictions · {record['run_id']}",
        )
        axes.legend()
        axes.grid(alpha=0.2)
        output = root / "results" / record["run_id"] / "comparison.png"
        temporary = output.with_name("comparison.tmp.png")
        figure.savefig(temporary, dpi=160)
        temporary.replace(output)
        return output
    finally:
        plt.close(figure)
