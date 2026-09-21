"""Run the shared split, fit models, and publish reproducible run artifacts."""

import json
import platform
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from .data import parse_date
from .features import FEATURE_COLUMNS
from .metrics import regression_metrics
from .models import TrainingOptions, fit_model, predict_volatility
from .splits import chronological_split, describe_split
from .storage import (
    git_provenance,
    load_feature_table,
    new_run_id,
    sha256,
    workspace_lock,
    write_json,
)

MODEL_NAMES = ("baseline", "ridge", "xgboost")


def train(
    root: Path,
    model="all",
    *,
    validation_start="2017-01-01",
    validation_end="2018-01-01",
    options=None,
):
    """Publish only complete invocations; one run ID covers all requested models."""
    options = options or TrainingOptions()
    options.validate()
    names = MODEL_NAMES if model == "all" else (model,)
    if any(name not in MODEL_NAMES for name in names):
        raise ValueError(f"Unknown model: {model}")
    start, end = parse_date(validation_start), parse_date(validation_end)
    with workspace_lock(root):
        table_path = root / "data/derived/feature_table.parquet"
        table, schema = load_feature_table(table_path)
        split = chronological_split(table, start, end)
        run_id = new_run_id()
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_provenance(root),
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
                },
            },
            "data_sha256": schema["table_sha256"],
            "source_sha256": {
                path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "feature_columns": FEATURE_COLUMNS,
            "target_column": "target_vol_5d",
            "prediction_rule": "clip_raw_prediction_at_zero",
            "metric_units": {
                "mse": "decimal_squared",
                "mae": "decimal",
                "rmse": "decimal",
                "r2": "unitless",
            },
            "validation_requested": [validation_start, validation_end],
            "training_options": asdict(options),
            "split": describe_split("outer", split),
            "models": [],
        }
        with tempfile.TemporaryDirectory(prefix=".train-", dir=root) as temporary:
            stage = Path(temporary)
            model_stage, result_stage = stage / "models", stage / "results"
            model_stage.mkdir()
            result_stage.mkdir()
            predictions = []
            for name in names:
                estimator, details = fit_model(name, split, options)
                metrics, clipped = {}, {}
                for split_name, frame in [("train", split.train), ("val", split.val)]:
                    predicted, clipped[split_name] = predict_volatility(
                        estimator, frame[FEATURE_COLUMNS]
                    )
                    metrics[split_name] = regression_metrics(
                        frame.target_vol_5d.to_numpy(), predicted
                    )
                    predictions.append(
                        pd.DataFrame(
                            {
                                "feature_date": frame.index,
                                "split": split_name,
                                "model": name,
                                "actual": frame.target_vol_5d.to_numpy(),
                                "predicted": predicted,
                            }
                        )
                    )
                suffix = "ubj" if name == "xgboost" else "joblib"
                filename = f"{name}-{run_id}.{suffix}"
                artifact = model_stage / filename
                if name == "xgboost":
                    estimator.save_model(artifact)
                else:
                    joblib.dump(estimator, artifact)
                record["models"].append(
                    {
                        "model": name,
                        "train_rows": len(split.train),
                        "val_rows": len(split.val),
                        **details,
                        "metrics": metrics,
                        "clipped_predictions": clipped,
                        "artifact": f"models/{run_id}/{filename}",
                        "artifact_sha256": sha256(artifact),
                    }
                )
            prediction_file = result_stage / "predictions.parquet"
            pd.concat(predictions, ignore_index=True).to_parquet(prediction_file, index=False)
            record["predictions"] = f"results/{run_id}/predictions.parquet"
            record["predictions_sha256"] = sha256(prediction_file)
            # A run manifest is the commit marker. Readers ignore directories
            # without one, including any interrupted publication.
            (root / "models").mkdir(exist_ok=True)
            (root / "results").mkdir(exist_ok=True)
            model_stage.rename(root / "models" / run_id)
            destination = root / "results" / run_id
            result_stage.rename(destination)
            log_lines = [
                json.dumps(
                    {
                        "run_id": run_id,
                        "created_at_utc": record["created_at_utc"],
                        "git": record["git"],
                        "versions": record["versions"],
                        "data_sha256": record["data_sha256"],
                        "source_sha256": record["source_sha256"],
                        "metric_units": record["metric_units"],
                        "split": record["split"],
                        "training_options": record["training_options"],
                        **item,
                    },
                    allow_nan=False,
                )
                + "\n"
                for item in record["models"]
            ]
            log_path = root / "results/train_log.jsonl"
            # Retain the old log offset so ordinary write failures cannot leave
            # partial records. The manifest remains the completion authority.
            with log_path.open("a+", encoding="utf-8") as stream:
                stream.seek(0, 2)
                offset = stream.tell()
                try:
                    stream.writelines(log_lines)
                    stream.flush()
                    write_json(destination / "run.json", record)
                except BaseException:
                    stream.seek(offset)
                    stream.truncate()
                    raise
            return record
