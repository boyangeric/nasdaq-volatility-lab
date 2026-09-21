"""Offline CLI integration, artifact round trips and failure recovery."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from xgboost import XGBRegressor

from nasdaq_volatility_lab.cli import main
from nasdaq_volatility_lab.data import fetch_data
from nasdaq_volatility_lab.features import FEATURE_COLUMNS
from nasdaq_volatility_lab.models import TrainingOptions, predict_volatility
from nasdaq_volatility_lab.storage import load_feature_table, load_run, sha256
from nasdaq_volatility_lab.training import train


def synthetic_download(ticker, parameters):
    dates = (
        mcal.get_calendar("NASDAQ")
        .schedule(
            start_date=parameters["start"],
            end_date=pd.Timestamp(parameters["end"]) - pd.Timedelta(days=1),
        )
        .index
    )
    step = np.arange(len(dates))
    close = 100 * np.exp(np.cumsum(0.0002 + 0.007 * np.sin(step / (3 if ticker == "QQQ" else 5))))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1000 + step % 27,
        },
        index=dates.as_unit("s"),
    )


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def cli(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--project-dir", str(self.root), *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def fetch(self):
        with patch("nasdaq_volatility_lab.data.download_prices", side_effect=synthetic_download):
            return fetch_data(self.root, "2014-01-01", "2018-03-01")

    def test_missing_data_and_missing_run_have_actionable_errors(self):
        code, output, error = self.cli("train")
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("fetch-data first", error)
        self.assertNotIn("Traceback", error)
        code, _, error = self.cli("show-results")
        self.assertEqual(code, 1)
        self.assertIn("train", error)
        code, _, error = self.cli("show-results", "--run-id", "../elsewhere")
        self.assertEqual(code, 1)
        self.assertIn("Invalid run ID", error)

    def test_fetch_train_show_and_reload_all_artifacts(self):
        with patch("nasdaq_volatility_lab.data.download_prices", side_effect=synthetic_download):
            code, output, error = self.cli(
                "fetch-data", "--start-date", "2014-01-01", "--end-date", "2018-03-01"
            )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(len(output.splitlines()), 1)
        self.assertIn("features=qqq_return_1d", output)
        # A training run must not silently fetch or refresh the user's data.
        with patch(
            "nasdaq_volatility_lab.data.download_prices",
            side_effect=AssertionError("Unexpected network"),
        ):
            code, output, error = self.cli(
                "train", "--n-estimators", "20", "--early-stopping-rounds", "3"
            )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(len(output.splitlines()), 3)
        run = load_run(self.root)
        table, _ = load_feature_table(self.root / "data/derived/feature_table.parquet")
        predictions = pd.read_parquet(self.root / run["predictions"])
        records = [
            json.loads(line)
            for line in (self.root / "results/train_log.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(records), 3)
        self.assertEqual({row["run_id"] for row in records}, {run["run_id"]})
        for row in run["models"]:
            self.assertEqual(sha256(self.root / row["artifact"]), row["artifact_sha256"])
            if row["model"] == "xgboost":
                estimator = XGBRegressor()
                estimator.load_model(self.root / row["artifact"])
                self.assertEqual(
                    estimator.get_booster().num_boosted_rounds(), row["parameters"]["n_estimators"]
                )
            else:
                estimator = joblib.load(self.root / row["artifact"])
            saved = predictions.loc[predictions.model.eq(row["model"])]
            restored, _ = predict_volatility(
                estimator, table.loc[saved.feature_date, FEATURE_COLUMNS]
            )
            np.testing.assert_allclose(restored, saved.predicted, atol=1e-10)
            self.assertEqual(
                row["metrics"],
                next(item["metrics"] for item in records if item["model"] == row["model"]),
            )
            if row["model"] == "baseline":
                target_mean = saved.loc[saved.split.eq("train"), "actual"].mean()
                np.testing.assert_allclose(restored, target_mean)
            if row["model"] == "ridge":
                train_dates = saved.loc[saved.split.eq("train"), "feature_date"]
                np.testing.assert_allclose(
                    estimator.named_steps["standardscaler"].mean_,
                    table.loc[train_dates, FEATURE_COLUMNS].mean(),
                )
        code, output, error = self.cli(
            "show-results", "--run-id", run["run_id"], "--metric", "mae", "--plot"
        )
        self.assertEqual((code, error), (0, ""))
        self.assertIn("MAE (pp)", output)
        self.assertNotIn("MSE (", output)
        chart = self.root / "results" / run["run_id"] / "comparison.png"
        self.assertEqual(chart.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        code, _, _ = self.cli("train", "--model", "ridge", "--alpha", "3")
        self.assertEqual(code, 0)
        latest = load_run(self.root)
        self.assertNotEqual(latest["run_id"], run["run_id"])
        self.assertEqual([row["model"] for row in latest["models"]], ["ridge"])
        self.assertEqual(latest["models"][0]["parameters"]["alpha"], 3)
        self.assertEqual(len(load_run(self.root, run["run_id"])["models"]), 3)

    def test_failed_refresh_retains_previous_data_and_unlocks(self):
        self.fetch()
        path = self.root / "data/derived/feature_table.parquet"
        previous = path.read_bytes()

        def fail_spy(ticker, parameters):
            if ticker == "SPY":
                raise RuntimeError("Provider unavailable")
            return synthetic_download(ticker, parameters)

        with patch("nasdaq_volatility_lab.data.download_prices", side_effect=fail_spy):
            code, _, error = self.cli(
                "fetch-data", "--start-date", "2014-01-01", "--end-date", "2018-03-01"
            )
        self.assertEqual(code, 1)
        self.assertIn("Provider unavailable", error)
        self.assertEqual(path.read_bytes(), previous)
        self.assertFalse((self.root / ".nasdaq-volatility-lab.lock").exists())
        self.fetch()
        backups = list(
            (self.root / "artifacts/data_backups").glob("*/derived/feature_table.parquet")
        )
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), previous)

    def test_failed_training_publication_does_not_replace_latest_or_append_log(self):
        self.fetch()
        previous = train(self.root, "baseline")
        log = self.root / "results/train_log.jsonl"
        old_bytes = log.read_bytes()
        with patch("nasdaq_volatility_lab.training.write_json", side_effect=OSError("Disk full")):
            code, _, error = self.cli("train", "--model", "ridge")
        self.assertEqual(code, 1)
        self.assertIn("Disk full", error)
        self.assertEqual(load_run(self.root)["run_id"], previous["run_id"])
        self.assertEqual(log.read_bytes(), old_bytes)
        self.assertFalse((self.root / ".nasdaq-volatility-lab.lock").exists())

    def test_invalid_options_and_insufficient_inner_history(self):
        for args in [
            ("--alpha", "nan"),
            ("--learning-rate", "-1"),
            ("--n-estimators", "0"),
            ("--max-depth", "0"),
            ("--reg-lambda", "-1"),
        ]:
            code, _, error = self.cli("train", *args)
            self.assertEqual(code, 1)
            self.assertNotIn("fetch-data first", error)
        self.fetch()
        with self.assertRaisesRegex(ValueError, "history before its inner"):
            train(
                self.root,
                "xgboost",
                validation_start="2014-08-01",
                validation_end="2015-01-01",
                options=TrainingOptions(n_estimators=2),
            )
