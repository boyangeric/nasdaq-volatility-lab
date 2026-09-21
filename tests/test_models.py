"""Prevent validation leakage and verify selection/refit semantics."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_volatility_lab.features import FEATURE_COLUMNS
from nasdaq_volatility_lab.metrics import regression_metrics
from nasdaq_volatility_lab.models import TrainingOptions, fit_model
from nasdaq_volatility_lab.reporting import render_results
from nasdaq_volatility_lab.splits import chronological_split


class ModelTests(unittest.TestCase):
    def test_xgboost_uses_only_inner_validation_then_refits_on_all_training(self):
        dates = pd.bdate_range("2014-01-01", "2017-03-01")
        table = pd.DataFrame(
            {name: np.arange(len(dates)) * 0.01 for name in FEATURE_COLUMNS}, index=dates
        )
        table["target_vol_5d"] = 0.2
        table["label_end_date"] = pd.Series(dates, index=dates).shift(-5)
        table.loc[dates[-5:], "target_vol_5d"] = np.nan
        split = chronological_split(table, "2017-01-01", "2018-01-01")
        fits = []

        class FakeXGB:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.best_iteration = 2
                self.best_score = 0.04

            def fit(self, features, target, **kwargs):
                fits.append((self.kwargs, features, target, kwargs))

            def evals_result(self):
                return {"validation_0": {"mae": [0.06, 0.05, 0.04, 0.05, 0.06]}}

        with patch("nasdaq_volatility_lab.models.XGBRegressor", FakeXGB):
            _, detail = fit_model(
                "xgboost", split, TrainingOptions(n_estimators=20, early_stopping_rounds=2)
            )
        self.assertEqual(len(fits), 4)
        for params, features, target, kwargs in fits[:-1]:
            inner_features, inner_target = kwargs["eval_set"][0]
            self.assertTrue(features.index.isin(split.train.index).all())
            self.assertTrue(inner_features.index.isin(split.train.index).all())
            self.assertFalse(inner_features.index.isin(split.val.index).any())
            self.assertLess(
                table.loc[features.index, "label_end_date"].max(), inner_features.index.min()
            )
            self.assertEqual(params["early_stopping_rounds"], 2)
        final_params, final_features, final_target, final_kwargs = fits[-1]
        pd.testing.assert_frame_equal(final_features, split.train[FEATURE_COLUMNS])
        pd.testing.assert_series_equal(final_target, split.train.target_vol_5d)
        self.assertEqual(final_kwargs, {})
        self.assertNotIn("early_stopping_rounds", final_params)
        self.assertEqual(final_params["n_estimators"], 3)
        # Equal inner MAE prefers lower depth and then stronger L2 penalty.
        self.assertEqual((final_params["max_depth"], final_params["reg_lambda"]), (2, 10))
        self.assertEqual(detail["selection"]["selected_candidate"], 2)
        fits.clear()
        with patch("nasdaq_volatility_lab.models.XGBRegressor", FakeXGB):
            fit_model("xgboost", split, TrainingOptions(max_depth=4, reg_lambda=5))
        self.assertEqual(len(fits), 2)  # one unique candidate plus final refit
        self.assertEqual((fits[-1][0]["max_depth"], fits[-1][0]["reg_lambda"]), (4, 5))

    def test_metric_values_units_sorting_and_undefined_r2(self):
        values = regression_metrics(np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.2, 0.2]))
        self.assertAlmostEqual(values["mse"], 0.02 / 3)
        self.assertAlmostEqual(values["mae"], 0.2 / 3)
        self.assertAlmostEqual(values["rmse"], np.sqrt(0.02 / 3))
        self.assertAlmostEqual(values["r2"], 0)
        self.assertIsNone(regression_metrics([0.1, 0.1], [0.2, 0.2])["r2"])
        self.assertIsNone(regression_metrics([0.1], [0.1])["r2"])
        self.assertIsNone(regression_metrics([0.1] * 3, [0.2] * 3)["r2"])
        with self.assertRaises(ValueError):
            regression_metrics([np.nan], [0.1])
        rows = []
        for name, mae, r2 in [
            ("worse", 0.08, -1),
            ("undefined", 0.05, None),
            ("better", 0.04, 0.5),
        ]:
            metrics = {"mse": 0.0016, "mae": mae, "rmse": 0.04, "r2": r2}
            rows.append(
                {
                    "model": name,
                    "metrics": {"train": metrics, "val": metrics},
                    "training_seconds": 1,
                }
            )
        record = {"models": rows}
        output = render_results(record)
        self.assertIn("16.0000", output)  # decimal MSE * 10,000 = pp squared
        self.assertIn("4.0000", output)
        self.assertIn("N/A", output)
        by_mae = render_results(record, "mae")
        self.assertLess(by_mae.index("better"), by_mae.index("worse"))
        by_r2 = render_results(record, "r2")
        self.assertLess(by_r2.index("better"), by_r2.index("worse"))
        self.assertLess(by_r2.index("worse"), by_r2.index("undefined"))
