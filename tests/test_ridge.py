"""Verify preprocessing boundaries with a deliberately shifted holdout."""

import unittest
import numpy as np
import pandas as pd
from nasdaq_volatility_lab.train_ridge import make_ridge_pipeline
from nasdaq_volatility_lab.metrics import error_metrics


class RidgeChecks(unittest.TestCase):
    def test_scaler_uses_only_training_and_predict_does_not_refit(self):
        train = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [10.0, 20.0, 30.0]})
        holdout = pd.DataFrame({"a": [1000.0], "b": [-1000.0]})
        model = make_ridge_pipeline().fit(train, [0.1, 0.2, 0.3])
        scaler = model.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, [2.0, 20.0])
        self.assertEqual(scaler.n_samples_seen_, 3)
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
        model.predict(holdout)
        np.testing.assert_array_equal(scaler.mean_, mean)
        np.testing.assert_array_equal(scaler.scale_, scale)

    def test_metric_units_and_rejection_of_missing_values(self):
        mae, rmse = error_metrics([0.02, -0.04])
        self.assertAlmostEqual(mae, 0.03)
        self.assertAlmostEqual(rmse, np.sqrt(0.001))
        for invalid in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                error_metrics(invalid)
