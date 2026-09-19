"""Quality reports must retain notes and reject material data errors."""

import unittest

import numpy as np
import pandas as pd

from nasdaq_volatility_lab.check_data import inspect_prices


class QualityChecks(unittest.TestCase):
    def test_rounding_note_is_not_a_material_price_violation(self):
        dates = pd.DatetimeIndex(["2024-01-02"])
        frame = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [np.nextafter(101.0, np.inf)], "Volume": [1000],
        }, index=dates)
        report = inspect_prices(frame, dates)
        self.assertTrue(report["passed"])
        self.assertEqual(report["findings"]["Close outside low/high (strict)"]["count"], 1)
        frame.loc[dates[0], "Close"] = 110.0
        self.assertFalse(inspect_prices(frame, dates)["passed"])
