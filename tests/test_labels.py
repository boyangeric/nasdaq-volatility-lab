"""Check production label alignment using synthetic prices, without downloads."""

import math
import unittest

import numpy as np
import pandas as pd

from nasdaq_volatility_lab.features import build_feature_table


class LabelChecks(unittest.TestCase):
    def test_future_window_and_incomplete_tail(self):
        dates = pd.bdate_range(end="2005-01-31", periods=100)
        closes = 100 * np.exp(np.arange(100) * 0.001)
        prices = pd.DataFrame({"Close": closes, "Volume": 1000.0}, index=dates)
        table = build_feature_table(prices, prices.copy())
        date = table.index[3]
        position = dates.get_loc(date)
        future_prices = closes[position : position + 6]
        returns = [math.log(b / a) for a, b in zip(future_prices, future_prices[1:])]
        expected = math.sqrt(252 / 5 * math.fsum(r * r for r in returns))
        self.assertAlmostEqual(table.loc[date, "target_vol_5d"], expected)
        self.assertEqual(table.loc[date, "label_end_date"], dates[position + 5])
        self.assertTrue(table["target_vol_5d"].iloc[:-5].notna().all())
        self.assertTrue(table[["target_vol_5d", "label_end_date"]].iloc[-5:].isna().all().all())

        changed = prices.copy()
        changed.loc[dates[position + 5], "Close"] *= 1.1
        changed_table = build_feature_table(changed, prices)
        self.assertNotEqual(
            table.loc[date, "target_vol_5d"], changed_table.loc[date, "target_vol_5d"]
        )
