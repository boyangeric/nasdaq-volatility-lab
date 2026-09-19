"""Synthetic feature contracts plus optional local-snapshot regression coverage."""

import math
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from nasdaq_volatility_lab.build_table import (
    build_feature_table, cumulative_log_return, historical_volatility,
    volatility_ratio, rolling_drawdown, moving_average_deviation, relative_volume,
)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deterministic, nonconstant inputs exercise every rolling window offline.
        dates = pd.bdate_range("2004-07-01", "2010-02-01")
        steps = np.arange(len(dates))
        cls.prices = pd.DataFrame({
            "Close": 100 * np.exp(np.cumsum(0.001 + 0.005 * np.sin(steps))),
            "Volume": 1000.0 + steps % 31,
        }, index=dates)
        cls.spy_prices = pd.DataFrame({
            "Close": 80 * np.exp(np.cumsum(0.0005 + 0.003 * np.cos(steps))),
        }, index=dates)
        cls.table = build_feature_table(cls.prices, cls.spy_prices)

    def test_preserves_both_inputs(self):
        qqq, spy = self.prices.copy(deep=True), self.spy_prices.copy(deep=True)
        build_feature_table(self.prices, self.spy_prices)
        pd.testing.assert_frame_equal(self.prices, qqq)
        pd.testing.assert_frame_equal(self.spy_prices, spy)

    def test_rejects_invalid_source_prices_and_volume(self):
        for column, value in (("Close", -1.0), ("Close", np.inf), ("Volume", np.nan)):
            invalid = self.prices.copy()
            invalid.loc[invalid.index[0], column] = value
            with self.assertRaises(ValueError):
                build_feature_table(invalid, self.spy_prices)
        with self.assertRaises(ValueError):
            build_feature_table(self.prices, self.spy_prices.iloc[1:])

    def test_label_from_six_prices(self):
        prices, table = self.prices, self.table
        first_date = table.index[0]
        position = prices.index.get_loc(first_date)
        # Independently reconstruct the first study label from six closing prices.
        window_prices = prices["Close"].iloc[position:position + 6].tolist()
        window_returns = [
            math.log(current / previous)
            for previous, current in zip(window_prices, window_prices[1:])
        ]
        manual_target = math.sqrt(252 / 5 * math.fsum(r * r for r in window_returns))
        assert math.isclose(table.loc[first_date, "target_vol_5d"], manual_target, rel_tol=1e-12)
        assert table.loc[first_date, "label_end_date"] == prices.index[position + 5]

    def test_cumulative_returns(self):
        prices, table = self.prices, self.table
        daily_return = cumulative_log_return(prices["Close"], 1)
        # Log returns add across time: compare endpoint ratios with rolling sums.
        # Changing prices AFTER a cutoff must not affect features ON or BEFORE it.
        cutoff = pd.Timestamp("2010-01-04")
        changed_close = prices["Close"].copy()
        changed_close.loc[changed_close.index > cutoff] *= 1.5
        for periods in (5, 20):
            actual = table[f"qqq_return_{periods}d"]
            summed = daily_return.rolling(periods, min_periods=periods).sum().loc[table.index]
            np.testing.assert_allclose(actual, summed, rtol=1e-10, atol=1e-14)
            original = cumulative_log_return(prices["Close"], periods).loc[:cutoff]
            changed = cumulative_log_return(changed_close, periods).loc[:cutoff]
            pd.testing.assert_series_equal(original, changed)

    def test_volatility(self):
        prices, table = self.prices, self.table
        daily_return = cumulative_log_return(prices["Close"], 1)
        cutoff = pd.Timestamp("2010-01-04")
        changed_close = prices["Close"].copy()
        changed_close.loc[changed_close.index > cutoff] *= 1.5
        # Known-size returns have known RMS; a flat market has an undefined 0/0 ratio.
        toy_returns = pd.Series([0.01, -0.01] * 30)
        changed_returns = cumulative_log_return(changed_close, 1)
        for window in (5, 20, 60):
            toy_vol = historical_volatility(toy_returns, window)
            assert toy_vol.iloc[:window - 1].isna().all()
            np.testing.assert_allclose(toy_vol.iloc[window - 1:], 0.01 * math.sqrt(252))
            pd.testing.assert_series_equal(
                historical_volatility(daily_return, window).loc[:cutoff],
                historical_volatility(changed_returns, window).loc[:cutoff],
            )
        flat_returns = pd.Series([0.0] * 60)
        flat_ratio = volatility_ratio(
            historical_volatility(flat_returns, 5), historical_volatility(flat_returns, 20)
        )
        assert flat_ratio.isna().all()
        pd.testing.assert_series_equal(
            table.loc[:cutoff, "qqq_vol_ratio_5_20"],
            volatility_ratio(
                historical_volatility(changed_returns, 5),
                historical_volatility(changed_returns, 20),
            ).reindex(table.index).loc[:cutoff],
            check_names=False,
        )

    def test_price_position(self):
        prices, table = self.prices, self.table
        cutoff = pd.Timestamp("2010-01-04")
        changed_close = prices["Close"].copy()
        changed_close.loc[changed_close.index > cutoff] *= 1.5
        # A recovered price can have zero CURRENT drawdown despite a prior decline.
        toy_close = pd.Series([100.0, 80.0, 100.0])
        assert math.isclose(rolling_drawdown(toy_close, 2).iloc[1], -0.2)
        assert rolling_drawdown(toy_close, 3).iloc[2] == 0
        assert math.isclose(moving_average_deviation(toy_close, 2).iloc[1], 80 / 90 - 1)
        assert table["qqq_drawdown_60d"].between(-1, 0, inclusive="right").all()
        for function, window in ((rolling_drawdown, 60), (moving_average_deviation, 20)):
            assert function(prices["Close"], window).iloc[:window - 1].isna().all()
            assert function(pd.Series([0.0] * window), window).isna().all()
            pd.testing.assert_series_equal(
                function(prices["Close"], window).loc[:cutoff],
                function(changed_close, window).loc[:cutoff],
            )

    def test_relative_volume(self):
        prices, table = self.prices, self.table
        first_date = table.index[0]
        position = prices.index.get_loc(first_date)
        cutoff = pd.Timestamp("2010-01-04")
        # A volume spike is compared with earlier sessions, not with a mean containing itself.
        toy_volume = pd.Series([100.0] * 20 + [200.0])
        toy_relative_volume = relative_volume(toy_volume)
        assert toy_relative_volume.iloc[:20].isna().all()
        assert toy_relative_volume.iloc[20] == 2.0
        assert pd.isna(relative_volume(pd.Series([0.0] * 20 + [100.0])).iloc[20])
        assert relative_volume(pd.Series([100.0] * 20 + [0.0])).iloc[20] == 0.0
        manual_relative_volume = prices["Volume"].iloc[position] / (
            math.fsum(prices["Volume"].iloc[position - 20:position]) / 20
        )
        assert math.isclose(
            table.loc[first_date, "qqq_relative_volume_20d"], manual_relative_volume, rel_tol=1e-12
        )
        changed_volume = prices["Volume"].copy()
        changed_volume.loc[changed_volume.index > cutoff] *= 2
        pd.testing.assert_series_equal(
            relative_volume(prices["Volume"]).loc[:cutoff],
            relative_volume(changed_volume).loc[:cutoff],
        )

    def test_spy_features(self):
        spy_prices, table = self.spy_prices, self.table
        first_date = table.index[0]
        spy_daily_return = cumulative_log_return(spy_prices["Close"], 1)
        cutoff = pd.Timestamp("2010-01-04")
        # Independently check SPY values on the first QQQ feature date.
        spy_position = spy_prices.index.get_loc(first_date)
        spy_window = spy_prices["Close"].iloc[spy_position - 20:spy_position + 1].tolist()
        spy_manual_returns = [math.log(b / a) for a, b in zip(spy_window, spy_window[1:])]
        assert math.isclose(
            table.loc[first_date, "spy_return_5d"], math.fsum(spy_manual_returns[-5:]), rel_tol=1e-10
        )
        assert math.isclose(
            table.loc[first_date, "spy_vol_20d"],
            math.sqrt(252 / 20 * math.fsum(r * r for r in spy_manual_returns)), rel_tol=1e-12
        )
        changed_spy_close = spy_prices["Close"].copy()
        changed_spy_close.loc[changed_spy_close.index > cutoff] *= 1.5
        pd.testing.assert_series_equal(
            cumulative_log_return(spy_prices["Close"], 5).loc[:cutoff],
            cumulative_log_return(changed_spy_close, 5).loc[:cutoff],
        )
        pd.testing.assert_series_equal(
            historical_volatility(spy_daily_return, 20).loc[:cutoff],
            historical_volatility(cumulative_log_return(changed_spy_close, 1), 20).loc[:cutoff],
        )


class FeatureSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        snapshot = cls.root / "data" / "snapshot"
        if not all((snapshot / name).exists() for name in ("QQQ.parquet", "SPY.parquet")):
            raise unittest.SkipTest("Local market snapshot is not available.")
        cls.prices = pd.read_parquet(snapshot / "QQQ.parquet")
        cls.spy_prices = pd.read_parquet(snapshot / "SPY.parquet")
        cls.table = build_feature_table(cls.prices, cls.spy_prices)

    def test_matches_saved_table_and_preserves_inputs(self):
        saved_path = self.root / "data" / "derived" / "feature_table.parquet"
        if not saved_path.exists():
            self.skipTest("Build the local feature table first.")
        expected = pd.read_parquet(saved_path)
        pd.testing.assert_frame_equal(self.table.reset_index()[expected.columns], expected)
        before = self.prices.copy(deep=True)
        build_feature_table(self.prices, self.spy_prices)
        pd.testing.assert_frame_equal(self.prices, before)
        self.assertEqual(int(self.table.target_vol_5d.isna().sum()), 5)

