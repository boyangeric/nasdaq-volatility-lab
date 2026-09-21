"""Build the validated feature table from fixed local snapshots."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "qqq_return_1d",
    "qqq_return_5d",
    "qqq_return_20d",
    "qqq_vol_5d",
    "qqq_vol_20d",
    "qqq_vol_60d",
    "qqq_vol_ratio_5_20",
    "qqq_drawdown_60d",
    "qqq_ma_deviation_20d",
    "qqq_relative_volume_20d",
    "spy_return_5d",
    "spy_vol_20d",
]


def cumulative_log_return(close, periods):
    """Net log return over past trading sessions, including today's close."""
    return np.log(close / close.shift(periods))


def historical_volatility(returns, window):
    """Annualized RMS of a complete trailing window, including today."""
    return np.sqrt(252 * returns.pow(2).rolling(window, min_periods=window).mean())


def volatility_ratio(short_vol, long_vol):
    """An undefined ratio stays missing when the denominator is zero."""
    return short_vol / long_vol.where(long_vol > 0)


def rolling_drawdown(close, window):
    """Current distance from the trailing closing high, including today."""
    high = close.rolling(window, min_periods=window).max()
    return close / high.where(high > 0) - 1


def moving_average_deviation(close, window):
    """Current distance from the trailing mean close, including today."""
    average = close.rolling(window, min_periods=window).mean()
    return close / average.where(average > 0) - 1


def relative_volume(volume, window=20):
    """Today's volume divided by the preceding window's mean, excluding today."""
    prior_average = volume.shift(1).rolling(window, min_periods=window).mean()
    return volume / prior_average.where(prior_average > 0)


def validate_feature_table(table):
    """Reject incomplete inputs while retaining rows with unknown future targets."""
    if not isinstance(table.index, pd.DatetimeIndex) or table.index.hasnans:
        raise ValueError("Feature dates must be a DatetimeIndex without missing dates.")
    if len(table) <= 5:
        raise ValueError("Feature table must include at least one complete five-day label.")
    if table.empty or not table.index.is_unique or not table.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be nonempty, sorted, and unique.")
    if not np.isfinite(table[FEATURE_COLUMNS]).all().all():
        raise ValueError("Input features must be finite and complete after warm-up.")
    labeled = table["target_vol_5d"].notna()
    if not table["label_end_date"].isna().equals(~labeled):
        raise ValueError("Target availability must match label-end date availability.")
    if not labeled.iloc[:-5].all() or labeled.iloc[-5:].any():
        raise ValueError("Expected exactly the final five targets to be missing.")
    if (table.loc[labeled, "target_vol_5d"] < 0).any():
        raise ValueError("Volatility targets must be nonnegative.")
    expected_ends = pd.Series(table.index, index=table.index).shift(-5)
    if not table["label_end_date"].equals(expected_ends.rename("label_end_date")):
        raise ValueError("Label end dates must match the fifth future trading row.")
    if not np.isfinite(table.loc[labeled, "target_vol_5d"]).all():
        raise ValueError("Observed targets must be finite.")
    if not (table.loc[labeled, "label_end_date"] > table.index[labeled]).all():
        raise ValueError("Labels must end after their prediction dates.")


def build_feature_table(prices, spy_prices, start_date="2005-01-01", end_date="2025-12-31"):
    """Calculate causal inputs and future labels without reading or writing files."""
    for frame in (prices, spy_prices):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
            raise ValueError("Price dates must be a DatetimeIndex without missing dates.")
        if frame.empty or not frame.columns.is_unique or "Close" not in frame:
            raise ValueError("Expected nonempty prices with unique columns and Close.")
        if not np.isfinite(frame["Close"]).all() or (frame["Close"] <= 0).any():
            raise ValueError("Closing prices must be finite and positive, including warm-up.")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("Price dates must be sorted and unique.")
    if not prices.index.equals(spy_prices.index):
        raise ValueError("Investigate mismatched ETF dates before joining.")
    if (
        "Volume" not in prices
        or not np.isfinite(prices["Volume"]).all()
        or (prices["Volume"] < 0).any()
    ):
        raise ValueError("QQQ volume must be finite and nonnegative.")
    # Normalize date precision before Parquet serialization. Providers may
    # return seconds while Arrow stores at least milliseconds; values stay exact.
    prices = prices.loc[:end_date].copy()
    spy_prices = spy_prices.loc[:end_date].copy()
    prices.index = prices.index.as_unit("ms")
    spy_prices.index = spy_prices.index.as_unit("ms")
    spy_daily_return = cumulative_log_return(spy_prices["Close"], 1)
    daily_return = np.log(prices["Close"] / prices["Close"].shift(1))

    table = pd.DataFrame(index=prices.index.copy())
    table.index.name = "feature_date"
    table["qqq_return_1d"] = daily_return
    for periods in (5, 20):
        table[f"qqq_return_{periods}d"] = cumulative_log_return(prices["Close"], periods)
    for window in (5, 20, 60):
        table[f"qqq_vol_{window}d"] = historical_volatility(daily_return, window)
    table["qqq_vol_ratio_5_20"] = volatility_ratio(table["qqq_vol_5d"], table["qqq_vol_20d"])
    table["qqq_drawdown_60d"] = rolling_drawdown(prices["Close"], 60)
    table["qqq_ma_deviation_20d"] = moving_average_deviation(prices["Close"], 20)
    table["qqq_relative_volume_20d"] = relative_volume(prices["Volume"])
    # Assign indexed Series so pandas aligns by trading date, not row position.
    table["spy_return_5d"] = cumulative_log_return(spy_prices["Close"], 5)
    table["spy_vol_20d"] = historical_volatility(spy_daily_return, 20)
    table["label_end_date"] = pd.Series(prices.index, index=prices.index).shift(-5)

    # Future information is kept separate from the historical feature columns.
    future_returns = pd.DataFrame(
        {f"r_plus_{step}": daily_return.shift(-step) for step in range(1, 6)}
    )

    # Require a complete five-return window; never substitute a shorter horizon.
    squared_sum = future_returns.pow(2).sum(axis=1, min_count=5)
    table["target_vol_5d"] = np.sqrt((252 / 5) * squared_sum)

    # Build features with warm-up data before selecting the requested date range.
    table = table.loc[start_date:end_date]

    validate_feature_table(table)
    return table
