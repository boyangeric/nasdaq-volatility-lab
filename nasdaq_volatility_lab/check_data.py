"""Write a structured quality report without modifying the snapshots."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from .paths import ROOT


def exceeds(left, right):
    """Compare bounds with a tiny relative tolerance for floating-point noise."""
    return (left > right) & ~np.isclose(left, right, rtol=1e-12, atol=0)


def inspect_prices(prices, expected_dates):
    """Return a report for validly shaped input; reject malformed table schemas."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not prices.columns.is_unique or not required.issubset(prices.columns):
        raise ValueError("Prices require unique Open, High, Low, Close, and Volume columns.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Prices require a DatetimeIndex without missing dates.")
    ohlc = prices[["Open", "High", "Low", "Close"]]
    market_values = prices[["Open", "High", "Low", "Close", "Volume"]]
    checks = {
        "Non-finite price or volume": ~np.isfinite(market_values).all(axis=1),
        "Non-positive price": (ohlc <= 0).any(axis=1),
        "Negative volume": prices["Volume"] < 0,
        "Close outside low/high (strict)": (prices["Close"] < prices["Low"])
        | (prices["Close"] > prices["High"]),
        "High below low (tolerant)": exceeds(prices["Low"], prices["High"]),
        "Open outside low/high (tolerant)": exceeds(prices["Low"], prices["Open"])
        | exceeds(prices["Open"], prices["High"]),
        "Close outside low/high (tolerant)": exceeds(prices["Low"], prices["Close"])
        | exceeds(prices["Close"], prices["High"]),
        # Zero volume needs review; it is not automatically invalid.
        "Zero volume (review)": prices["Volume"] == 0,
    }
    findings = {
        name: {
            "count": int(mask.sum()),
            "sample_dates": prices.index[mask][:5].strftime("%Y-%m-%d").tolist(),
        }
        for name, mask in checks.items()
    }
    missing_dates = expected_dates.difference(prices.index)
    unexpected_dates = prices.index.difference(expected_dates)
    missing_cells = {name: int(count) for name, count in prices.isna().sum().items()}
    blocking = any(
        item["count"] for name, item in findings.items()
        if name != "Close outside low/high (strict)"
    )
    passed = (
        not prices.empty and prices.index.is_monotonic_increasing
        and prices.index.is_unique and not any(missing_cells.values())
        and missing_dates.empty and unexpected_dates.empty and not blocking
    )
    return {
        "rows": len(prices), "sorted": prices.index.is_monotonic_increasing,
        "duplicate_dates": int(prices.index.duplicated().sum()),
        "missing_cells": missing_cells, "findings": findings,
        "missing_dates": missing_dates.strftime("%Y-%m-%d").tolist(),
        "unexpected_dates": unexpected_dates.strftime("%Y-%m-%d").tolist(),
        "passed": bool(passed),
    }


def inspect_snapshot(snapshot_dir):
    """Return counts and findings; strict floating-point flags remain visible."""
    snapshot_dir = Path(snapshot_dir)
    metadata = json.loads((snapshot_dir / "metadata.json").read_text())
    params = metadata["parameters"]
    calendar = mcal.get_calendar("NASDAQ")
    expected = calendar.schedule(
        start_date=params["start"],
        end_date=pd.Timestamp(params["end"]) - pd.Timedelta(days=1),
    ).index
    frames = {ticker: pd.read_parquet(snapshot_dir / f"{ticker}.parquet") for ticker in ("QQQ", "SPY")}
    reports = {ticker: inspect_prices(frame, expected) for ticker, frame in frames.items()}
    aligned = frames["QQQ"].index.equals(frames["SPY"].index)
    return {
        "calendar_version": mcal.__version__, "calendar_alias": "NASDAQ",
        "expected_sessions": len(expected), "dates_aligned": aligned,
        "tickers": reports,
        "passed": aligned and all(report["passed"] for report in reports.values()),
    }


def main():
    root = ROOT
    report = inspect_snapshot(root / "data" / "snapshot")
    output = root / "artifacts" / "data_quality.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise ValueError(f"Data quality checks require review: {output}")


if __name__ == "__main__":
    main()
