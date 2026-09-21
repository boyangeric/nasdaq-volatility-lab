"""Download, validate and publish coherent QQQ/SPY snapshots and features."""

import contextlib
import io
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from .features import FEATURE_COLUMNS, build_feature_table
from .quality import inspect_prices
from .storage import new_run_id, sha256, workspace_lock, write_json


def parse_date(value):
    try:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid date {value!r}; use YYYY-MM-DD.") from exc


def download_prices(ticker, parameters):
    """Translate noisy provider failures into actionable CLI errors."""
    logger = logging.getLogger("yfinance")
    previous = logger.disabled
    logger.disabled = True
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            frame = yf.download(ticker, **parameters)
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for {ticker}. Check network access and retry fetch-data."
        ) from exc
    finally:
        logger.disabled = previous
    if frame is None or frame.empty:
        raise ValueError(
            f"No data returned for {ticker}. Check dates/network and retry fetch-data."
        )
    return frame


def fetch_data(root, start_date, end_date):
    """Refresh an inclusive feature range, with automatic pre-range warm-up.

    Download one adjusted-price vintage; do not append newly adjusted prices
    to a differently adjusted history. Publish only after all checks pass.
    """
    start, end = parse_date(start_date), parse_date(end_date)
    if start >= end:
        raise ValueError("--start-date must precede --end-date.")
    if end >= pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize():
        raise ValueError("--end-date must be before today to exclude incomplete daily bars.")
    raw_start = start - pd.DateOffset(months=6)
    calendar = mcal.get_calendar("NASDAQ")
    expected = calendar.schedule(start_date=raw_start, end_date=end).index
    if (expected < start).sum() < 60 or (expected >= start).sum() <= 5:
        raise ValueError(
            "Range needs warm-up and at least six feature sessions. Expand --start-date/--end-date."
        )
    parameters = {
        "start": str(raw_start.date()),
        "end": str((end + pd.Timedelta(days=1)).date()),
        "interval": "1d",
        "auto_adjust": True,
        "back_adjust": False,
        "repair": False,
        "keepna": True,
        "actions": True,
        "multi_level_index": False,
        "threads": False,
        "progress": False,
    }
    with workspace_lock(root):
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=root) as temporary:
            stage = Path(temporary) / "data"
            snapshot, derived = stage / "snapshot", stage / "derived"
            snapshot.mkdir(parents=True)
            derived.mkdir()
            frames, reports = {}, {}
            for ticker in ("QQQ", "SPY"):
                frame = download_prices(ticker, parameters)
                reports[ticker] = inspect_prices(frame, expected)
                if not reports[ticker]["passed"]:
                    raise ValueError(
                        f"{ticker} failed date/price/volume quality checks; previous data retained. Check provider data and retry fetch-data."
                    )
                frame.to_parquet(snapshot / f"{ticker}.parquet")
                frames[ticker] = frame
            table = build_feature_table(frames["QQQ"], frames["SPY"], start_date, end_date)
            saved = table.reset_index()
            path = derived / "feature_table.parquet"
            saved.to_parquet(path, index=False)
            try:
                pd.testing.assert_frame_equal(saved, pd.read_parquet(path), check_freq=False)
            except AssertionError as exc:
                raise ValueError(
                    "Feature-table serialization check failed; previous data retained. Retry with --debug before fetch-data for details."
                ) from exc
            timestamp = datetime.now(timezone.utc).isoformat()
            write_json(
                snapshot / "metadata.json",
                {
                    "source": "Yahoo Finance via yfinance",
                    "yfinance_version": yf.__version__,
                    "download_completed_at_utc": timestamp,
                    "parameters": parameters,
                    "requested_feature_range": [start_date, end_date],
                    "quality_status": "passed",
                },
            )
            write_json(
                snapshot / "quality.json",
                {"calendar_version": mcal.__version__, "tickers": reports},
            )
            schema = {
                "schema_version": 2,
                "feature_columns": FEATURE_COLUMNS,
                "target_column": "target_vol_5d",
                "horizon_trading_days": 5,
                "annualization_factor": 252,
                "generated_at_utc": timestamp,
                "row_count": len(table),
                "labeled_row_count": int(table.target_vol_5d.notna().sum()),
                "date_range": [str(table.index.min().date()), str(table.index.max().date())],
                "requested_range": [start_date, end_date],
                "table_sha256": sha256(path),
                "source_sha256": {
                    name: sha256(snapshot / name)
                    for name in ("QQQ.parquet", "SPY.parquet", "metadata.json")
                },
                "builder_sha256": sha256(Path(__file__).with_name("features.py")),
            }
            write_json(derived / "feature_schema.json", schema)
            current = root / "data"
            backup = root / "artifacts/data_backups" / new_run_id()
            if current.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                current.rename(backup)
            try:
                stage.rename(current)
            except BaseException:
                if backup.exists():
                    backup.rename(current)
                raise
            return schema
