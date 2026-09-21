# Data acquisition and conventions

`nasdaq-volatility-lab fetch-data` downloads daily QQQ and SPY bars through
yfinance, validates them, and publishes `data/snapshot/` and `data/derived/`.
The default feature range is 2005-01-01 through 2025-12-31 inclusive; six months
of earlier data supply rolling-feature warm-up. Only the download command
requires network access. Requested dates, provider/library versions and data
hashes are recorded beside the snapshot and feature table.

## Adjustment and volume

`auto_adjust=True` uses adjusted closing prices and consistently adjusted OHLC.
The adjustment accounts for provider split/dividend adjustments; prices are not
historical executable quotes. Compute returns from adjusted `Close` without
adding the dividend column again. `actions=True` retains corporate-action fields.
`repair=False` disables optional repair, not ordinary library parsing.

Provider volume is retained as daily shares/ETF units, not dollar turnover. It
is not multiplied by the price-adjustment factor. yfinance's parsing can turn
missing provider volume into zero, so zero-volume rows fail the quality gate
and require investigation. No independent claim is made about Yahoo's upstream
historical volume split-adjustment policy. Early-close sessions remain in the
data and can affect relative-volume comparisons.

## Quality and publication

Checks require nonempty, sorted, unique dates; finite positive OHLC; finite,
nonnegative volume with no zero-volume rows; no missing cells; consistent OHLC
bounds; and complete expected calendar coverage for both tickers. The calendar
comes from the versioned `NASDAQ` schedule in pandas_market_calendars.

OHLC bounds allow relative floating-point tolerance `1e-12`. Strict Close/High
rounding flags remain in the report. Prices are not rounded, filled or clipped.
Quality checks establish internal consistency, not independent verification of
every market observation.

A refresh downloads both tickers into a temporary directory and checks the
feature-table Parquet round trip before publication. Failed downloads or checks
leave previous data intact. Successful publication moves the previous data
folder to `artifacts/data_backups/<id>/`. A full range is refreshed so differently
adjusted price vintages are never appended together.

Historical provider values may change. Hashes identify the exact local bytes
used for a run, but a later download may not reproduce them. These retrospective
snapshots are not institutional point-in-time data. Data and backups remain
Git-ignored.
