# Research data snapshot

## Scope and reproduction

- Source: Yahoo Finance via yfinance 1.7.0; downloaded on 2026-09-16 UTC.
- Files: `data/snapshot/QQQ.parquet`, `SPY.parquet`, and `metadata.json`.
- Each ETF has 5,411 rows from 2004-07-01 through 2025-12-31.
- Request: `[2004-07-01, 2026-01-01)`, daily bars. Exact parameters and per-ticker download timestamps are in the metadata.
- Main study: 2005-01-01 through 2025-12-31. The 2004 portion supplies rolling-feature warm-up only and is excluded from evaluation.
- Run `python -m nasdaq_volatility_lab.check_data` to inspect the existing snapshot without network access. It writes `artifacts/data_quality.json` and raises an error if checks need review; it does not print to the console.
- `download_data.py` creates a snapshot only when its target directory does not exist. Do not routinely refresh historical data. A new download may differ from these files.

## Price and volume conventions

`auto_adjust=True` uses Yahoo's adjusted close as `Close` and scales `Open`, `High`, and `Low` by `Adj Close / Close`. Yahoo describes adjusted close as accounting for applicable splits and dividend distributions. These are adjusted USD-per-share price series, not historical executable quotes. Compute returns from this consistent adjusted `Close`; do not add the dividend column again.

`Volume` is the daily share/ETF-unit volume returned by the provider, not dollar turnover. In the installed yfinance implementation, `parse_quotes` reads the provider's volume field, and `auto_adjust` does not scale it. The history processing fills missing volume with zero and casts it to integer even with `keepna=True`. This snapshot has no zero or negative volume, and neither ETF reports a stock split within its saved date range.

We retain provider volume without applying price-adjustment factors. Yahoo's general upstream historical volume split-adjustment policy was not established from the documentation reviewed; do not call this independently verified, unadjusted tape volume. Revisit this limitation before extending to instruments with splits. Early-close sessions are retained; shorter sessions can affect daily volume comparisons.

`actions=True` retains dividends, splits, and capital gains fields for inspection. `repair=False` disables the optional yfinance repair mode; it does not disable ordinary parsing, adjustment, or missing-volume handling. These files are library-processed daily bars, not raw API responses.

## Completed checks

Both ETFs passed checks for nonempty data, ordered and unique dates, missing cells, finite OHLC/volume values, positive prices and volume, and matching ticker date sets.

An independent calendar comparison used pandas_market_calendars 5.4.0. Its `NASDAQ` alias uses the shared `NYSEExchangeCalendar` daily-session rules as a US equity reference. The requested interval, not the observed endpoints, produced 5,411 expected sessions. Neither ETF has missing or unexpected dates against that reference. Early closes remain sessions. This is a versioned calendar-library check, not a certification by an exchange.

OHLC bounds pass with relative floating-point tolerance `rtol=1e-12`, `atol=0`. Strict comparison flags SPY on 2007-04-20 and 2012-03-26: `Close - High` is approximately `1.4210854715202004e-14` in each row. This is at floating-point rounding scale. The checker retains strict flags and reports tolerant comparisons separately. No prices were rounded, clipped, filled, or removed by this project.

The metadata records the reviewed status and SHA-256 hashes of the two Parquet files. This status describes the reviewed snapshot, not an automatic guarantee for future changes. The inspection report retains strict floating-point flags as notes and checks material bounds separately. Passing the implemented checks is not independent certification of every observation.

## Limits and publication

These checks establish internal consistency and date coverage, not independent verification of every price. Historical provider data can be revised. A snapshot downloaded in 2026 is not an institutional point-in-time dataset, even though the research period ends in 2025. Final-test performance has not been evaluated.

The `data/` directory is ignored by Git. Publish the scripts and documentation; review the provider's redistribution terms before sharing market-data files.

## References

- [Yahoo adjusted-close definition](https://help.yahoo.com/kb/SLN28256.html)
- [yfinance download parameters](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [Market calendar documentation](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html)
- Installed yfinance 1.7.0 implementation: `yfinance/utils.py` (`parse_quotes`, `auto_adjust`) and `yfinance/scrapers/history.py` (`Volume` handling).
