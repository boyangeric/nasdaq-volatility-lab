"""Download a fixed snapshot without replacing an existing snapshot."""

from datetime import datetime, timezone
import json

import yfinance as yf

from .paths import ROOT

START = "2004-07-01"  # Include roughly six months before the 2005 study period.
END = "2026-01-01"  # Exclusive: include all of December 2025.


def main():
    output_dir = ROOT / "data" / "snapshot"
    if output_dir.exists():
        raise FileExistsError("Snapshot directory already exists. Inspect it before downloading again.")

    params = {
        "start": START,
        "end": END,
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

    frames = {}
    download_times = {}
    for ticker in ("QQQ", "SPY"):
        prices = yf.download(ticker, **params)
        if prices is None or prices.empty:
            raise RuntimeError(f"No data returned for {ticker}; no snapshot saved.")
        frames[ticker] = prices
        download_times[ticker] = datetime.now(timezone.utc).isoformat()

    # Save only after both downloads return data. Quality checks follow next.
    output_dir.mkdir(parents=True)
    for ticker, prices in frames.items():
        prices.to_parquet(output_dir / f"{ticker}.parquet", engine="pyarrow")

    metadata = {
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "download_completed_at_utc": download_times,
        "parameters": params,
        "quality_status": "pending",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
