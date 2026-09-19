# nasdaq_volatility_lab

A Python research project predicting QQQ's next five trading days of annualized
volatility from daily QQQ and SPY data. Predictions are made after the close;
the target measures volatility, not price direction or investment returns.

## Current status

Completed through Day 2: fixed data snapshots, quality checks, 12 causal features,
future labels, chronological splits, a historical-volatility baseline, and a
StandardScaler → Ridge Pipeline. Only the 2017 development fold has been scored.
XGBoost, full walk-forward model evaluation, and the interface remain planned.
The 2025 final test has not been scored.

## Repository layout

```text
nasdaq_volatility_lab/     Importable Python package and executable workflow modules
  build_table.py    Pure feature functions and feature-table creation
  split_data.py     Chronological splitting and boundary audit
  storage.py        Shared artifact integrity and schema validation
  train_ridge.py    StandardScaler → Ridge and 2017 evaluation
  baseline.py       Historical-volatility evaluation
  metrics.py        Shared MAE/RMSE calculation
  check_data.py     Snapshot quality checks
  download_data.py  Fixed snapshot acquisition
  paths.py          Checkout-relative data/output location
tests/              Synthetic unit tests and optional snapshot regression test
docs/               Data provenance, feature definitions, split protocol
data/               Local snapshots and derived tables (ignored)
artifacts/          Local reports, predictions, and models (ignored)
```

Run modules from the repository root using `python -m nasdaq_volatility_lab.<module>`.
Imports have no download, training, or file-writing side effects.

## Setup and reproduction

Validated locally with Python 3.13.14. From the repository directory:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
Tests using local market data skip when the snapshot is absent; synthetic
label, split, quality, and Ridge tests run without downloading data.

To reproduce the current workflow, run these commands in order, stopping if any
command fails. Only the download step needs network access:

```sh
python -m nasdaq_volatility_lab.download_data
python -m nasdaq_volatility_lab.check_data
python -m nasdaq_volatility_lab.build_table
python -m nasdaq_volatility_lab.split_data
python -m nasdaq_volatility_lab.baseline
python -m nasdaq_volatility_lab.train_ridge
python -m unittest discover -s tests -v
```

Skip `download_data.py` if you already have the fixed snapshot. It deliberately
refuses to overwrite `data/snapshot/`. Provider revisions mean a new download
may not reproduce the original snapshot or metrics exactly. Direct dependencies
are pinned; this is not a complete cross-platform dependency lock.

Modules are silent on success and raise errors on failure. Runtime validation
also runs under `python -O`. Generated outputs remain
local and are ignored by Git:

| Path | Contents |
| --- | --- |
| `data/snapshot/` | QQQ/SPY Parquet files and download metadata |
| `artifacts/data_quality.json` | Quality checks and review notes |
| `data/derived/` | Feature table, schema, hashes, and split manifest |
| `artifacts/baseline_2017/` | Baseline predictions and metrics |
| `artifacts/ridge_2017/` | Ridge predictions, metrics, and fitted Pipeline |

Rebuilding derived data or running evaluation replaces the corresponding output
files. A quality report failure requires investigation before continuing.

## Method

Daily log returns are `r_t = ln(P_t / P_(t-1))`, using adjusted closing prices.
The target is `sqrt((252 / 5) * sum(r_(t+i)^2 for i=1..5))`, a zero-mean RMS
proxy for annualized realized volatility. A target of `0.25` means 25% annualized
volatility. The final five rows retain their inputs but have no complete target.

The 12 inputs describe QQQ returns, historical volatility, volatility ratio,
drawdown, moving-average deviation, relative volume, and SPY returns/volatility.
See [FEATURES.md](docs/FEATURES.md) for exact definitions.

Training starts in 2005, with 2004 data used only for feature warm-up.
Development folds cover 2017–2024; 2025 is reserved for final testing.
Every split excludes five trading rows before evaluation and checks that training
labels end before evaluation starts. Development labels cannot extend into 2025.
See [SPLITS.md](docs/SPLITS.md) for the full boundary protocol.

The baseline predicts today's trailing 20-day volatility. The Ridge model uses
`StandardScaler` fitted only on training rows, followed by
`Ridge(alpha=1.0, solver="svd")`. Negative predictions are clipped to zero and
counted by the evaluation code. The saved Pipeline itself returns raw predictions;
apply the same clipping rule when using it. Its implementation is in
[train_ridge.py](nasdaq_volatility_lab/train_ridge.py).

## Initial development results

Original local snapshot: 3,016 training rows and 251 validation rows in 2017.
Errors below are **percentage points of annualized volatility**.

| Model | MAE | RMSE |
| --- | ---: | ---: |
| 20-day historical volatility | 4.0507 | 5.1290 |
| Ridge, alpha = 1 | 3.9814 | 4.8422 |

Ridge produced no negative predictions on this fold. This small single-year
improvement does not establish generalization or statistical significance.
Five-day labels overlap, so daily errors are not independent. No tuning or
final-test scoring has been performed.

## Data and project notes

Yahoo Finance data is downloaded through yfinance. Adjusted prices account for
provider adjustments and are not historical executable quotes. The snapshot was
downloaded after the study period and is not a point-in-time data archive.
See [DATA.md](docs/DATA.md) for provenance, volume limitations, and quality findings.
Market data, fitted models, caches, and credentials are excluded from commits.

Learning notes are maintained locally and excluded from version control.
This repository is an educational research project, not a trading system.
