# nasdaq-volatility-lab

A command-line workflow for forecasting QQQ's next five trading days of
annualized volatility using QQQ and SPY daily data. It downloads and validates
data, trains a mean baseline, Ridge and XGBoost, and saves comparable results.
Predictions use information available after the close of each feature date.

## Setup

Python 3.12+ is required; dependencies were validated on Python 3.13.14.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On macOS, XGBoost requires OpenMP (`brew install libomp`). On Windows, activate
with `.venv\Scripts\Activate.ps1`. Direct dependencies are pinned in
`requirements.txt`; this is not a complete cross-platform dependency lock.

## Commands

The interface uses Python’s standard-library `argparse`: three subcommands and
explicit flags need no additional CLI framework dependency.

Run from the repository root with the environment activated:

```sh
nasdaq-volatility-lab fetch-data
nasdaq-volatility-lab train
nasdaq-volatility-lab show-results --plot
```

`python -m nasdaq_volatility_lab` is an equivalent entry point. Only `fetch-data`
needs network access. Both entry points default to the current directory as the
data/output workspace; use `--project-dir /path/to/workspace` **before** the
subcommand to use another location. `--debug` shows full error tracebacks.

### Fetch data

```sh
nasdaq-volatility-lab fetch-data --start-date 2005-01-01 --end-date 2025-12-31
```

Dates are inclusive and specify the feature-table range. Six months of earlier
data are downloaded automatically for rolling-feature warm-up. The end date
must precede today in New York to exclude incomplete daily bars. The default
range is fixed at 2005–2025; updating beyond it requires an explicit end date.

Both ETFs must pass calendar, OHLC, missing-value and volume checks. The command
prints the row count, labeled row count, date range and feature names. The final
five rows have no complete future target and are retained but excluded from
scoring. A successful refresh publishes a complete snapshot and feature table;
previous data is moved to `artifacts/data_backups/`. Download or validation
failures retain the existing data. Different adjusted-price vintages are never
stitched together. Provider revisions may change historical values.

### Train models

```sh
nasdaq-volatility-lab train --model all
nasdaq-volatility-lab train --model ridge --alpha 10
nasdaq-volatility-lab train --model xgboost --learning-rate 0.03 --n-estimators 1500
nasdaq-volatility-lab train --validation-start 2020-01-01 --validation-end 2021-01-01
```

Training requires `data/derived/feature_table.parquet` and its hash-checked schema;
it never downloads data. Each invocation creates a new run ID. By default,
`split.train` contains data before 2017, excluding a five-session gap, and
`split.val` contains eligible 2017 rows. `--validation-end` is **exclusive**.
No random split is used. All selected models use the same training and validation
rows. The original local snapshot gives 3,016 training rows and 251 validation rows.

| Model | Fitting procedure |
| --- | --- |
| `baseline` | `DummyRegressor(strategy="mean")`: every prediction equals the training target mean, the optimal constant for training squared-error loss. |
| `ridge` | Training-only `StandardScaler` → `Ridge(alpha=1, solver="svd")`. Override with `--alpha`. |
| `xgboost` | Automatically select parameters and tree count on an inner chronological split, then refit on all outer training rows. |

XGBoost uses `reg:squarederror` to fit trees and **inner validation MAE** for
selection. Inner validation is the last calendar year represented in outer
training, with its own five-session gap. Defaults search these three
`(max_depth, reg_lambda)` pairs: `(2, 1)`, `(2, 10)`, `(3, 10)`.
`--max-depth` or `--reg-lambda` fixes that dimension and duplicate candidates
are removed. At equal MAE, prefer lower depth, then higher L2 penalty.

Every candidate uses `learning_rate=0.05`, at most 1,000 boosting rounds and
30 rounds of early-stopping patience. `--learning-rate`, `--n-estimators` and
`--early-stopping-rounds` override these values. **`--n-estimators` is a search
ceiling, not the final tree count.** The winner is refitted using
`best_iteration + 1` trees with no outer validation data passed to fitting.
The run records candidate learning curves, selected parameters and whether the
round ceiling was reached. The run needs history before the inner validation
year; otherwise fetch a longer range or select a later validation start.

Predictions are clipped at zero to respect the volatility domain. Saved
estimators return raw predictions; use `models.predict_volatility` after loading
an artifact to reproduce this rule. Load Ridge/baseline with `joblib.load` and
XGBoost with `XGBRegressor().load_model`. The baseline artifact stores its fitted
constant; Ridge stores both the scaler and coefficients; XGBoost stores trees.

Each success line includes the run ID, model, training/validation counts, total
training time and key parameters. XGBoost total time includes candidate search
and final refit; metrics, serialization and plotting are excluded. Search/refit
times are also recorded separately.

### Show results

```sh
nasdaq-volatility-lab show-results
nasdaq-volatility-lab show-results --metric mae
nasdaq-volatility-lab show-results --run-id RUN_ID --plot
```

The default displays the latest **completed invocation**, containing only the
models trained in that invocation. Use `train --model all` for a three-model
comparison. Results from different runs are not combined. `--metric` accepts
`mse`, `mae`, `rmse` or `r2`, shows just that metric, and sorts models by validation
performance (R² descending, errors ascending, undefined values last).

The table contains train/validation metrics and training time. MAE and RMSE use
**percentage points (pp)**; MSE uses **pp²**; R² is dimensionless. For example,
MAE `0.04` in stored decimal units displays as `4.0000 pp`. R² is `N/A` for
constant targets or fewer than two samples. JSON logs retain full precision in
explicitly documented decimal units. The optional plot overlays actual and
model validation predictions, with volatility **levels in %** on the y-axis.

## Files

```text
nasdaq_volatility_lab/
  cli.py             argparse command definitions and concise console output
  data.py            data acquisition, quality checks and snapshot publication
  quality.py         independent price/calendar validation
  features.py        pure feature and target calculations
  splits.py          chronological train/gap/validation boundaries
  models.py          estimators, inner selection and prediction rule
  training.py        fitting workflow and completed-run publication
  metrics.py         common regression metrics
  reporting.py       tables and optional prediction plot
  storage.py         schemas, hashes, run IDs and workspace locking
  __main__.py        python -m entry point
pyproject.toml       package and console-script configuration
requirements.txt    direct runtime dependencies
tests/              offline feature, leakage, persistence and CLI tests
docs/               data conventions, feature definitions and split protocol
data/               snapshots and derived features (Git-ignored)
models/<run-id>/     fitted .joblib/.ubj artifacts (Git-ignored)
results/<run-id>/    run.json, predictions.parquet, optional comparison.png
results/train_log.jsonl  one JSON record per model, sharing the invocation run ID
```

A run records data/artifact hashes, feature order, split dates, counts, clipping
counts, metrics, selection details, dependency versions and Git commit/dirty
status. A `run.json` manifest marks completion; incomplete runs are ignored.
A workspace lock prevents simultaneous data refresh and training writes.

## Method and validation

The target is `sqrt((252 / 5) * sum(r_(t+i)^2 for i=1..5))`, where
`r_t = ln(P_t / P_(t-1))` uses adjusted closing prices. It is an annualized
realized-volatility RMS measure with an assumed zero daily mean, not a price or
return forecast. A target of `0.25` means 25% annualized volatility.

The 12 causal inputs cover QQQ returns, historical volatility, price positions
and relative volume, plus SPY returns and volatility. See [features](docs/FEATURES.md),
[data conventions](docs/DATA.md) and [split boundaries](docs/SPLITS.md).

```sh
python -m unittest discover -s tests -v
```

Synthetic tests run offline. The optional local snapshot equivalence test skips
if data is absent. The default 2017 period has been viewed during development;
its results are validation results, not a fresh blind test. Overlapping five-day
targets make errors dependent. Retrospectively adjusted data is not a
point-in-time archive. This CLI provides reproducible research evaluation, not
production deployment, live risk controls or a trading strategy.
