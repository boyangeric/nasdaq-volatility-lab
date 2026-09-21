# Feature table

`nasdaq-volatility-lab fetch-data` generates `data/derived/feature_table.parquet`
and `feature_schema.json`. `features.build_feature_table` calculates the same
table without file I/O. Training reads the existing table and verifies its
hash, explicit feature whitelist, target horizon and missing-value pattern.

The default 2005–2025 range contains 5,283 rows in the original local snapshot,
including 5,278 complete targets. Requested ranges may produce different counts.
The table stays unsplit until `train` applies the chronological split.

## Definitions and units

Prices below are adjusted closing prices. Windows count trading sessions. Unless explicitly excluded, today is included; prediction takes place after the close. Let `r_t = ln(P_t / P_(t-1))` and `HV_w = sqrt((252/w) * sum of the latest w squared daily log returns)`. Full windows are required.

| Field | Definition | Unit / role |
|---|---|---|
| `feature_date` | Prediction date; historical information ends here | Date metadata, not X |
| `qqq_return_1d` | `ln(P_t / P_(t-1))` | Log return, decimal |
| `qqq_return_5d` | `ln(P_t / P_(t-5))` | Cumulative log return, decimal |
| `qqq_return_20d` | `ln(P_t / P_(t-20))` | Cumulative log return, decimal |
| `qqq_vol_5d` | `HV_5` for QQQ | Annualized volatility, decimal |
| `qqq_vol_20d` | `HV_20` for QQQ | Annualized volatility, decimal |
| `qqq_vol_60d` | `HV_60` for QQQ | Annualized volatility, decimal |
| `qqq_vol_ratio_5_20` | `HV_5 / HV_20` | Dimensionless ratio |
| `qqq_drawdown_60d` | `P_t / max(P_(t-59), ..., P_t) - 1` | Relative distance, decimal; current drawdown, not window maximum drawdown |
| `qqq_ma_deviation_20d` | `P_t / mean(P_(t-19), ..., P_t) - 1` | Relative distance, decimal |
| `qqq_relative_volume_20d` | `V_t / mean(V_(t-20), ..., V_(t-1))` | Multiple; denominator excludes today |
| `spy_return_5d` | SPY `ln(P_t / P_(t-5))` | Cumulative log return, decimal |
| `spy_vol_20d` | SPY `HV_20` | Annualized volatility, decimal |
| `label_end_date` | Date of the fifth future trading session | Date metadata, not X |
| `target_vol_5d` | `sqrt((252/5) * sum(r_(t+1)^2, ..., r_(t+5)^2))` for QQQ | Target y; annualized volatility, decimal |

Returns and price distances are not annualized. Volatility `0.25` displays as `25%`; relative volume `2.0` means twice its historical mean. Stored features are unscaled; the Ridge Pipeline fits StandardScaler on training rows only.

## Missing values and model inputs

- Only columns explicitly listed in `feature_schema.json` under `feature_columns` belong in X. Neither target nor date metadata belongs in X. Temporary future-return columns are not exported.
- Model inputs are finite and complete. Future input gaps must be investigated; do not silently fill or remove them.
- Ratios with zero or missing denominators remain NaN. A genuine zero current volume with a positive historical mean gives ratio zero, but should still be reviewed under the source-data checks.
- The last five rows (2025-12-24, 26, 29, 30, 31) lack complete future observations within this snapshot. Their target is NaN and their label-end date is NaT. Their features remain available.
- A row with a missing target cannot be used as a labeled training/evaluation sample. It can supply inputs for inference. Labeled rows still require chronological splitting and the five-session gap.
- Source-file, builder, and output hashes are recorded in the schema. Data-source limitations remain those documented in `DATA.md`. Derived data stays local under the ignored `data/` directory.
