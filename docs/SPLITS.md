# Chronological split protocol

Run `python -m nasdaq_volatility_lab.split_data` to verify boundaries and regenerate `data/derived/split_manifest.json`. It reads the validated feature table without changing it. Run `python -m unittest discover -s tests -v` for the full suite, including six synthetic boundary tests (no extra test dependency).

## Why a gap is necessary

A feature date before validation is not sufficient: its five-day target can use prices inside validation. For example, the 2016-12-23 target ends on 2017-01-03. Holidays determine which days are sessions; the gap prevents target overlap with the next evaluation period.

`chronological_split` counts the five trading rows immediately before the evaluation interval and removes them from training samples. It additionally requires every training `label_end_date` to be strictly earlier than the first evaluation session. Equality is rejected. The source table must contain all trading rows, sorted and unique; do not drop unlabeled rows before calculating gaps. Gap rows are not deleted from the source data and may still supply past observations for causal rolling features.

The function returns training rows, gap rows, eligible evaluation rows, and excluded evaluation rows. The input remains unchanged. Use only the schema's `feature_columns` for model inputs, and `target_vol_5d` for the target. Dates are boundary metadata. No scaler or model is fitted by this script.

## Expanding-window development

| Evaluation year | Training rows | Gap sessions | Evaluation rows | Excluded evaluation rows |
|---|---:|---:|---:|---:|
| 2017 | 3016 | 5 | 251 | 0 |
| 2018 | 3267 | 5 | 251 | 0 |
| 2019 | 3518 | 5 | 252 | 0 |
| 2020 | 3770 | 5 | 253 | 0 |
| 2021 | 4023 | 5 | 252 | 0 |
| 2022 | 4275 | 5 | 251 | 0 |
| 2023 | 4526 | 5 | 250 | 0 |
| 2024 | 4776 | 5 | 247 | 5 |

All training periods start on 2005-01-03. Evaluation membership follows feature-date year. Development evaluation labels must finish before 2025-01-01; thus 2024-12-24, 26, 27, 30, and 31 are excluded from the 2024 evaluation. Earlier years may have labels ending in the next development year. They are known when scoring retrospectively, but may enter a subsequent fold's training only when that fold's training-label boundary permits. Adjacent labels remain dependent; this split does not make samples independent.

For later parameter selection or early stopping, each outer training set gets an inner split: use its final calendar year as inner validation, with the same five-session gap. Both inner subsets are restricted to outer training rows, and inner training labels must finish before inner validation. This audit prepares the boundaries only; it does not tune anything. The final fitting procedure will later be determined using development results, not the test set.

## Reserved final test

- Eligible final training feature dates: 2005-01-03 through 2024-12-23 (5,028 rows).
- Latest training label ends on 2024-12-31, before the first test session, 2025-01-02.
- Five gap sessions: 2024-12-24, 26, 27, 30, 31.
- Test rows with complete labels: 2025-01-02 through 2025-12-23 (245 rows).
- The last five 2025 rows have incomplete labels within the fixed snapshot and remain excluded from scoring, not deleted from the feature table.

Only dates, counts, and label availability were audited. No model has been trained or scored on the final test. Final fitting and one-time evaluation remain deferred until the modeling choices are locked. No within-2025 retraining is planned.
