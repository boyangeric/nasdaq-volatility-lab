# Chronological split protocol

`train` calls `chronological_split` in `nasdaq_volatility_lab/splits.py`.
It returns `split.train`, `split.gap`, `split.val` (an alias for `evaluation`)
and rows excluded from evaluation because their targets are unavailable.
The source feature table remains unchanged. No random split is used.

## Outer training and validation

Validation dates lie in `[--validation-start, --validation-end)`, defaulting to
2017-01-01 through 2018-01-01 exclusive. All models use identical eligible rows.
Training contains earlier feature dates except the final five trading rows.
The function also verifies every training `label_end_date` is strictly before
the first actual validation session. Equality is rejected.

A date preceding validation is insufficient by itself: its target can still use
validation prices. In the original snapshot, the 2016-12-23 target ends on
2017-01-03, the first validation date. The gap excludes that sample from training.
Count sessions before dropping incomplete targets; holidays are not sessions.
Gap rows can still supply known historical observations for later rolling features.

The default original snapshot has 3,016 training rows (2005-01-03–2016-12-22),
five gap sessions and 251 validation rows. Training labels end no later than
2016-12-30. The last five feature-table rows have incomplete targets and cannot
be scored. Actual included/excluded dates are recorded in each run manifest.

Validation membership follows the feature date. A complete validation target
may finish after the requested validation end; it is scored retrospectively.
This never permits training labels to cross into validation. If extending this
workflow to a strictly as-of evaluation, impose a separate label cutoff using
the split function's `label_cutoff` argument.

## Inner XGBoost selection

Only outer training rows enter the inner split. Its validation dates are in the
last calendar year represented in outer training; an outer split within a year
therefore produces a partial inner year. Inner training excludes its own final
five pre-validation sessions and enforces the same label-end boundary.

For the default original snapshot, this yields 2,764 inner training rows and
247 inner validation rows in 2016. Each candidate fits trees on inner training
and checks inner validation MAE after every boosting round. Early stopping
selects the best round; the candidate with the lowest inner MAE wins.

A fresh model is then fitted to all 3,016 outer training rows using the chosen
parameters and `best_iteration + 1` trees. Outer validation is used only for
final scoring, never early stopping or candidate selection. The mean baseline
and Ridge also fit exclusively on outer training; Ridge's scaler stays inside
its fitted Pipeline.

Repeatedly choosing options after viewing outer scores makes that period part
of model development. These results are not a fresh blind test, and overlapping
five-day labels mean daily errors are not independent.
