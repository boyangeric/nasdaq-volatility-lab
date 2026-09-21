"""Model construction and training-only XGBoost model selection."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import FEATURE_COLUMNS
from .splits import chronological_split, describe_split


@dataclass(frozen=True)
class TrainingOptions:
    alpha: float = 1.0
    learning_rate: float = 0.05
    n_estimators: int = 1000
    early_stopping_rounds: int = 30
    max_depth: int | None = None
    reg_lambda: float | None = None

    def validate(self):
        for name in ("alpha", "learning_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("n_estimators", "early_stopping_rounds"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer.")
        if self.reg_lambda is not None and (
            not np.isfinite(self.reg_lambda) or self.reg_lambda < 0
        ):
            raise ValueError("reg_lambda must be finite and nonnegative.")


def xgboost_parameters(options, depth, penalty, count):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "learning_rate": options.learning_rate,
        "n_estimators": count,
        "max_depth": depth,
        "reg_lambda": penalty,
        "reg_alpha": 0.0,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
    }


def select_xgboost(split, options):
    """Choose parameters and rounds without inspecting any outer validation row.

    The inner validation period is the last calendar year represented in the
    outer training set. Its own five-session gap prevents label overlap.
    """
    year = split.train.index.max().year
    try:
        inner = chronological_split(split.train, f"{year}-01-01", f"{year + 1}-01-01")
    except ValueError as exc:
        raise ValueError(
            "XGBoost needs history before its inner validation year. Fetch an earlier --start-date or move --validation-start later."
        ) from exc
    candidates = list(
        dict.fromkeys(
            (
                options.max_depth if options.max_depth is not None else depth,
                options.reg_lambda if options.reg_lambda is not None else penalty,
            )
            for depth, penalty in [(2, 1.0), (2, 10.0), (3, 10.0)]
        )
    )
    records = []
    for number, (depth, penalty) in enumerate(candidates, start=1):
        model = XGBRegressor(
            **xgboost_parameters(options, depth, penalty, options.n_estimators),
            early_stopping_rounds=options.early_stopping_rounds,
        )
        model.fit(
            inner.train[FEATURE_COLUMNS],
            inner.train.target_vol_5d,
            eval_set=[(inner.val[FEATURE_COLUMNS], inner.val.target_vol_5d)],
            verbose=False,
        )
        history = model.evals_result()["validation_0"]["mae"]
        records.append(
            {
                "candidate": number,
                "max_depth": depth,
                "reg_lambda": penalty,
                # best_iteration is zero-based; refit needs the number of trees.
                "best_iteration": int(model.best_iteration),
                "selected_n_estimators": int(model.best_iteration) + 1,
                "inner_mae": float(model.best_score),
                "rounds_evaluated": len(history),
                "reached_round_limit": len(history) == options.n_estimators,
                "inner_mae_history": [float(value) for value in history],
            }
        )
    best = min(
        records,
        key=lambda row: (row["inner_mae"], row["max_depth"], -row["reg_lambda"], row["candidate"]),
    )
    params = xgboost_parameters(
        options, best["max_depth"], best["reg_lambda"], best["selected_n_estimators"]
    )
    return params, {
        "inner_split": describe_split("inner", inner),
        "candidates": records,
        "selected_candidate": best["candidate"],
    }


def fit_model(name, split, options):
    """Fit each model on split.train; return timing and selection provenance."""
    options.validate()
    started = perf_counter()
    selection = None
    selection_seconds = 0.0
    if name == "baseline":
        model = DummyRegressor(strategy="mean")
        parameters = {"strategy": "mean"}
    elif name == "ridge":
        parameters = {"alpha": options.alpha, "solver": "svd", "scaler": "StandardScaler"}
        # Pipeline fits the scaler on training data and reuses it at prediction.
        model = make_pipeline(StandardScaler(), Ridge(alpha=options.alpha, solver="svd"))
    elif name == "xgboost":
        parameters, selection = select_xgboost(split, options)
        selection_seconds = perf_counter() - started
        # Refit on ALL outer training rows. Do not pass an outer eval_set.
        model = XGBRegressor(**parameters)
    else:
        raise ValueError(f"Unknown model: {name}")
    fit_started = perf_counter()
    model.fit(split.train[FEATURE_COLUMNS], split.train.target_vol_5d)
    fit_seconds = perf_counter() - fit_started
    if name == "baseline":
        parameters["constant"] = float(model.constant_.ravel()[0])
    return model, {
        "parameters": parameters,
        "selection": selection,
        "selection_seconds": selection_seconds,
        "fit_seconds": fit_seconds,
        "training_seconds": perf_counter() - started,
    }


def predict_volatility(model, features):
    """Enforce the nonnegative volatility domain consistently for every model.

    Stored estimators return raw predictions. Use this helper when loading an
    artifact for inference to reproduce the metrics reported by this project.
    """
    raw = np.asarray(model.predict(features), dtype=float)
    if raw.shape != (len(features),) or not np.isfinite(raw).all():
        raise ValueError("Model produced invalid predictions.")
    return np.maximum(raw, 0.0), int((raw < 0).sum())
