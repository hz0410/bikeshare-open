"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train                          fit, score, write metrics.json
     python -m src.train --target-delta           validation-only experiment
     python -m src.train --target-delta --include-test   final comparison

The default run scores every model twice on identical rows: without the
newest feature (`before_feature`) and with it (the top-level model keys).
The experiment modes print only; they never write metrics.json.
"""
import argparse
import json

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


LAG_COL = "cnt_lag_1"
NEW_FEATURE = "cnt_roll_max_7"  # scored before (excluded) and after (included)


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = LAG_COL):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline(LAG_COL),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def _score(y, pred) -> dict:
    return {
        "r2": round(float(r2_score(y, pred)), 4),
        "mae": round(float(mean_absolute_error(y, pred)), 1),
    }


def _fit_and_score_model(model, cols, train, parts, target_delta=False) -> dict:
    """Fit a fresh clone and score it on the original ride-count scale.

    With target_delta the model learns cnt - cnt_lag_1 and cnt_lag_1 is added
    back to each prediction before any metric is computed.
    """
    model = clone(model)
    y = train[TARGET] - train[LAG_COL] if target_delta else train[TARGET]
    model.fit(train[cols], y)
    scores = {}
    for split, part in parts.items():
        pred = model.predict(part[cols])
        if target_delta:
            pred = pred + part[LAG_COL].to_numpy()
        scores[split] = _score(part[TARGET], pred)
    return scores


def fit_and_score() -> dict:
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    before_cols = [c for c in cols if c != NEW_FEATURE]
    train, valid, test = temporal_split(df)
    parts = {"valid": valid, "test": test}

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
    }
    for name, model in MODELS.items():
        results[name] = _fit_and_score_model(model, cols, train, parts)
    results["before_feature"] = {
        name: _fit_and_score_model(model, before_cols, train, parts)
        for name, model in MODELS.items()
    }
    return results


def target_delta_experiment(include_test: bool) -> dict:
    """Plain HGB vs HGB on cnt - cnt_lag_1 vs ridge vs persistence.

    Same rows, same chronological split, fresh clones. Test is scored only
    when include_test is set; nothing is written to disk.
    """
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)
    parts = {"valid": valid}
    if include_test:
        parts["test"] = test
    runs = {
        "hgb_plain": (MODELS["hgb"], False),
        "hgb_delta": (MODELS["hgb"], True),
        "ridge": (MODELS["ridge"], False),
        "baseline_persistence": (MODELS["baseline_persistence"], False),
    }
    return {
        name: _fit_and_score_model(model, cols, train, parts, delta)
        for name, (model, delta) in runs.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-delta", action="store_true",
                        help="experiment: HGB on cnt - cnt_lag_1 (prints validation only)")
    parser.add_argument("--include-test", action="store_true",
                        help="with --target-delta, also score the test split")
    args = parser.parse_args()
    if args.include_test and not args.target_delta:
        parser.error("--include-test requires --target-delta")

    if args.target_delta:
        exp = target_delta_experiment(args.include_test)
        splits = list(next(iter(exp.values())))
        for split in splits:
            print(f"{split}:")
            for name, sc in exp.items():
                print(f"  {name:22s} R2={sc[split]['r2']:8.4f}  MAE={sc[split]['mae']:8.1f}")
    else:
        res = fit_and_score()
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(res, indent=2))
        print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
