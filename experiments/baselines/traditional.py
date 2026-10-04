"""Single-process traditional ML baselines for MASTER-EXT25.

The feature protocol matches the existing XGBoost baseline:
last-day Alpha158 + eight-day mean Alpha158 + last-day Market63 (379 dims).
Targets, chronological splits, daily metrics, and saved prediction artifacts
match the unified MASTER-EXT25 evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from coster.runtime import (
    DATASET_ROOT,
    ROOT,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALID_END,
    VALID_START,
    UniverseStore,
    correlation,
    metrics_by_period,
    summarize_daily,
    write_daily_csv,
)


FEATURE_PROTOCOL = "last Alpha158 + 8-day mean Alpha158 + last Market63"


def transform(sequence: np.ndarray) -> np.ndarray:
    """Convert an eight-day 221-dimensional sequence to 379 tabular features."""
    alpha = sequence[:, :, :158]
    market = sequence[:, -1, 158:]
    return np.concatenate((alpha[:, -1], alpha.mean(axis=1), market), axis=1).astype(
        np.float32, copy=False
    )


def normalized_target(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same daily 2.5% tail removal and z-score used during training."""
    positions = np.flatnonzero(np.isfinite(labels))
    values = labels[positions].astype(np.float64)
    count = int(0.025 * len(values))
    if count:
        order = np.argsort(values)
        keep = order[count:-count]
        positions = positions[keep]
        values = values[keep]
    scale = values.std()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("daily target has zero or invalid standard deviation")
    values = (values - values.mean()) / scale
    return positions, values.astype(np.float32)


def materialize(
    store: UniverseStore, start: int, end: int, max_days: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    dates = store.dates_between(start, end)
    if max_days is not None:
        dates = dates[:max_days]
    for number, day in enumerate(dates, 1):
        sequence, labels, _ = store.batch(int(day), training=True)
        keep, target = normalized_target(labels)
        feature_blocks.append(transform(sequence[keep]))
        target_blocks.append(target)
        if number % 500 == 0 or number == len(dates):
            print(f"materialize {start}-{end}: {number}/{len(dates)}", flush=True)
    return np.concatenate(feature_blocks), np.concatenate(target_blocks)


def evaluate(
    model,
    scaler: StandardScaler | None,
    store: UniverseStore,
    start: int,
    end: int,
    max_days: int | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    predictions: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    dates_all: list[np.ndarray] = []
    instruments_all: list[np.ndarray] = []
    dates = store.dates_between(start, end)
    if max_days is not None:
        dates = dates[:max_days]
    for number, day in enumerate(dates, 1):
        sequence, labels, instruments = store.batch(int(day), training=False)
        features = transform(sequence)
        if scaler is not None:
            features = scaler.transform(features)
        pred = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
        rows.append(
            {
                "date": int(day),
                "n": int(len(labels)),
                "finite_labels": int(np.isfinite(labels).sum()),
                "IC": correlation(pred, labels),
                "RankIC": correlation(pred, labels, rank=True),
            }
        )
        predictions.append(pred)
        labels_all.append(labels.astype(np.float32))
        dates_all.append(np.full(len(labels), int(day), dtype=np.int32))
        instruments_all.append(instruments.astype("S8"))
        if number % 200 == 0 or number == len(dates):
            print(f"evaluate {start}-{end}: {number}/{len(dates)}", flush=True)
    return rows, {
        "dates": np.concatenate(dates_all),
        "instruments": np.concatenate(instruments_all),
        "predictions": np.concatenate(predictions),
        "labels": np.concatenate(labels_all),
    }


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_store: UniverseStore,
    alphas: list[float],
    max_eval_days: int | None,
) -> tuple[Ridge, StandardScaler, dict]:
    scaler = StandardScaler(copy=False)
    train_x = scaler.fit_transform(train_x)
    candidates = []
    best_model = None
    best_score = -float("inf")
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=1000)
        model.fit(train_x, train_y)
        rows, _ = evaluate(
            model, scaler, valid_store, VALID_START, VALID_END, max_eval_days
        )
        metrics = summarize_daily(rows)
        score = 0.5 * (metrics["IC"] + metrics["RankIC"])
        candidates.append({"alpha": alpha, "metrics": metrics, "selection_score": score})
        print(
            f"ridge alpha={alpha:g} valid_IC={metrics['IC']:.6f} "
            f"valid_RankIC={metrics['RankIC']:.6f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("ridge validation did not produce a model")
    return best_model, scaler, {
        "selection_metric": "0.5 * (validation IC + validation RankIC)",
        "candidates": candidates,
        "selected_alpha": float(best_model.alpha),
    }


def fit_random_forest(
    train_x: np.ndarray, train_y: np.ndarray, seed: int, trees: int
) -> tuple[RandomForestRegressor, None, dict]:
    config = {
        "n_estimators": trees,
        "max_depth": 12,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
        "max_samples": 0.5,
        "bootstrap": True,
        "random_state": seed,
        "n_jobs": 1,
    }
    model = RandomForestRegressor(**config)
    model.fit(train_x, train_y)
    return model, None, {"fixed_pre_registered_configuration": config}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ridge", "random_forest"), required=True)
    parser.add_argument("--universe", choices=("csi300", "csi800"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--max-train-days", type=int)
    parser.add_argument("--max-eval-days", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "traditional_baselines",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    store = UniverseStore(DATASET_ROOT, args.universe)
    started = time.time()

    with threadpool_limits(limits=1):
        train_x, train_y = materialize(
            store, TRAIN_START, TRAIN_END, args.max_train_days
        )
        print(
            f"train={train_x.shape}; target={train_y.shape}; "
            f"memory_GB={(train_x.nbytes + train_y.nbytes) / 1e9:.3f}",
            flush=True,
        )
        if args.model == "ridge":
            model, scaler, selection = fit_ridge(
                train_x, train_y, store, args.ridge_alphas, args.max_eval_days
            )
        else:
            model, scaler, selection = fit_random_forest(
                train_x, train_y, args.seed, args.rf_trees
            )
        del train_x, train_y
        rows, prediction_arrays = evaluate(
            model, scaler, store, TEST_START, TEST_END, args.max_eval_days
        )

    prefix = f"{args.model}_{args.universe}_seed{args.seed}"
    joblib.dump({"model": model, "scaler": scaler}, args.output_dir / f"{prefix}.joblib")
    result = {
        "source": "traditional ML baseline on MASTER-EXT25",
        "model": args.model,
        "universe": args.universe,
        "seed": args.seed,
        "features": FEATURE_PROTOCOL,
        "feature_dimension": 379,
        "target": "daily 2.5% tail removal followed by cross-sectional z-score",
        "splits": {
            "train": [TRAIN_START, TRAIN_END],
            "valid": [VALID_START, VALID_END],
            "test": [TEST_START, TEST_END],
        },
        "selection": selection,
        "metrics": metrics_by_period(rows),
        "elapsed_seconds": time.time() - started,
        "mode": "single-process; CPU thread pools limited to 1; estimator n_jobs=1",
        "arguments": vars(args) | {"output_dir": str(args.output_dir.resolve())},
    }
    (args.output_dir / f"{prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_daily_csv(args.output_dir / f"{prefix}_daily.csv", rows)
    np.savez_compressed(
        args.output_dir / f"{prefix}_predictions.npz", **prediction_arrays
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
