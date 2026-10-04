"""Train a single-thread GPU XGBoost expert on temporal Alpha158 summaries."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np
import xgboost as xgb

from coster.runtime import (
    DATASET_ROOT, ROOT, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALID_END, VALID_START,
    UniverseStore, correlation, metrics_by_period, write_daily_csv,
)


def transform(sequence: np.ndarray) -> np.ndarray:
    alpha = sequence[:, :, :158]
    market = sequence[:, -1, 158:]
    return np.concatenate((alpha[:, -1], alpha.mean(axis=1), market), axis=1).astype(np.float32)


def normalized_target(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.flatnonzero(np.isfinite(labels))
    values = labels[positions].astype(np.float64)
    count = int(0.025 * len(values))
    if count:
        order = np.argsort(values)
        keep = order[count:-count]
        positions = positions[keep]
        values = values[keep]
    values = (values - values.mean()) / values.std()
    return positions, values.astype(np.float32)


def materialize(store: UniverseStore, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    feature_blocks = []
    target_blocks = []
    dates = store.dates_between(start, end)
    for number, day in enumerate(dates, 1):
        sequence, labels, _ = store.batch(int(day), training=True)
        keep, target = normalized_target(labels)
        feature_blocks.append(transform(sequence[keep]))
        target_blocks.append(target)
        if number % 500 == 0 or number == len(dates):
            print(f"materialize {start}-{end}: {number}/{len(dates)}", flush=True)
    return np.concatenate(feature_blocks), np.concatenate(target_blocks)


def evaluate_model(model: xgb.Booster, store: UniverseStore) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    predictions = []
    labels_all = []
    dates_all = []
    instruments_all = []
    dates = store.dates_between(TEST_START, TEST_END)
    for number, day in enumerate(dates, 1):
        sequence, labels, instruments = store.batch(int(day), training=False)
        pred = model.predict(xgb.DMatrix(transform(sequence), nthread=1))
        rows.append({
            "date": int(day), "n": int(len(labels)),
            "finite_labels": int(np.isfinite(labels).sum()),
            "IC": correlation(pred, labels),
            "RankIC": correlation(pred, labels, rank=True),
        })
        predictions.append(pred.astype(np.float32))
        labels_all.append(labels.astype(np.float32))
        dates_all.append(np.full(len(labels), int(day), dtype=np.int32))
        instruments_all.append(instruments.astype("S8"))
        if number % 200 == 0 or number == len(dates):
            print(f"evaluate: {number}/{len(dates)}", flush=True)
    return rows, {
        "dates": np.concatenate(dates_all),
        "instruments": np.concatenate(instruments_all),
        "predictions": np.concatenate(predictions),
        "labels": np.concatenate(labels_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=("csi300", "csi800"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "traditional_baselines")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    started = time.time()
    train_x, train_y = materialize(store, TRAIN_START, TRAIN_END)
    valid_x, valid_y = materialize(store, VALID_START, VALID_END)
    print(f"train={train_x.shape}; valid={valid_x.shape}; memory_GB={(train_x.nbytes + valid_x.nbytes)/1e9:.3f}", flush=True)
    dtrain = xgb.DMatrix(train_x, label=train_y, nthread=1)
    dvalid = xgb.DMatrix(valid_x, label=valid_y, nthread=1)
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "tree_method": "hist", "device": "cuda", "nthread": 1,
        "max_depth": 6, "eta": 0.03, "subsample": 0.8,
        "colsample_bytree": 0.7, "min_child_weight": 20,
        "lambda": 2.0, "alpha": 0.1, "max_bin": 256,
        "seed": args.seed,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=args.rounds,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stopping, verbose_eval=50,
    )
    del dtrain, dvalid, train_x, train_y, valid_x, valid_y
    prefix = f"xgboost_{args.universe}_seed{args.seed}"
    model.save_model(args.output_dir / f"{prefix}.ubj")
    rows, predictions = evaluate_model(model, store)
    result = {
        "source": "non-MASTER XGBoost temporal-summary expert",
        "universe": args.universe, "seed": args.seed,
        "best_iteration": int(model.best_iteration),
        "best_score": float(model.best_score),
        "metrics": metrics_by_period(rows),
        "features": "last Alpha158 + 8-day mean Alpha158 + last market63",
        "parameters": params, "elapsed_seconds": time.time() - started,
        "mode": "single-process; nthread=1; CUDA histogram building",
    }
    (args.output_dir / f"{prefix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_daily_csv(args.output_dir / f"{prefix}_daily.csv", rows)
    np.savez_compressed(args.output_dir / f"{prefix}_predictions.npz", **predictions)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
