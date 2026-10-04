"""Independent semantic spot checks for MASTER-EXT-clean-v1.

This script is deliberately single-process and single-threaded.  It verifies
the two formulas most vulnerable to silent implementation errors: the exact
future-label offset and the ordering of the three 21-dimensional market blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


from data_tools.build_master_ext_clean_v1 import (  # noqa: E402
    ALPHA158_NAMES,
    TARGET_INDICES,
    index_calendar,
    load_adjusted_symbol,
)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,default=ROOT/"datasets/master_ext_clean_v1")
    args=parser.parse_args()
    dataset=args.dataset_root
    database = dataset / "intermediate" / "master_ext_clean.sqlite"
    raw_dir = dataset / "alpha158_raw" / "by_instrument"
    market_path = dataset / "market63.npz"

    con = sqlite3.connect(database)
    calendar_values = index_calendar(con)
    calendar = pd.Index(calendar_values, name="trade_date")
    calendar_int = np.array([int(day.replace("-", "")) for day in calendar_values], dtype=np.int32)
    calendar_pos = {int(day): pos for pos, day in enumerate(calendar_int)}

    files = sorted(raw_dir.glob("*.npz"))
    selected_files = [files[pos] for pos in np.linspace(0, len(files) - 1, 12, dtype=int)]
    label_checks = []
    max_label_abs_error = 0.0
    compared_labels = 0
    for path in selected_files:
        code = path.stem[2:]
        frame, _ = load_adjusted_symbol(con, code, calendar)
        close = frame["close"].to_numpy(dtype=np.float64)
        expected = np.full(len(calendar), np.nan, dtype=np.float64)
        expected[:-5] = close[5:] / close[1:-4] - 1.0
        with np.load(path) as raw:
            positions = np.array([calendar_pos[int(day)] for day in raw["dates"]], dtype=int)
            observed = raw["label"].astype(np.float64)
        expected_selected = expected[positions]
        comparable = np.isfinite(observed) & np.isfinite(expected_selected)
        error = float(np.max(np.abs(observed[comparable] - expected_selected[comparable]))) if comparable.any() else 0.0
        max_label_abs_error = max(max_label_abs_error, error)
        compared_labels += int(comparable.sum())
        label_checks.append({"instrument": path.stem, "compared": int(comparable.sum()), "max_abs_error": error})

    with np.load(market_path) as market:
        names = market["names"].astype(str).tolist()
        market_dates = market["dates"].astype(np.int32)
        market_raw = market["raw"].astype(np.float64)
    market_checks = []
    max_market_abs_error = 0.0
    for block, index_code in enumerate(TARGET_INDICES):
        rows = con.execute(
            "SELECT trade_date,close FROM index_daily WHERE index_code=? ORDER BY trade_date",
            (index_code,),
        ).fetchall()
        dates = np.array([int(day.replace("-", "")) for day, _ in rows], dtype=np.int32)
        close = pd.Series([value for _, value in rows], dtype=float)
        expected = (close / close.shift(1) - 1.0).to_numpy()
        observed = market_raw[:, block * 21]
        comparable = np.isfinite(observed) & np.isfinite(expected)
        error = float(np.max(np.abs(observed[comparable] - expected[comparable])))
        max_market_abs_error = max(max_market_abs_error, error)
        market_checks.append(
            {
                "index_code": index_code,
                "first_column": names[block * 21],
                "dates_identical": bool(np.array_equal(dates, market_dates)),
                "max_abs_error": error,
            }
        )

    con.close()
    status = "PASS"
    if len(ALPHA158_NAMES) != 158 or len(names) != 63:
        status = "FAIL"
    if max_label_abs_error > 1e-6 or max_market_abs_error > 1e-6:
        status = "FAIL"
    if not all(item["dates_identical"] for item in market_checks):
        status = "FAIL"

    result = {
        "status": status,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "single-process single-threaded",
        "alpha158_feature_count": len(ALPHA158_NAMES),
        "market_feature_count": len(names),
        "label_formula": "adjusted_close[t+5] / adjusted_close[t+1] - 1",
        "label_sample": {
            "instruments": len(selected_files),
            "compared_values": compared_labels,
            "max_abs_error": max_label_abs_error,
            "details": label_checks,
        },
        "market_ret1_checks": {
            "max_abs_error": max_market_abs_error,
            "details": market_checks,
        },
    }
    output = ROOT / "audit_outputs" / "master_ext_semantic_spotcheck.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
