"""Validate materialized CSI 300 daily, interval, and Qlib outputs."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-audit", type=Path, required=True)
    parser.add_argument("--csi800-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    daily_path = args.dataset_dir / "csi300_constituents_daily_2010_2025.csv.gz"
    intervals_path = args.dataset_dir / "csi300_membership_intervals_2010_2025.csv"
    qlib_path = args.dataset_dir / "csi300_qlib_instruments_2010_2025.txt"

    rows = 0
    duplicate_adjacent_keys = 0
    invalid_codes = 0
    date_counts: Counter[str] = Counter()
    dates: list[str] = []
    seen_dates: set[str] = set()
    previous_key = None
    with gzip.open(daily_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            key = (row["trade_date"], row["stock_code"])
            if key == previous_key:
                duplicate_adjacent_keys += 1
            previous_key = key
            if not re.fullmatch(r"\d{6}", row["stock_code"]):
                invalid_codes += 1
            date_counts[row["trade_date"]] += 1
            if row["trade_date"] not in seen_dates:
                seen_dates.add(row["trade_date"])
                dates.append(row["trade_date"])
    ordered_dates = sorted(dates)

    intervals = pd.read_csv(intervals_path, dtype={"stock_code": str})
    intervals["stock_code"] = intervals["stock_code"].str.zfill(6)
    invalid_interval_ranges = intervals.loc[intervals["start_date"] > intervals["end_date"]]
    interval_overlap_count = 0
    for _, group in intervals.sort_values(["stock_code", "start_date"]).groupby("stock_code"):
        previous_end = None
        for row in group.itertuples(index=False):
            if previous_end is not None and row.start_date <= previous_end:
                interval_overlap_count += 1
            previous_end = row.end_date

    coverage_rows = 0
    missing_interval_boundary_dates = 0
    date_set = set(ordered_dates)
    for row in intervals.itertuples(index=False):
        if row.start_date not in date_set or row.end_date not in date_set:
            missing_interval_boundary_dates += 1
            continue
        start = bisect.bisect_left(ordered_dates, row.start_date)
        end = bisect.bisect_right(ordered_dates, row.end_date)
        coverage_rows += end - start

    qlib_rows = 0
    malformed_qlib_rows = 0
    with qlib_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            qlib_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 3 or not re.fullmatch(r"(?:SH|SZ)\d{6}", parts[0]):
                malformed_qlib_rows += 1

    reconstruction = json.loads(args.reconstruction_audit.read_text(encoding="utf-8"))
    csi800 = json.loads(args.csi800_validation.read_text(encoding="utf-8"))
    csi800_pass = all(
        value["is_subset"] and value["csi800_count"] in {798, 799, 800}
        for value in csi800["comparisons"].values()
    )
    checks = {
        "daily_rows_equal_expected": rows == 1_165_800,
        "daily_unique_dates_equal_expected": len(date_counts) == 3_886,
        "daily_all_dates_have_300": set(date_counts.values()) == {300},
        "daily_no_adjacent_duplicate_keys": duplicate_adjacent_keys == 0,
        "daily_codes_are_six_digits": invalid_codes == 0,
        "interval_ranges_valid": len(invalid_interval_ranges) == 0,
        "intervals_do_not_overlap": interval_overlap_count == 0,
        "interval_boundaries_are_trading_dates": missing_interval_boundary_dates == 0,
        "interval_coverage_equals_daily_rows": coverage_rows == rows,
        "qlib_rows_equal_interval_rows": qlib_rows == len(intervals),
        "qlib_rows_well_formed": malformed_qlib_rows == 0,
        "reconstruction_state_checks_pass": (
            reconstruction["state_checks"]["reverse_violation_count"] == 0
            and reconstruction["state_checks"]["forward_violation_count"] == 0
            and not reconstruction["state_checks"]["non_300_reverse_count_checks"]
        ),
        "csi800_snapshot_checks_pass": csi800_pass,
    }
    result = {
        "single_process": True,
        "single_threaded": True,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "daily_rows": rows,
            "daily_unique_dates": len(date_counts),
            "daily_count_distribution": {
                str(key): int(value) for key, value in Counter(date_counts.values()).items()
            },
            "daily_min_date": min(date_counts),
            "daily_max_date": max(date_counts),
            "interval_rows": int(len(intervals)),
            "interval_coverage_rows": coverage_rows,
            "qlib_rows": qlib_rows,
            "duplicate_adjacent_keys": duplicate_adjacent_keys,
            "invalid_daily_codes": invalid_codes,
            "interval_overlap_count": interval_overlap_count,
            "missing_interval_boundary_dates": missing_interval_boundary_dates,
            "malformed_qlib_rows": malformed_qlib_rows,
        },
        "csi800_comparisons": csi800["comparisons"],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.dataset_dir.iterdir())
            if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
