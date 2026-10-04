"""Reconstruct CSI 300 membership from an official anchor and CSMAR changes.

The implementation is deliberately single-process and single-threaded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import openpyxl
import pandas as pd


INDEX_CODE = "000300"
CSI800_CODE = "000906"
ADD_ACTION = "1"
DELETE_ACTION = "2"


def norm_code(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:06d}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):06d}"
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text))).zfill(6)
    return text


def norm_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return text


def market_from_exchange(value: Any, code: str) -> str:
    text = str(value or "").upper()
    if "SHANGHAI" in text or "SSE" in text or "\u4e0a\u6d77" in text:
        return "SSE"
    if "SHENZHEN" in text or "SZSE" in text or "\u6df1\u5733" in text:
        return "SZSE"
    return "SSE" if code.startswith(("5", "6", "9")) else "SZSE"


def qlib_code(code: str, market: str) -> str:
    return ("SH" if market == "SSE" else "SZ") + code


def iter_csmar_rows(path: Path) -> tuple[list[str], Iterable[tuple[Any, ...]], openpyxl.Workbook]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    sheet.reset_dimensions()
    rows = sheet.iter_rows(values_only=True)
    headers = [str(value).strip() if value is not None else "" for value in next(rows)]
    next(rows, None)
    next(rows, None)
    return headers, rows, workbook


def load_anchor(path: Path) -> tuple[pd.DataFrame, str, dict[str, str], dict[str, str]]:
    anchor = pd.read_csv(path, dtype={"stock_code": str, "index_code": str})
    required = {"snapshot_date", "index_code", "stock_code", "stock_name", "exchange"}
    missing = sorted(required.difference(anchor.columns))
    if missing:
        raise ValueError(f"Anchor is missing columns: {missing}")
    anchor["stock_code"] = anchor["stock_code"].map(norm_code)
    dates = sorted(anchor["snapshot_date"].dropna().astype(str).unique())
    if len(dates) != 1:
        raise ValueError(f"Anchor must have exactly one snapshot date, got {dates}")
    if len(anchor) != 300 or anchor["stock_code"].nunique() != 300:
        raise ValueError("Anchor must contain exactly 300 unique constituent codes")
    if set(anchor["index_code"].map(norm_code)) != {INDEX_CODE}:
        raise ValueError("Anchor index code is not exclusively 000300")
    markets = {
        row.stock_code: market_from_exchange(row.exchange, row.stock_code)
        for row in anchor.itertuples(index=False)
    }
    names = {row.stock_code: str(row.stock_name) for row in anchor.itertuples(index=False)}
    return anchor, dates[0], markets, names


def scan_changes(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(root.rglob("IDX_Chgsmp*.xlsx"), key=lambda value: str(value).lower())
    records: list[dict[str, Any]] = []
    file_stats: list[dict[str, Any]] = []
    for number, path in enumerate(files, 1):
        print(f"[changes {number}/{len(files)}] {path.parent.name}/{path.name}", flush=True)
        headers, rows, workbook = iter_csmar_rows(path)
        rows_read = 0
        target_rows = 0
        try:
            for raw in rows:
                rows_read += 1
                row = raw[: len(headers)]
                code = norm_code(row[0] if row else None)
                if code and code > INDEX_CODE:
                    break
                if code != INDEX_CODE:
                    continue
                target_rows += 1
                records.append(
                    {
                        "index_code": code,
                        "change_date": norm_date(row[1]),
                        "stock_code": norm_code(row[2]),
                        "stock_name": str(row[3]).strip() if row[3] is not None else "",
                        "action": str(row[4]).strip() if row[4] is not None else "",
                        "security_type": str(row[5]).strip() if row[5] is not None else "",
                        "announcement_date": norm_date(row[6]),
                        "market": str(row[7]).strip() if row[7] is not None else "",
                        "source_file": str(path.resolve()),
                    }
                )
        finally:
            workbook.close()
        file_stats.append(
            {
                "path": str(path.resolve()),
                "rows_read_until_early_stop": rows_read,
                "target_rows": target_rows,
            }
        )
    raw_frame = pd.DataFrame(records)
    if raw_frame.empty:
        raise ValueError("No 000300 change events were found")
    key = ["change_date", "stock_code", "action", "security_type", "market"]
    exact_duplicate_rows = int(raw_frame.duplicated(key, keep=False).sum())
    frame = raw_frame.drop_duplicates(key, keep="first").copy()
    unknown_actions = sorted(set(frame["action"]) - {ADD_ACTION, DELETE_ACTION})
    if unknown_actions:
        raise ValueError(f"Unexpected action codes: {unknown_actions}")
    conflicting = (
        frame.groupby(["change_date", "stock_code"])["action"]
        .nunique()
        .loc[lambda value: value.gt(1)]
    )
    stats = {
        "workbook_count": len(files),
        "raw_rows": int(len(raw_frame)),
        "unique_event_rows": int(len(frame)),
        "exact_duplicate_rows": exact_duplicate_rows,
        "min_change_date": str(frame["change_date"].min()),
        "max_change_date": str(frame["change_date"].max()),
        "actions": {str(key): int(value) for key, value in frame["action"].value_counts().items()},
        "conflicting_date_code_count": int(len(conflicting)),
        "conflicting_date_codes": [list(index) for index in conflicting.index],
        "files": file_stats,
    }
    return frame.sort_values(["change_date", "action", "stock_code"]).reset_index(drop=True), stats


def scan_calendar(root: Path) -> tuple[list[str], dict[str, Any]]:
    files = sorted(root.rglob("IDX_Idxtrd*.xlsx"), key=lambda value: str(value).lower())
    dates: set[str] = set()
    file_stats: list[dict[str, Any]] = []
    for number, path in enumerate(files, 1):
        print(f"[calendar {number}/{len(files)}] {path.parent.name}/{path.name}", flush=True)
        headers, rows, workbook = iter_csmar_rows(path)
        rows_read = 0
        target_rows = 0
        try:
            for raw in rows:
                rows_read += 1
                row = raw[: len(headers)]
                code = norm_code(row[0] if row else None)
                if code and code > INDEX_CODE:
                    break
                if code != INDEX_CODE:
                    continue
                target_rows += 1
                value = norm_date(row[1])
                if value:
                    dates.add(value)
        finally:
            workbook.close()
        file_stats.append(
            {
                "path": str(path.resolve()),
                "rows_read_until_early_stop": rows_read,
                "target_rows": target_rows,
            }
        )
    ordered = sorted(dates)
    if not ordered:
        raise ValueError("No 000300 trading dates were found")
    return ordered, {
        "workbook_count": len(files),
        "unique_dates": len(ordered),
        "min_date": ordered[0],
        "max_date": ordered[-1],
        "files": file_stats,
    }


def group_events(frame: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"adds": set(), "deletes": set()}
    )
    for row in frame.itertuples(index=False):
        target = "adds" if row.action == ADD_ACTION else "deletes"
        grouped[row.change_date][target].add(row.stock_code)
    return dict(grouped)


def reverse_to_date(
    anchor_codes: set[str],
    events: dict[str, dict[str, set[str]]],
    anchor_date: str,
    target_date: str,
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    current = set(anchor_codes)
    state_violations: list[dict[str, Any]] = []
    count_checks: list[dict[str, Any]] = []
    for event_date in sorted(events, reverse=True):
        if event_date > anchor_date or event_date <= target_date:
            continue
        adds = events[event_date]["adds"]
        deletes = events[event_date]["deletes"]
        missing_added_after = sorted(adds - current)
        deleted_still_present_after = sorted(deletes & current)
        if missing_added_after or deleted_still_present_after:
            state_violations.append(
                {
                    "direction": "reverse",
                    "change_date": event_date,
                    "missing_added_after": missing_added_after,
                    "deleted_still_present_after": deleted_still_present_after,
                }
            )
        after_count = len(current)
        current.difference_update(adds)
        current.update(deletes)
        count_checks.append(
            {
                "change_date": event_date,
                "add_count": len(adds),
                "delete_count": len(deletes),
                "after_count": after_count,
                "before_count": len(current),
            }
        )
    return current, state_violations, count_checks


def scan_csi800_validation(
    root: Path, target_dates: set[str]
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    folder = root / "\u6307\u6570\u6210\u5206\u80a1\u6743\u91cd\u6587\u4ef65"
    files = sorted(folder.glob("IDX_Smprat*.xlsx"), key=lambda value: str(value).lower())
    found: dict[str, set[str]] = {value: set() for value in target_dates}
    file_stats: list[dict[str, Any]] = []
    max_target_date = max(target_dates)
    for number, path in enumerate(files, 1):
        print(f"[CSI800 {number}/{len(files)}] {path.name}", flush=True)
        headers, rows, workbook = iter_csmar_rows(path)
        target_rows = 0
        rows_read = 0
        try:
            for raw in rows:
                rows_read += 1
                row = raw[: len(headers)]
                code = norm_code(row[0] if row else None)
                if code and code > CSI800_CODE:
                    break
                if code != CSI800_CODE:
                    continue
                target_rows += 1
                row_date = norm_date(row[1])
                if row_date in found:
                    stock_code = norm_code(row[2])
                    if stock_code:
                        found[row_date].add(stock_code)
                if row_date and row_date > max_target_date:
                    break
        finally:
            workbook.close()
        file_stats.append(
            {
                "path": str(path.resolve()),
                "rows_read_until_stop": rows_read,
                "target_rows_until_stop": target_rows,
            }
        )
    return found, {"files": file_stats, "target_dates": sorted(target_dates)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csmar-root", type=Path, required=True)
    parser.add_argument("--anchor-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--skip-csi800", action="store_true")
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    anchor, anchor_date, markets, names = load_anchor(args.anchor_csv)
    changes, change_stats = scan_changes(args.csmar_root)
    calendar, calendar_stats = scan_calendar(args.csmar_root)
    events = group_events(changes)
    for row in changes.itertuples(index=False):
        markets.setdefault(row.stock_code, row.market or market_from_exchange("", row.stock_code))
        names.setdefault(row.stock_code, row.stock_name)

    target_dates = [value for value in calendar if args.start_date <= value <= args.end_date]
    if not target_dates:
        raise ValueError("Requested date range has no trading dates")
    first_date, last_date = target_dates[0], target_dates[-1]
    current, reverse_violations, count_checks = reverse_to_date(
        set(anchor["stock_code"]), events, anchor_date, first_date
    )

    normalized_events_path = args.output_dir / "csi300_csmar_changes_normalized.csv"
    changes.to_csv(normalized_events_path, index=False, encoding="utf-8-sig")
    count_checks_frame = pd.DataFrame(count_checks)
    count_checks_frame.to_csv(
        args.output_dir / "csi300_event_count_checks.csv", index=False, encoding="utf-8-sig"
    )

    daily_path = args.output_dir / "csi300_constituents_daily_2010_2025.csv.gz"
    snapshot_path = args.output_dir / "csi300_event_snapshots_2010_2025.csv.gz"
    interval_records: list[dict[str, Any]] = []
    active_since = {code: first_date for code in current}
    left_censored = set(current)
    daily_count_distribution: Counter[int] = Counter()
    forward_violations: list[dict[str, Any]] = []
    event_dates = sorted(value for value in events if first_date < value <= last_date)
    event_index = 0
    previous_trade_date = first_date
    validation_dates = {"2024-11-18"}
    reconstructed_validation: dict[str, set[str]] = {}

    with gzip.open(daily_path, "wt", encoding="utf-8", newline="") as daily_file, gzip.open(
        snapshot_path, "wt", encoding="utf-8", newline=""
    ) as snapshot_file:
        daily_writer = csv.writer(daily_file)
        snapshot_writer = csv.writer(snapshot_file)
        daily_writer.writerow(["trade_date", "index_code", "stock_code", "market", "qlib_code"])
        snapshot_writer.writerow(["change_date", "stock_code", "market", "qlib_code"])
        for day_index, trade_date in enumerate(target_dates):
            if day_index > 0:
                while event_index < len(event_dates) and event_dates[event_index] <= trade_date:
                    event_date = event_dates[event_index]
                    adds = events[event_date]["adds"]
                    deletes = events[event_date]["deletes"]
                    missing_deletes = sorted(deletes - current)
                    already_present_adds = sorted(adds & current)
                    if missing_deletes or already_present_adds:
                        forward_violations.append(
                            {
                                "direction": "forward",
                                "change_date": event_date,
                                "missing_deletes_before": missing_deletes,
                                "already_present_adds_before": already_present_adds,
                            }
                        )
                    for code in sorted(deletes):
                        if code in active_since:
                            interval_start = active_since.pop(code)
                            interval_records.append(
                                {
                                    "index_code": INDEX_CODE,
                                    "stock_code": code,
                                    "market": markets.get(code, market_from_exchange("", code)),
                                    "qlib_code": qlib_code(code, markets.get(code, market_from_exchange("", code))),
                                    "start_date": interval_start,
                                    "end_date": previous_trade_date,
                                    "left_censored": interval_start == first_date,
                                    "right_censored": False,
                                }
                            )
                    current.difference_update(deletes)
                    for code in adds:
                        active_since[code] = trade_date
                    current.update(adds)
                    for code in sorted(current):
                        market = markets.get(code, market_from_exchange("", code))
                        snapshot_writer.writerow([event_date, code, market, qlib_code(code, market)])
                    event_index += 1
            daily_count_distribution[len(current)] += 1
            for code in sorted(current):
                market = markets.get(code, market_from_exchange("", code))
                daily_writer.writerow([trade_date, INDEX_CODE, code, market, qlib_code(code, market)])
            if trade_date in validation_dates:
                reconstructed_validation[trade_date] = set(current)
            previous_trade_date = trade_date

    for code, start_date in active_since.items():
        market = markets.get(code, market_from_exchange("", code))
        interval_records.append(
            {
                "index_code": INDEX_CODE,
                "stock_code": code,
                "market": market,
                "qlib_code": qlib_code(code, market),
                "start_date": start_date,
                "end_date": last_date,
                "left_censored": start_date == first_date,
                "right_censored": True,
            }
        )
    intervals = pd.DataFrame(interval_records).sort_values(["qlib_code", "start_date"])
    intervals_path = args.output_dir / "csi300_membership_intervals_2010_2025.csv"
    intervals.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    qlib_path = args.output_dir / "csi300_qlib_instruments_2010_2025.txt"
    intervals[["qlib_code", "start_date", "end_date"]].to_csv(
        qlib_path, sep="\t", header=False, index=False
    )

    csi800_result: dict[str, Any] = {"skipped": args.skip_csi800}
    if not args.skip_csi800:
        csi800_dates = {anchor_date, "2024-11-18"}
        csi800_sets, csi800_scan = scan_csi800_validation(args.csmar_root, csi800_dates)
        reconstructed_validation[anchor_date] = set(anchor["stock_code"])
        comparisons: dict[str, Any] = {}
        for value in sorted(csi800_dates):
            reconstructed = reconstructed_validation.get(value, set())
            csi800 = csi800_sets.get(value, set())
            comparisons[value] = {
                "reconstructed_count": len(reconstructed),
                "csi800_count": len(csi800),
                "not_in_csi800_count": len(reconstructed - csi800),
                "not_in_csi800_codes": sorted(reconstructed - csi800),
                "is_subset": bool(reconstructed) and not (reconstructed - csi800),
            }
        csi800_result = {"skipped": False, "scan": csi800_scan, "comparisons": comparisons}

    per_date_balance = []
    for change_date, value in sorted(events.items()):
        per_date_balance.append(
            {
                "change_date": change_date,
                "add_count": len(value["adds"]),
                "delete_count": len(value["deletes"]),
                "difference": len(value["adds"]) - len(value["deletes"]),
            }
        )
    balance_frame = pd.DataFrame(per_date_balance)
    balance_frame.to_csv(
        args.output_dir / "csi300_event_date_balance.csv", index=False, encoding="utf-8-sig"
    )
    non_300_checks = [row for row in count_checks if row["before_count"] != 300 or row["after_count"] != 300]
    audit = {
        "single_process": True,
        "single_threaded": True,
        "method": "Official 2026-07-31 anchor, reverse CSMAR action 1 additions and action 2 deletions",
        "anchor": {
            "path": str(args.anchor_csv.resolve()),
            "date": anchor_date,
            "rows": int(len(anchor)),
            "unique_codes": int(anchor["stock_code"].nunique()),
        },
        "requested_range": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "first_trading_date": first_date,
            "last_trading_date": last_date,
            "trading_dates": len(target_dates),
        },
        "changes": change_stats,
        "calendar": calendar_stats,
        "event_date_balance": {
            "event_dates": int(len(balance_frame)),
            "unbalanced_dates": balance_frame.loc[balance_frame["difference"].ne(0)].to_dict("records"),
        },
        "state_checks": {
            "reverse_violation_count": len(reverse_violations),
            "reverse_violations": reverse_violations,
            "forward_violation_count": len(forward_violations),
            "forward_violations": forward_violations,
            "non_300_reverse_count_checks": non_300_checks,
            "daily_constituent_count_distribution": {
                str(key): int(value) for key, value in sorted(daily_count_distribution.items())
            },
        },
        "outputs": {
            "normalized_events": str(normalized_events_path.resolve()),
            "daily_membership_gzip": str(daily_path.resolve()),
            "event_snapshots_gzip": str(snapshot_path.resolve()),
            "membership_intervals": str(intervals_path.resolve()),
            "qlib_instruments": str(qlib_path.resolve()),
            "interval_rows": int(len(intervals)),
            "daily_rows": int(len(target_dates) * 300),
        },
        "csi800_validation": csi800_result,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "range": audit["requested_range"],
        "event_balance": audit["event_date_balance"],
        "state_checks": audit["state_checks"],
        "csi800_validation": audit["csi800_validation"],
        "outputs": audit["outputs"],
        "elapsed_seconds": audit["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
