"""Rebuild CSI300/CSI800 packages with strictly training-only normalization.

This script consumes the already generated raw Alpha158 instrument partitions.
It is single-process and forces numerical backends to one thread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
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


FIT_START = 20100104
FIT_END = 20191224
ROBUST_EPS = 1e-12


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_stats(raw_files: list[Path], mask_name: str, output: Path) -> dict[str, int]:
    chunks: list[np.ndarray] = []
    rows = 0
    fit_min = None
    fit_max = None
    for path in raw_files:
        with np.load(path) as data:
            select = data[mask_name] & (data["dates"] >= FIT_START) & (data["dates"] <= FIT_END)
            if select.any():
                chunk = data["features"][select]
                selected_dates = data["dates"][select]
                chunks.append(chunk)
                rows += len(chunk)
                current_min = int(selected_dates.min())
                current_max = int(selected_dates.max())
                fit_min = current_min if fit_min is None else min(fit_min, current_min)
                fit_max = current_max if fit_max is None else max(fit_max, current_max)
    train = np.concatenate(chunks, axis=0)
    median = np.nanmedian(train, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(train - median), axis=0).astype(np.float32)
    scale = ((mad + ROBUST_EPS) * 1.4826).astype(np.float32)
    np.savez(output, median=median, mad=mad, scale=scale, training_rows=np.int64(rows),
             fit_date_min=np.int32(fit_min), fit_date_max=np.int32(fit_max))
    del train, chunks
    return {"training_rows": rows, "features": 158, "fit_date_min": fit_min,
            "fit_date_max": fit_max, "validation_rows_used": 0}


def fit_market(source: Path, output: Path) -> dict[str, int]:
    with np.load(source) as data:
        dates = np.array(data["dates"], copy=True)
        raw = np.array(data["raw"], copy=True)
        names = np.array(data["names"], copy=True)
    select = (dates >= FIT_START) & (dates <= FIT_END)
    train = raw[select]
    median = np.nanmedian(train, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(train - median), axis=0).astype(np.float32)
    scale = ((mad + ROBUST_EPS) * 1.4826).astype(np.float32)
    normalized = np.clip((raw - median) / scale, -3, 3)
    normalized[~np.isfinite(normalized)] = 0.0
    np.savez(output, dates=dates, raw=raw, normalized=normalized.astype(np.float32),
             median=median, mad=mad, scale=scale, names=names,
             fit_date_min=np.int32(dates[select].min()),
             fit_date_max=np.int32(dates[select].max()))
    return {"training_days": int(select.sum()), "features": int(raw.shape[1]),
            "fit_date_min": int(dates[select].min()),
            "fit_date_max": int(dates[select].max()), "validation_days_used": 0}

def package_universe(
    raw_files: list[Path],
    output_root: Path,
    universe: str,
    mask_name: str,
    stats_path: Path,
    market_path: Path,
) -> dict[str, object]:
    stats = np.load(stats_path)
    median = stats["median"]
    scale = stats["scale"]
    market = np.load(market_path)
    market_map = {int(day): row for day, row in zip(market["dates"], market["normalized"])}
    counts: Counter[int] = Counter()
    for path in raw_files:
        with np.load(path) as data:
            selected_dates = data["dates"][data[mask_name]]
            counts.update((selected_dates // 10000).tolist())
    universe_root = output_root / universe / "by_year"
    universe_root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {"universe": universe, "mask": mask_name, "years": {}}
    for year in sorted(counts):
        nrows = counts[year]
        year_dir = universe_root / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "features": np.lib.format.open_memmap(year_dir / "features.npy", mode="w+", dtype=np.float32, shape=(nrows, 221)),
            "labels": np.lib.format.open_memmap(year_dir / "labels.npy", mode="w+", dtype=np.float32, shape=(nrows,)),
            "dates": np.lib.format.open_memmap(year_dir / "dates.npy", mode="w+", dtype=np.int32, shape=(nrows,)),
            "instruments": np.lib.format.open_memmap(year_dir / "instruments.npy", mode="w+", dtype="S8", shape=(nrows,)),
        }
        cursor = 0
        for path in raw_files:
            with np.load(path) as data:
                select = data[mask_name] & (data["dates"] // 10000 == year)
                count = int(select.sum())
                if not count:
                    continue
                raw = data["features"][select]
                normalized = np.clip((raw - median) / scale, -3, 3)
                normalized[~np.isfinite(normalized)] = 0.0
                dates = data["dates"][select]
                end = cursor + count
                arrays["features"][cursor:end, :158] = normalized
                arrays["features"][cursor:end, 158:] = np.vstack([market_map[int(day)] for day in dates])
                arrays["labels"][cursor:end] = data["label"][select]
                arrays["dates"][cursor:end] = dates
                arrays["instruments"][cursor:end] = path.stem.encode("ascii")
                cursor = end
        if cursor != nrows:
            raise AssertionError((universe, year, cursor, nrows))
        for array in arrays.values():
            array.flush()
        del array
        del arrays
        date_view = np.load(year_dir / "dates.npy", mmap_mode="r")
        instrument_view = np.load(year_dir / "instruments.npy", mmap_mode="r")
        date_values = np.array(date_view, copy=True)
        instrument_values = np.array(instrument_view, copy=True)
        del date_view, instrument_view
        order = np.lexsort((instrument_values, date_values))
        del date_values, instrument_values
        if not np.array_equal(order, np.arange(nrows)):
            for filename in ("features.npy", "labels.npy", "dates.npy", "instruments.npy"):
                source = np.load(year_dir / filename, mmap_mode="r")
                temporary = year_dir / (filename + ".sorted")
                target = np.lib.format.open_memmap(temporary, mode="w+", dtype=source.dtype, shape=source.shape)
                for start in range(0, nrows, 25_000):
                    target[start : start + 25_000] = source[order[start : start + 25_000]]
                target.flush()
                del source, target
                os.replace(temporary, year_dir / filename)
        features = np.load(year_dir / "features.npy", mmap_mode="r")
        labels = np.load(year_dir / "labels.npy", mmap_mode="r")
        dates = np.load(year_dir / "dates.npy", mmap_mode="r")
        instruments = np.load(year_dir / "instruments.npy", mmap_mode="r")
        ordered = bool(np.all((dates[1:] > dates[:-1]) | ((dates[1:] == dates[:-1]) & (instruments[1:] >= instruments[:-1]))))
        audit["years"][str(year)] = {
            "rows": nrows,
            "shape": list(features.shape),
            "finite_features": bool(np.isfinite(features).all()),
            "finite_labels": int(np.isfinite(labels).sum()),
            "ordered": ordered,
            "path": str(year_dir.resolve()),
        }
        del features, labels, dates, instruments
        log(f"{universe} {year}: {nrows:,} rows")
    audit["rows"] = int(sum(counts.values()))
    audit["status"] = "PASS" if all(
        item["shape"][1] == 221 and item["finite_features"] and item["ordered"]
        for item in audit["years"].values()
    ) else "FAIL"
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-root", type=Path,
                        default=Path("datasets/master_ext_clean_v1"))
    parser.add_argument("--output-dataset-root", type=Path,
                        default=Path("datasets/master_ext_strict_20191224_v1"))
    parser.add_argument("--audit-output", type=Path,
                        default=Path("audit_outputs/master_ext_strict_20191224_v1_audit.json"))
    args = parser.parse_args()
    source_root = args.source_dataset_root.resolve()
    output_root = args.output_dataset_root.resolve()
    if source_root == output_root:
        raise ValueError("strict output must differ from source dataset")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"strict output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_files = sorted((source_root / "alpha158_raw" / "by_instrument").glob("*.npz"))
    if not raw_files:
        raise ValueError("No raw Alpha158 partitions found")
    source_manifest = source_root / "manifest.json"
    market_source = source_root / "market63.npz"
    if not source_manifest.exists() or not market_source.exists():
        raise FileNotFoundError("source manifest or raw Market63 package missing")

    log("fitting shared Market63 statistics through 2019-12-24")
    market = fit_market(market_source, output_root / "market63.npz")
    stats: dict[str, object] = {}
    packages: dict[str, object] = {}
    date_tag = str(FIT_END)
    for universe, mask in (("csi300", "is_csi300"), ("csi800", "is_csi800")):
        stats_path = output_root / f"alpha158_robust_stats_{universe}_20100104_{date_tag}.npz"
        log(f"fitting {universe} strictly training-only robust statistics")
        stats[universe] = fit_stats(raw_files, mask, stats_path)
        packages[universe] = package_universe(
            raw_files, output_root / "master_input", universe, mask,
            stats_path, output_root / "market63.npz",
        )

    temporal_audit = {
        "training_feature_cutoff": FIT_END,
        "validation_start": 20200102,
        "alpha158_fit_max": {key: value["fit_date_max"] for key, value in stats.items()},
        "market63_fit_max": market["fit_date_max"],
        "validation_feature_overlap": False,
        "labels_used_for_normalization": False,
    }
    status = "PASS" if (
        all(item["status"] == "PASS" for item in packages.values())
        and all(item["fit_date_max"] <= FIT_END for item in stats.values())
        and market["fit_date_max"] <= FIT_END
    ) else "FAIL"
    result = {
        "material_passport": {
            "dataset": "MASTER-EXT-strict-20191224-v1",
            "source_dataset": str(source_root),
            "source_manifest_sha256": sha256(source_manifest),
            "mode": "single-process single-threaded",
            "normalization": "separate strictly training-only RobustZScoreNorm per stock universe",
            "fit_range": ["2010-01-04", "2019-12-24"],
            "validation": ["2020-01-02", "2021-12-24"],
            "test": ["2022-01-04", "2025-12-31"],
            "feature_shape": "158 Alpha158 + 63 market = 221",
        },
        "market63": market,
        "stats": stats,
        "packages": packages,
        "temporal_audit": temporal_audit,
        "status": status,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"status={status} saved={args.audit_output}")
    if status != "PASS":
        raise RuntimeError("strict dataset audit failed")

if __name__ == "__main__":
    main()

