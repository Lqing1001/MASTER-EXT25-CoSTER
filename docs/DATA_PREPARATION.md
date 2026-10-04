# Data preparation

The paper uses licensed Chinese A-share data from CSMAR. No raw data or security-level arrays are included. The final source-file inventory is provided in `reference/source_files.csv`. Acquire complete source exports and preserve the original workbook names and field headers. The scripts use openpyxl, not OCR.

## Inputs

Place stock exports anywhere under `datasets/master_extension_raw/`. Place index exports and the constituent anchor under its `index_csmar/` subdirectory. Subdirectories are searched recursively. Complete CSI 800 stock coverage is required; a collection limited to CSI 300 quotes is insufficient.

| Filename pattern | Purpose / required field identifiers |
|---|---|
| `LS_SymbolQuotationD*.xlsx` | `Symbol`, `TradingDate`, `OpenPrice`, `HighPrice`, `LowPrice`, `ClosePrice`, `Volume`, `Amount`, `MarketValue`, `CirculatedMarketValue` |
| `TRD_AdjustFactor*.xlsx` | `Symbol`, `TradingDate`, `FwardFactor`, `BwardFactor`, `CumulateFwardFactor`, `CumulateBwardFactor` |
| `IDX_Idxtrd*.xlsx` | index trading calendar and prices: `Indexcd`, `Idxtrd01`, `Idxtrd05`, `Idxtrd07` |
| `IDX_Smprat*.xlsx` | CSI 800 snapshot membership: `Indexcd`, `Enddt`, `Stkcd`; original weight fields are retained by the source |
| `IDX_Chgsmp*.xlsx` | constituent-change records; preserve all original CSMAR column names and action codes |
| `IDX_Idxinfo*.xlsx` | Index basic information; retain the native CSMAR fields for source validation |
| CSI 300 anchor CSV | `snapshot_date,index_code,stock_code,stock_name,exchange` |

The workbook readers expect the native CSMAR export layout: field names in row 1, descriptive rows 2–3, observations starting at row 4. Dates and zero-padded six-digit codes are normalized by the readers. `IDX_Chgsmp` action 1 is addition, action 2 is deletion; reconstruction reverses effective-date events from a verified anchor.

The experiment used an official CSI 300 closing-weight anchor dated **2026-07-31**, containing exactly 300 distinct securities for index `000300`. Retain constituent-change records through that anchor (including the June 2026 changes), even though the feature/test period ends in 2025. The anchor reconstructs historical membership; it is not a predictor input. Obtain the corresponding anchor and changes under their source terms. A later, different source vintage is not guaranteed to reconstruct identical observations.

## Build sequence

Run from the repository root. Replace paths with your licensed source locations if needed.

```bash
python -m data_tools.reconstruct_csi300_history --csmar-root datasets/master_extension_raw/index_csmar --anchor-csv datasets/master_extension_raw/index_csmar/external_anchors/official_csi300_closeweight_20260731.csv --output-dir datasets/csi300_reconstructed_2010_2025 --audit-output reports/csi300_reconstruction.json
python -m data_tools.build_master_ext_clean_v1 --raw-root datasets/master_extension_raw --csi300-membership datasets/csi300_reconstructed_2010_2025/csi300_constituents_daily_2010_2025.csv.gz --output-dir datasets/master_ext_clean_v1 --audit-dir reports/data_raw
python -m data_tools.rebuild_strict_train_only_packages --source-dataset-root datasets/master_ext_clean_v1 --output-dataset-root datasets/master_ext_strict_20191224_v1 --audit-output reports/data_strict.json
python -m data_tools.validate_master_ext_semantics --dataset-root datasets/master_ext_clean_v1
```

The raw feature builder extracts quotes, factors and memberships and creates `alpha158_raw/by_instrument/*.npz` plus `market63.npz`. **The raw-feature directory is not the training dataset.** The final packager fits Alpha158 statistics separately for CSI 300 and CSI 800 and fits shared Market63 statistics using only 2010-01-04 through 2019-12-24. It produces a PASS/FAIL audit and refuses to overwrite a nonempty output directory.

The raw Alpha158 archives retain `dates`, `features`, `label`, `is_csi300`, `is_csi800`. They support the membership sensitivity analysis. Market63 uses indices 000300, 000905 and 000906 in that order, with 21 features each. CSI 500 is a market-feature source, not a third prediction universe.

Additional reconstruction checks are available through:

```bash
python -m data_tools.validate_csi300_reconstruction_outputs --help
python -m data_tools.validate_csi300_csi800_snapshots --help
```

## Final training schema

```text
datasets/master_ext_strict_20191224_v1/
  manifest.json
  market63.npz
  alpha158_robust_stats_csi300_20100104_20191224.npz
  alpha158_robust_stats_csi800_20100104_20191224.npz
  master_input/
    csi300/by_year/year=2010/ ... year=2025/
    csi800/by_year/year=2010/ ... year=2025/
```

Each annual directory contains `features.npy` (N×221 float32), `labels.npy` (N float32), `dates.npy` (N integer YYYYMMDD), and `instruments.npy` (N byte strings such as `SH600000`, dtype S8). Rows are sorted by date/security. The input reader assembles eight-day histories from these daily arrays; it does not load a precomputed N×8×221 tensor.

The final calendar has 3,886 dates. Test predictions span 969 dates, with 964 valid daily-correlation dates; the last five dates have unavailable forward labels. The raw target is adjusted close at t+5 divided by adjusted close at t+1 minus one. Training masks, within-window padding and early-sample feature boundary treatments follow the supplied implementation and manuscript Section 3.2.

`reference/data_audit.json` contains aggregate expected package sizes, fit boundaries and a source-manifest fingerprint. Source licenses, versions, missing instruments and export cutoffs can change reconstructed values. These inputs must be resolved before comparing model results. Workbooks, SQLite files, arrays, generated source inventories and detailed prediction archives remain local and are ignored by Git.
