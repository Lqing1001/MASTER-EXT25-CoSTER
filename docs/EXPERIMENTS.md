# Experiments

## Aggregate results

Run `python scripts/reproduce_results.py`. It uses 964 evaluation dates, ten matched seeds, Bartlett Newey–West lag 10, and 5,000 noncircular 20-day bootstrap blocks with seed 20261110. The supplied mean daily correlations are aggregate model outputs, not security-level scores. Wealth is compounded within each seed before averaging. Bands show the minimum–maximum seed range. Cost error bars use sample SD.

## Model training

Follow `docs/DATA_PREPARATION.md` for source layouts. Train/normalization: 2010-01-04–2019-12-24; validation: 2020-01-02–2021-12-24; test: 2022-01-04–2025-12-31. Inputs are eight daily observations with 221 features. The target holds from t+1 close to t+5 close (four intervals).

```bash
python -m experiments.train --dataset-root datasets/master_ext_strict_20191224_v1 --output-root runs/new_training --seeds 0 1 2 3 4 5 6 7 8 9 --dry-run
```

Remove `--dry-run` to train. This is a fresh experiment; exact historical numerical reproduction additionally requires the stored checkpoints and original staged initialization. It is not claimed that a new training run will produce bit-identical results. The original study reused existing seeds 0–2 and trained seeds 3–9 under `experiments/controls/train.py new_main`.

## Controlled experiments

The optional checkpoint folder contains `csi300/seed0/master.pt`, `base.pt`, `dense.pt`, `full.pt`, and the analogous files for all ten seeds and both universes. Set `COSTER_CHECKPOINT_ROOT` to that folder to use the exact main-model anchors. `MASTER_EXT_DATASET_ROOT` selects prepared data; `COSTER_EXPERIMENT_ROOT` selects a separate writable experiment output; `COSTER_DEVICE` defaults to `cuda:0`.

```bash
python experiments/controls/train.py ablation --universe csi300 --seed 0
python experiments/controls/train.py selector --universe csi300 --seed 0 --mode top2
python experiments/controls/analyze.py
python experiments/controls/membership.py
```

These commands require the licensed prepared arrays and all preceding cached predictions for the intended comparison. `analyze.py` discovers available predictions: only label a run ten-seed after verifying all seeds 0–9 are present. Do not set `COSTER_CHECKPOINT_ROOT` when training new main models: it is an input-anchor override.

`ensembles.py` retains the recorded baseline directory conventions for the deterministic Ridge and seed-0 GRU; those private files are not bundled. Its additional baseline training dependency is included. The exact historical experiment scripts are supplied for traceability, not advertised as a data-free one-command training pipeline.

Portfolio recalculation additionally requires `COSTER_PRICE_DB` (the prepared SQLite database), `COSTER_FACTOR_FILE` (CH-3 workbook), and verified total-return benchmark inputs under the experiment output's `data_audit/`. Generate those inputs with `audit_total_return_data.py` from licensed CSMAR workbooks. Then use `portfolio.py`; it evaluates freeze/cash and replacement rules. Borrow inventory, price-limit execution, impact, capacity and financing are not modeled.

## Manuscript

`paper/manuscript_v2.pdf` contains the manuscript; `paper/source/` contains its LaTeX sources. Compile `template.tex` using pdflatex and BibTeX.
