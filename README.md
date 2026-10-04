# MASTER-EXT25 and CoSTER

Code and final reference results for **A New Chinese A-Share Market Benchmark for Data-Driven Investment Management: MASTER-EXT25 and the CoSTER Stock-Ranking Model**.

[中文说明](README_zh.md) · [Data preparation](docs/DATA_PREPARATION.md) · [Experiments](docs/EXPERIMENTS.md) · [Investment evaluation](docs/INVESTMENT.md)

MASTER-EXT25 extends point-in-time Chinese A-share stock-ranking evaluation through 2025. CoSTER combines market-guided MASTER scores with a continuous temporal expert using learned temporal positions and sparse temporal selection. Each score is standardized over the complete daily universe before equal-weight combination.

## Contents

| Directory | Contents |
|---|---|
| `CoSTER.py` | Self-contained final model, staged training and inference |
| `data_tools/` | CSMAR ingestion, constituent reconstruction, Alpha158/Market63 features, train-only normalization and validation |
| `experiments/` | Serial training and all final comparison baselines |
| `evaluation/` | Predictions, paired inference, validation sensitivity, membership sensitivity and matrix operation counts |
| `investment/` | Overlapping-cohort long–short/long-only backtests, costs and CH-3 attribution |
| `configs/` | Final experimental protocol and prediction-manifest example |
| `reference/` | Final manuscript tables, aggregate/per-seed results, hashes and environment |
| `paper/` | Final English/Chinese PDFs, architecture diagram and bibliography |
| `third_party/` | PRISM-VQ modules and upstream license notices |

## Quick start

Python 3.12 was used for the experiments. Install the appropriate PyTorch 2.8 build for your machine, then:

```bash
python -m pip install -r requirements.txt
python -m unittest tests.test_release
python CoSTER.py demo
```

The demo uses random weights. CUDA is required by the training entry points for CoSTER and ACT; prediction, portfolio evaluation and unit tests support CPU. The recorded training environment is in `reference/environment.json`.

Prepare licensed CSMAR source files using [the data guide](docs/DATA_PREPARATION.md). Training can then be launched from the repository root:

```bash
python -m experiments.train --dataset-root datasets/master_ext_strict_20191224_v1 --output-root runs/coster --include-baselines --dry-run
```

Remove `--dry-run` to execute. Use a new output directory. For matched training with the paper's frozen encoder, first extract the separate **CoSTER_final_checkpoints.zip** release asset into this repository and add `--checkpoint-root checkpoints`. Without this option, the four-epoch encoder is trained from scratch. See [experiment details](docs/EXPERIMENTS.md) for single-stage and inference commands.

## Final protocol

| Item | Setting |
|---|---|
| Train and normalization fit | 2010-01-04 to 2019-12-24 |
| Validation | 2020-01-02 to 2021-12-24 |
| Test | 2022-01-04 to 2025-12-31 |
| Universes | CSI 300, CSI 800 |
| Input | 8 days × (158 stock + 63 market features) |
| CoSTER/MASTER training seeds | 0, 1, 2 |
| Broad baseline comparison | Seed 0 |
| Temporal sparsity | k=1, supported by full-model validation V over {1,2,4,8} on CSI 300 seed 0 |
| Continuous score weight | 0.5 |

The same k and weight are used across both universes and all seeds. The sensitivity experiment uses validation data; test data are used for final evaluation.

## Reference results

Final tables are in `reference/tables/`; Table 3 is also provided as `reference/predictive_seed0.csv`. `reference/final_evidence.json` preserves multi-seed prediction and portfolio evidence. The reference files contain final results only. Training commands generate their own run logs locally.

CSMAR raw exports, security-level features/predictions and CH-3 source workbooks are not included. Users must obtain the relevant source files. Trained weights are distributed separately from the Git source tree; their hashes are in `reference/checkpoints_manifest.json`.

## Citation and license

Authors: Qingqing Liu, Hong Rao and Xiaorui Dong. See `CITATION.cff` and the final manuscript. No publication DOI is assigned here. Project code is under the included MIT license; vendored components retain their own licenses. See [third-party notices](THIRD_PARTY_NOTICES.md). Manuscript PDFs are provided for reference and are not covered by the software license.
