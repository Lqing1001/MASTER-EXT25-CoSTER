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
| `third_party/` | PRISM-VQ modules and upstream license notices |
