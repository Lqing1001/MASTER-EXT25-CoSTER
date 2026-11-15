# MASTER-EXT25 and CoSTER

Code and reference results for **A New Chinese A-Share Market Benchmark for Data-Driven Investment Management: MASTER-EXT25 and the CoSTER Stock-Ranking Model**.

[中文说明](README_zh.md) · [Data preparation](docs/DATA_PREPARATION.md) · [Experiments](docs/EXPERIMENTS.md) · [Investment evaluation](docs/INVESTMENT.md)

MASTER-EXT25 extends Chinese A-share stock-ranking evaluation through 2025. CoSTER combines market-guided MASTER scores with a continuous temporal expert using learned temporal positions and sparse temporal selection. Each branch's scores are standardized over the daily universe before equal-weight combination.

## Contents

| Directory | Contents |
|---|---|
| `CoSTER.py` | Model, staged training and inference |
| `data_tools/` | Data preparation, features and validation |
| `experiments/` | Baselines, ablations and controlled experiments |
| `evaluation/`, `investment/` | Predictive and portfolio evaluation |
| `reference/` | Experimental results |
| `third_party/` | External modules and license notices |

## Quick start

```bash
pip install -r requirements.txt
python CoSTER.py demo
python scripts/reproduce_results.py
```

The demo uses random weights. The reproduction script uses the included aggregate results. Full training and evaluation require separately obtained CSMAR data; see the guides above.

## License

Project code is provided under the MIT license. Third-party components retain their respective licenses. See `CITATION.cff` for citation information.
