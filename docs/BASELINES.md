# Comparison baselines

All baselines share MASTER-EXT25, the chronological split, target processing and full-universe daily evaluation. Table 3 uses training seed 0. Run the suite using `experiments.train --include-baselines` or invoke the modules below with `--help`.

Set `MASTER_EXT_DATASET_ROOT` before individual baseline commands:

```powershell
$env:MASTER_EXT_DATASET_ROOT = (Resolve-Path datasets/master_ext_strict_20191224_v1).Path
```

```bash
export MASTER_EXT_DATASET_ROOT="$PWD/datasets/master_ext_strict_20191224_v1"
```

| Model | Module | Implementation |
|---|---|---|
| Ridge / Random Forest | `experiments.baselines.traditional` | 379-dimensional temporal summaries; Ridge validation search; RF 200 trees |
| XGBoost | `experiments.baselines.xgboost` | GPU histogram boosting, validation RMSE early stopping |
| LSTM / GRU / Transformer / LightGBM | `experiments.baselines.run_additional_baselines` | Sequence neural models; LightGBM on the same temporal summaries |
| StockMamba | `experiments.baselines.run_stockmamba_paper` | Published-equation reconstruction with pure-PyTorch Mamba-2 SSD |
| ACT | `experiments.baselines.act` | Dynamic-relation adaptation; static industry/region graphs are not used |
| PRISM-VQ | `experiments.baselines.run_prism_official_adapted` | Authors' VQVAE and loading-generator modules, adapted to the benchmark |
| MASTER | `CoSTER.py train-master` | Market-guided MASTER implementation |

Examples:

```bash
python -m experiments.baselines.traditional --model ridge --universe csi300 --seed 0 --output-dir runs/ridge/csi300
python -m experiments.baselines.act --universe csi300 --seed 0 --output-dir runs/act/csi300
python -m experiments.baselines.run_additional_baselines --model lightgbm --universe csi300 --seed 0
python -m experiments.baselines.run_stockmamba_paper --universe csi300 --seed 0
python -m experiments.baselines.run_prism_official_adapted --universe csi300 --seed 0
```

StockMamba uses 20 epochs, Adam at 1e-5 and FFN width 1024. The FFN width is an explicit reconstruction choice because the article does not specify it. SSD retains the upstream Apache-2.0 notice. This is a paper reconstruction, not author-provided StockMamba code.

PRISM-VQ upstream commit: `7d02635d0cdeec2f4e7278a9e73acd0eb67a8fa6`. Adaptations: 8-day histories, Market63 priors, a single five-day auxiliary target, shared training labels and validation selection. The frozen quantizer remains in evaluation mode; PyTorch's LayerNorm-specific fused encoder fast path is disabled for the upstream RMSNorm. Pretraining/prediction budgets are 4/12 epochs, prediction patience 4.

CoSTER, StockMamba and ACT have separate model classes; no baseline proxy versions are selected by the final experiment commands. `reference/predictive_seed0.csv` contains the displayed final comparison values.
