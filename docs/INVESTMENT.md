# Investment experiments

The input is a CSV manifest with `universe,model,seed,path`. Paths are relative to the manifest file, or absolute local paths. Each NPZ must contain `dates,instruments,labels,predictions`. Start from `configs/investment_manifest.example.csv` and list MASTER, Continuous and CoSTER for both universes and all seeds.

```bash
python -m investment.run --manifest configs/investment_manifest.example.csv --database datasets/master_ext_clean_v1/intermediate/master_ext_clean.sqlite --output-dir runs/investment
```

For CH-3 attribution, add `--factors PATH_TO_CH3.xlsx` (CSV also supported). Required fields: integer `date` in YYYYMMDD and decimal-return `mktrf,SMB,VMG`. Factors should cover every return date. The final study used the CH-3 workbook `CH3_factors_daily_202608.xlsx` from the Mingshi database (https://www.mingshiim.com/database). Obtain the source workbook separately.

## Accounting convention

- Signals use the complete prediction universe at date t. Rank endpoints contain floor(10% × N) securities each.
- Enter at the next close and hold through t+5, using four overlapping cohorts. Each cohort receives one-quarter of the strategy allocation.
- Long–short exposure is +0.5 on the top decile and −0.5 on the bottom decile at full allocation. This is a theoretical spread portfolio; no stock-borrow availability/cost model is included.
- Long-only exposure uses the top decile, reserves frozen positions first, and caps liquid purchases at the remaining capital.
- Missing/zero-volume/zero-amount days block trading. Existing positions use carried-forward marked prices; failed entries stay in cash and are not reranked using next-day information.
- One-way turnover = half of the absolute drift-adjusted weight change. Cost = quoted one-way basis points × full absolute weight change. Net returns are gross daily P&L minus these costs.
- The final test price window is 2022–2025. A new cohort is opened only when its full holding horizon remains in the window. The kernel reproduces the paper's fixed weight-based accounting; it is not an order-book execution simulator.

## Outputs

`per_seed.csv` reports CAGR, Sharpe, maximum drawdown and turnover at 0/5/10/20/30 bps. Long-only rows also include benchmark CAGR, annual mean active return and information ratio. `summary.csv` reports mean and sample standard deviation across seeds. `daily.csv` contains strategy-level daily returns; `audit.json` checks prices against the original forward-return labels and reports blocked trades.

When factors are supplied, `factor_regressions.csv` reports seed-average long–short CH-3 regressions and same-seed CoSTER-minus-MASTER incremental alpha at 0 and 10 bps. HAC uses Bartlett lag 10, the n/(n−k) correction and asymptotic normal p-values; alpha is annualized by 252. The spread-P&L convention does not subtract a full unit of risk-free return. Inference averages paired runs by date before regression.
