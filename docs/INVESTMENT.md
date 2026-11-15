# Investment evaluation

Portfolio evaluation uses four overlapping cohorts, entry at t+1 close and exit at t+5 close. The scenarios include long-only and theoretical long–short decile portfolios, one-way costs of 0/5/10/20/30 bp, and freeze/cash or replacement entry rules. Long-only active returns use total-return indices H00300 and H00906.

Aggregate results are in `reference/portfolio/`. To reproduce the figures from these results, run `python scripts/reproduce_results.py`.

For portfolio recomputation from individual-stock scores, use `experiments/controls/portfolio.py`. Set `COSTER_EXPERIMENT_ROOT` to the prediction/output directory, `COSTER_PRICE_DB` to the prepared SQLite database, and `COSTER_FACTOR_FILE` to the CH-3 factor workbook. The factor fields are `date`, `mktrf`, `SMB` and `VMG`; returns use decimal units. Total-return benchmark inputs belong in the experiment directory's `data_audit/` folder and can be prepared using `experiments/controls/audit_total_return_data.py`.

The input prices, stock-level predictions and factor workbooks must be obtained or generated separately. Short-borrow availability, financing, price-limit execution, market impact and capacity are outside the simulation.
