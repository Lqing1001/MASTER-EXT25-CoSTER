# Manuscript-to-evidence map

| Manuscript item | Evidence | Implementation |
|---|---|---|
| Seed-0 broad comparison | `reference/predictive_seed0.csv` | `experiments/baselines/` |
| Ten seeds; component matrix | `tables/per_seed.csv`, `tables/summary.csv` | `controls/train.py`, `analyze.py` |
| NW/Holm and bootstrap | `tables/newey_west.csv`, `tables/bootstrap.csv` | `analyze.py` |
| M+B minus MASTER | `figure_data/base_fusion_contrasts.csv` | `scripts/reproduce_results.py` |
| Selector controls | `tables/selectors.csv`, `tables/control_contrasts.csv` | `train.py`, `analyze.py` |
| Simple ensembles | `tables/ensembles.csv` | `ensembles.py` |
| Top-1 gradient diagnostics | `diagnostics/` | `train.py diagnostic` |
| Membership sensitivity | `membership/` | `membership.py` |
| Costs, long-only, alpha, exposures | `portfolio/` | `portfolio.py` |
| Paired seeds; rolling; bridge; wealth; costs | `figure_data/`, `figures/` | `scripts/reproduce_results.py` |

Evidence paths without a prefix are under `reference/`; controlled-experiment implementations are under `experiments/controls/`. The main comparisons use ten seeds; the broad baseline comparison uses seed 0.

M+B versus MASTER RankIC: CSI300 delta 0.002265, p=0.2112; CSI800 delta 0.003299, p=0.0179. These are unadjusted exploratory tests. For two-MASTER comparisons, five disjoint pairs are compared to CoSTER seeds 0,2,4,6,8, not to all ten CoSTER seeds.
