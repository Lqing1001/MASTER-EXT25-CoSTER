# Release verification

The public directory was checked independently of the original research imports.

- All 18 published command entry points passed `--help`; Python sources passed compilation.
- Three unit tests cover full-model output and parameter count, non-circular block intervals, portfolio holding timing, trade blocking, cost turnover identities and HAC regression covariance.
- ACT, LSTM, GRU, Transformer, StockMamba and PRISM-VQ passed finite-output forward checks.
- Six final checkpoint runs (two universes × seeds 0/1/2) were loaded from the separate release asset. Predictions on 2022-01-04 matched the corresponding saved paper predictions with maximum absolute error at most 4.45e-16.
- The CSI 300 seed-0 final predictions for MASTER, Continuous and CoSTER were run through the independent investment CLI for the entire test period. CAGR, Sharpe and maximum drawdown matched the final reference results in all 30 model/strategy/cost combinations.
- The paired IC and RankIC moving-block intervals for both universes matched the final reference values to floating-point precision.
- Matrix-operation counts matched the final paper: continuous-expert overhead 21.9576% at N=300 and 18.5407% at N=800; parameter counts MASTER 775041 and Continuous 402792.

This packaging check did not rerun full training, rebuild the full dataset or retrain the sensitivity models. The release contains those executable workflows and the final reference evidence. `reference/release_verification.json` records the numerical checks. `MANIFEST.json` contains file hashes; the separate checkpoint manifest records every weight file's digest.
