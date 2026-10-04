# Model and prediction experiments

Run commands from the repository root. Paths below are relative to that root. All numerical training entry points use one CPU numerical thread and serial daily cross-sections.

## Checkpoint inference

Extract `CoSTER_final_checkpoints.zip` into the repository. It contains both universes, seeds 0/1/2, and four state dictionaries per run. The manifest provides SHA-256 digests. The file names are the load API's storage names; the public model is CoSTER.

```bash
python -m evaluation.predict --dataset-root datasets/master_ext_strict_20191224_v1 --run-dir checkpoints/csi300/seed0 --universe csi300 --seed 0 --split test --output-dir runs/predictions/csi300/seed0 --device cuda
```

This writes complete-universe MASTER, Continuous and CoSTER predictions, daily IC/RankIC, and period summaries. Repeat for both universes and seeds 0/1/2. Inference must precede finite-label filtering: every stock in a daily universe participates in attention and score standardization.

## Training

```bash
python -m experiments.train --dataset-root datasets/master_ext_strict_20191224_v1 --checkpoint-root checkpoints --output-root runs/coster --include-baselines
```

`--checkpoint-root` reuses the frozen four-epoch encoder embedded in each released ranker checkpoint. All ranker/residual parameters are initialized and trained by the final algorithm. Omit the option to train the encoder from scratch as well; optimization randomness and GPU kernels can then yield different numerical checkpoints.

Individual commands for one run:

```bash
python CoSTER.py train-master --dataset-root datasets/master_ext_strict_20191224_v1 --universe csi300 --seed 0 --output-dir runs/coster/csi300/seed0/master
python CoSTER.py train-continuous --dataset-root datasets/master_ext_strict_20191224_v1 --universe csi300 --seed 0 --encoder-checkpoint checkpoints/csi300/seed0/continuous_prism/no_vq_csi300_seed0_base_best.pt --output-dir runs/coster/csi300/seed0/continuous_prism
python CoSTER.py train-sparse --dataset-root datasets/master_ext_strict_20191224_v1 --universe csi300 --seed 0 --selected 1 --continuous-dir runs/coster/csi300/seed0/continuous_prism --output-dir runs/coster/csi300/seed0/sparse_top1
```

Budgets: MASTER 40 epochs; continuous encoder/ranker/temporal/sparse stages 4/12/10/10 epochs. Stage checkpoint selection uses the relevant branch's validation V. The selected full-model sparsity is k=1, with equal standardized score weights. `--help` lists each stage's parameters.

## Validation sensitivity

```bash
python -m evaluation.sensitivity --dataset-root datasets/master_ext_strict_20191224_v1 --run-dir checkpoints/csi300/seed0 --output-dir runs/sensitivity --device cuda
```

Table 7: fix the final temporal anchor and train sparse residuals with k=2/4/8 for 10 epochs, with matched seed-0 initialization; reuse the final k=1 residual. Compare the resulting full CoSTER validation V. The weight scan uses the fixed k=1 expert and alpha=0/0.25/0.5/0.75/1, where alpha is the continuous score weight. This command never evaluates the test split.

## Paired inference

```bash
python -m evaluation.paired --candidate runs/predictions/csi300/seed0/CoSTER_daily.csv runs/predictions/csi300/seed1/CoSTER_daily.csv runs/predictions/csi300/seed2/CoSTER_daily.csv --baseline runs/predictions/csi300/seed0/MASTER_daily.csv runs/predictions/csi300/seed1/MASTER_daily.csv runs/predictions/csi300/seed2/MASTER_daily.csv --output-dir runs/paired/csi300
```

Input files must have identical stock-universe definitions and correspond to the same seeds in the same order. The tool verifies matching dates. For each metric, paired daily differences are averaged across seeds before Newey–West inference (lag 10) and the non-circular moving-block bootstrap (block 20, 5000 replicates, seed 20261001). Confidence intervals are percentile intervals for the mean daily difference.

## Additional final analyses

```bash
python -m evaluation.model_cost
python -m evaluation.membership --raw-features datasets/master_ext_clean_v1/alpha158_raw/by_instrument --predictions runs/predictions/csi300/seed0/MASTER_predictions.npz runs/predictions/csi300/seed0/CoSTER_predictions.npz --output runs/membership.json
```

Matrix-operation counting uses two FLOPs per multiply-accumulate and excludes normalization, activation, sorting and score combination. Membership sensitivity filters the fixed predictions at scoring time; it does not rerun a changed attention universe. The final reference values are in `reference/`.
