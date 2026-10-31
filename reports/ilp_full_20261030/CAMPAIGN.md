# ILP campaign ilp_full_20261030

## Run

- Notebook: `notebooks/01_ilp_validation.ipynb`, executed with papermill into
  `ilp_validation_full_20261030.ipynb` (this folder), 47 cells, no error.
- Code: commit `168793e` of this repository.
- Inputs, both produced by the training run `run_full_20261030` (see
  `../run_full_20261030/`), SHA-256 checked before the run:
  - `data/inference/plne_parameters.json`: `89b7e176...0faa`
  - `results/run_full_20261030/holdout_scores.parquet` (patient level, not versioned): `334c5898...b5c2`
- Configuration: `CampaignConfig.production()`, one clinic day of 8 slots, R = 30
  seeds per cell, HiGHS with one thread per ILP, target gap 1e-6, time limit
  600 s per ILP, c_O = 0. Full configuration in `results/ilp_campaign_config.json`.
- Machine: Intel Core i7-6700HQ (4 cores, 8 threads), 16 GB RAM, Python 3.13.13;
  one solving process per physical core, peak 565 MB per process.
- Duration: 2026-10-31 05:31 to 07:19 UTC (1 h 48 min).
- Holdout q_i: 2,073 of 4,013,341 lie outside Q_CLIP = (0.01, 0.999) (isotonic
  calibration end blocks) and are clipped when they enter an instance.

## Check 1: the two ILPs that reached the time limit

448 of the 450 ILPs of the design of experiments (`results/doe_ilp.csv`) are
proven optimal. Two reached the 600 s limit, both with seed 1016 in the two most
overbooked cells (14 and 16 patients for 8 slots); their remaining gap on the
complete cost was 0.47 % and 1.09 %. Other seeds of these cells need up to 343 s.

They were re-solved on the same laptop, on the same instances rebuilt from their
cell (the B3 warm-start cost matches the campaign to the last digit), with a
3,600 s limit: `resolve_time_limited.py`, output in
`resolve_time_limited_output.jsonl`.

| Cell | Seed | n | Campaign Z (600 s) | Proven optimum Z* | Difference | Time to proof |
|---|---|---|---|---|---|---|
| ratio 1.75 | 1016 | 14 | 177.544948 | 177.544948 | 0 | 693 s |
| ratio 2.0 | 1016 | 16 | 172.142431 | 172.138932 | 0.0035 (0.002 %) | 1,153 s |

Both incumbents were already optimal or within 0.002 % of the optimum: the limit
interrupted the proof of optimality, not the search. The results of the campaign
are therefore those of proven optima up to 0.002 % on one ILP of 450.

## Check 2: robust statistics of the holdout days

`results/holdout_days_summary.csv` reports the mean difference with a 95 %
confidence interval and a Wilcoxon signed-rank test. The mean of the realised cost
is sensitive to a few days with large differences, so the median and the number
of days won are given here, computed from `results/holdout_days.csv`
(difference = policy minus ILP; positive means the ILP is better; 100 days).

| Policy | Cost | Mean | Median | Days where the ILP is better |
|---|---|---|---|---|
| ILP oracle (true groups) | expected, model | -20.61 | -13.45 | 0 |
| ILP oracle (true groups) | expected, day | -25.24 | -12.81 | 12 |
| ILP oracle (true groups) | realised | -26.74 | -13.25 | 13 (4 ties) |
| B1 single booking | expected, model | +16.25 | +13.38 | 88 |
| B1 single booking | expected, day | +1.36 | +9.70 | 70 |
| B1 single booking | realised | -0.52 | +8.25 | 71 |
| B3 greedy | expected, model | -0.02 | -0.01 | 48 (2 ties) |
| B3 greedy | expected, day | +5.76 | +3.08 | 57 |
| B3 greedy | realised | +5.44 | +3.00 | 55 (6 ties) |
| B5 uniform overbooking | expected, model | +13.36 | +11.13 | 70 |
| B5 uniform overbooking | expected, day | +34.18 | +14.26 | 72 |
| B5 uniform overbooking | realised | +35.55 | +13.25 | 69 (3 ties) |

For B1 on the realised cost, the mean (-0.52, interval [-8.23 ; 7.19]) and the
Wilcoxon test (p = 0.038) do not disagree: the ILP is better on 71 days of 100
with a median gain of 8.25, while a few days where every booked patient came
(the ILP overbooked, B1 did not) pull the mean down to about zero.
