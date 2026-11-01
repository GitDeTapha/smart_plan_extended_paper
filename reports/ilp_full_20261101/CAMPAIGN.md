# ILP campaign ilp_full_20261101

Every patient is served: constraint C1 is an equality (`Config.serve_all = True`,
commit `519b07c`). This campaign replaces `../ilp_full_20261030/`, run with C1 as
an inequality, in which the ILP could leave patients out.

## Run

- Notebook: `notebooks/01_ilp_validation.ipynb`, executed with papermill into
  `ilp_validation_full_20261101.ipynb` (this folder), 44 cells, no error.
- Code: commit `afe8a67` of this repository.
- Inputs, both produced by the training run `run_full_20261030` (see
  `../run_full_20261030/`), SHA-256 checked before the run:
  - `data/inference/plne_parameters.json`: `89b7e176...0faa`
  - `results/run_full_20261030/holdout_scores.parquet` (patient level, not versioned): `334c5898...b5c2`
- Configuration: `CampaignConfig.production()`, one clinic day of 8 slots, R = 30
  seeds per cell, HiGHS with one thread per ILP, target gap 1e-6, time limit
  600 s per ILP, c_O = 0. Full configuration in `results/ilp_campaign_config.json`.
- Machine: Intel Core i7-6700HQ (4 cores, 8 threads), 16 GB RAM, Python 3.13.13;
  one solving process per physical core, peak 597 MB per process.
- Holdout q_i: 2,073 of 4,013,341 lie outside Q_CLIP = (0.01, 0.999) (isotonic
  calibration end blocks) and are clipped when they enter an instance.

### How the run was executed

The ILPs were solved by a first execution at commit `5fc0bc9`, from 2026-10-31
22:53 to 2026-11-01 07:35 UTC. The laptop was suspended from 00:53:49 to 07:13:44
UTC (system journal), so the solving time was about 2 h 22 min. Every result is
stored in a solution cache keyed by cell, seed, method and solver configuration.

HiGHS measures its time limit on the wall clock, which kept running during the
suspension. The three ILPs that were running at that moment, all in the holdout
days, therefore stopped on `maxTimeLimit` at wake-up, with a HiGHS run time of
about 22,810 s for 18 to 37 s of computation. No other ILP was running: the
design of experiments, the benchmarks and the sensitivity analyses had ended
before the suspension.

These three cache entries were moved out of the cache (kept, not deleted, in
`results/ilp_cache_serveall_suspend_quarantine/`, not versioned) and the notebook
was executed again at commit `afe8a67`, from 07:38 to 07:41 UTC. That execution
read every other ILP from the cache and solved the three holdout ILPs again; it
is the notebook stored here. Commit `afe8a67` changes only two figures of the
notebook (one label per point of the Pareto front, a single tick on a single-level
axis), not the model, the instances or the solver. Long runs are now executed
under `systemd-inhibit --what=sleep:idle`.

The three ILPs were also solved apart, in an empty cache, to check that the
suspension altered nothing but their status: `resolve_suspended_holdout.py`,
output in `holdout_days_resolved.csv`.

| Day | ILP | Status at wake-up | Gap at wake-up | Status re-solved | Model cost on the true day, both runs |
|---|---|---|---|---|---|
| 24 | oracle (true groups) | maxTimeLimit | 14.6 % | optimal | 201.446727 |
| 27 | oracle (true groups) | maxTimeLimit | 9.5 % | optimal | 66.747842 |
| 28 | ILP (predicted groups) | maxTimeLimit | 3.3 % | optimal | 135.239088 |

The incumbents found before the suspension were already optimal; only the proof
was missing. On day 27 the re-solved oracle returns another optimal schedule (same
model cost), whose realised cost is 66.5 instead of 69.5.

## Check 1: the ILP that reached the time limit

449 of the 450 ILPs of the design of experiments (`results/doe_ilp.csv`) are
proven optimal. One reached the 600 s limit: seed 1016 in the most overbooked
cell (16 patients for 8 slots, every slot double-booked), remaining gap 1.41 %
on the complete cost.

It was re-solved on the same laptop, on the same instance rebuilt from its cell
(the B3 warm-start cost matches the campaign to 1e-13), with a 3,600 s
limit: `resolve_time_limited.py`, output in `resolve_time_limited_output.jsonl`.

| Cell | Seed | n | Campaign Z (600 s) | Proven optimum Z* | Difference | Time to proof |
|---|---|---|---|---|---|---|
| ratio 2.0 | 1016 | 16 | 172.168131 | 172.138932 | 0.0292 (0.017 %) | 1,389 s |

The incumbent was within 0.017 % of the optimum: the limit interrupted the
proof of optimality, not the search. The results of the campaign are therefore
those of proven optima up to 0.017 % on one ILP of 450. This optimum equals
the one of the earlier campaign (`../ilp_full_20261030/`): with 16 patients for 8
slots, every slot holds two patients under both formulations of C1.

## Check 2: robust statistics of the holdout days

`results/holdout_days_summary.csv` reports the mean difference with a 95 %
confidence interval and a Wilcoxon signed-rank test. The mean of the realised cost
is sensitive to a few days with large differences, so the median and the number
of days won are given here, computed from `results/holdout_days.csv`
(difference = policy minus ILP; positive means the ILP is better; 100 days).
B1 serves 8 patients of 12; the other policies serve all 12.

| Policy | Cost | Mean | Median | Days where the ILP is better |
|---|---|---|---|---|
| ILP oracle (true groups) | expected, model | -19.90 | -13.20 | 0 |
| ILP oracle (true groups) | expected, day | -24.88 | -14.16 | 7 |
| ILP oracle (true groups) | realised | -24.61 | -13.50 | 9 (3 ties) |
| B1 single booking | expected, model | +16.01 | +13.62 | 83 |
| B1 single booking | expected, day | -3.56 | +7.05 | 62 |
| B1 single booking | realised | -4.75 | +7.00 | 63 |
| B3 greedy | expected, model | -0.26 | -0.00 | 47 (3 ties) |
| B3 greedy | expected, day | +0.83 | +0.34 | 52 (1 tie) |
| B3 greedy | realised | +1.21 | +2.75 | 52 (7 ties) |
| B5 uniform overbooking | expected, model | +13.12 | +9.75 | 72 |
| B5 uniform overbooking | expected, day | +29.25 | +16.08 | 73 |
| B5 uniform overbooking | realised | +31.32 | +14.25 | 69 (3 ties) |

For B1 on the realised cost, the mean (-4.75, interval [-14.24 ; 4.73], Wilcoxon
p = 0.39) is not significant, while the ILP is better on 63 days of 100 with a
median gain of 7.00: a few days with large differences pull the mean below zero.
B1 leaves 4 patients of 12 unscheduled each day, which its cost does not count.
B3 and the ILP are not distinguishable on the holdout days (p > 0.5 on the three
costs); the ILP is planned on the predicted groups, which are wrong for 33 % of
the patients, and judged on the true ones.
