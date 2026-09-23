# Smart Plan — Cluster-Predict-Schedule

Data-driven outpatient appointment scheduling. Two predictive models (no-show
probability, service-time group) feed a stochastic overbooking integer program
that allows at most two patients per slot, trading slot idle time against
patient waiting time.

The pipeline follows CRISP-DM and lives in a single notebook:

| Section | Content |
|---|---|
| 1-2 | Business understanding, exploration of the raw consultations |
| 3 | Cleaning, patient-history features, one leak-proof 70/20/10 split |
| 4 | No-show classifiers and service-time clustering + group classifier |
| 5 | Test-set evaluation and probability calibration (Brier) |
| 6 | Serialised inference artifacts, single `predict_patient` entry point |
| 7 | ILP instance built from held-out patients, solved with Pyomo/HiGHS |

## Repository layout

```
data_driven_plan.py      all reusable functions (I/O, features, models, ILP core)
gen_main_nb.py           generator of the notebook — edit this, not the .ipynb
data_driven_plan.ipynb   the pipeline: orchestration, narrative, outputs
scripts/csv_to_parquet.py  converts the raw CSV into chunked Parquet input
requirements.txt         pinned dependencies (Python 3.13)
LICENSE                  MIT
```

The dataset is **not** part of this repository: it contains patient-level
consultation records. `data/` and every generated
artifact (`figures/`, `models/`, `results/`, `tables/`) are untracked.

## Setup

```bash
python3 -m venv ml_env
./ml_env/bin/pip install -r requirements.txt
```

The ILP uses HiGHS through Pyomo (`highspy`); CBC is used as a fallback if
HiGHS is unavailable.

## Running

Convert the raw CSV once, then run the notebook top to bottom:

```bash
./ml_env/bin/python scripts/csv_to_parquet.py \
  --csv data/consult_waits_2024_03_25.csv \
  --out-dir data/raw_data_parquet --chunksize 500000 --compression snappy

./ml_env/bin/jupyter lab data_driven_plan.ipynb
```

All data paths derive from `SMART_PLAN_DATA` (default `./data`), so the
notebook runs unchanged against a dataset stored elsewhere:

```bash
export SMART_PLAN_DATA=/mnt/volume/data
```

To run it headless on a server, execute it with nbconvert:

```bash
SMART_PLAN_DATA=/path/to/data jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 --output run.ipynb data_driven_plan.ipynb
```

Section 3.2bis and sections 4a/4b each provide a resume point: once the
preprocessed Parquet file and the model bundles exist, the expensive cells can
be skipped on a later run.

## Results of a validated run

`reports/` holds one executed run on the full dataset (40,133,413 rows;
train 28.1M / test 8.0M / inference holdout 4.0M). Precision, recall and
F1-score are macro-averaged over both classes.

| Model | AUC | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| XGBoost | **0.7538** | 0.9455 | 0.5764 | 0.5855 | 0.5806 |
| RandomForest | 0.7302 | 0.9475 | 0.5735 | 0.5742 | 0.5738 |
| AdaBoost | 0.7248 | 0.9265 | 0.5427 | 0.5710 | 0.5516 |
| ANN | 0.6396 | 0.9312 | 0.5325 | 0.5462 | 0.5374 |

No-show prediction, test set, at each model's optimal threshold. Brier score of
the calibrated XGBoost: 0.0290. The A/B service-time classifier ranks the same
way (XGBoost 0.7362). Clustering: K=2, bootstrapped silhouette 0.6917 ± 0.0035,
cutoff 38.5 min. The ILP instance built from 30 held-out patients over 6 slots
solves in 3.3 s (Z_complete 40.7221).

- `run_20260923_0229.ipynb` — the executed notebook, figures included
- `recap_data_driven_plan.csv`, `training_history.csv` — summary tables

The decision threshold trades recall against accuracy at identical AUC: 0.5855
recall and 0.9455 accuracy under `f1_macro` (threshold 0.75) against 0.6943
recall and 0.7154 accuracy under `youden` (threshold 0.43). Which one to report
depends on the cost of missing a no-show against the cost of overbooking.

## Tuning the models

Every training lever is exposed in the notebook's section 4.0 as a
`TrainingConfig`: model families, search strategy (`grid`/`random`/`none`) and
budget, cross-validation folds, optimised score, subsample sizes, SMOTE, class
weighting and the decision-threshold criterion. Nothing is hard-coded in
`data_driven_plan.py`.

Section 4c compares runs: the threshold criterion alone (no retraining),
a single retrained family (`train_single_model`), or a full campaign.
`record_training_run(...)` archives each attempt and `training_history_table()`
puts them side by side.

## Editing the notebook

`data_driven_plan.ipynb` is generated. Change `gen_main_nb.py`, then:

```bash
python3 gen_main_nb.py
```

Regenerating overwrites the notebook, including the outputs of cells that were
already executed.

## License

MIT — see [LICENSE](LICENSE).
