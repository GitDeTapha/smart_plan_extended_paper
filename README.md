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

No-show prediction, test set, each model at its optimal threshold:

| Model | AUC | Accuracy | Precision | Recall | F1-score | train-test AUC gap |
|---|---|---|---|---|---|---|
| XGBoost | **0.77** | 0.94 | 0.58 | 0.59 | **0.59** | 0.00 |
| RandomForest | 0.76 | 0.95 | 0.58 | 0.58 | 0.58 | 0.02 |
| AdaBoost | 0.69 | 0.89 | 0.53 | 0.58 | 0.53 | 0.00 |
| ANN | 0.63 | 0.94 | 0.54 | 0.54 | 0.54 | 0.03 |

Brier score of the calibrated XGBoost: 0.0288. The A/B service-time classifier
ranks the same way (XGBoost 0.73). Clustering: K=2, bootstrapped silhouette
0.6917 ± 0.0035, cutoff 38.5 min. The ILP instance built from 30 held-out
patients over 6 slots solves in 2.9 s (Z_complete 39.9288).

The near-zero train-test gaps matter as much as the scores. An earlier run
reported 0.99 cross-validated ROC-AUC against 0.75 on the test set, because
SMOTE was applied to the tuning subset before the cross-validation: the search
then selected whichever model best memorised interpolated data, and the forest
it picked scored a perfect 1.00 on its own training set. Resampling now happens
inside each fold only.

- `run_20260926_2002.ipynb` — the executed notebook, figures included
- `recap_data_driven_plan.csv`, `training_history.csv` — summary tables

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
