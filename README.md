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
```

The dataset is **not** part of this repository: it contains patient-level
consultation records. `data/`, `data_parquet_chunks/` and every generated
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
  --out-dir data_parquet_chunks --chunksize 500000 --compression snappy

./ml_env/bin/jupyter lab data_driven_plan.ipynb
```

Section 3.2bis and sections 4a/4b each provide a resume point: once the
preprocessed Parquet file and the model bundles exist, the expensive cells can
be skipped on a later run.

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
