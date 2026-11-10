# Smart Plan: Cluster-Predict-Schedule

Data-driven outpatient appointment scheduling with overbooking. Two predictions
made when an appointment is booked (the probability that the patient does not
show up, and the service-time group of the consultation) feed an integer linear
program that puts at most two patients in a slot and minimises the expected cost
of idle time and waiting time.

This project extends the conference paper whose code is at
https://github.com/GitDeTapha/Data_driven_model_in_AOS. Every choice is set in
the "Parameters" cell at the top of each notebook, with its reason. The notebooks
are committed executed: their tables and figures can be read on GitHub without
running anything.

## Status

All four notebooks in [notebooks/](notebooks/) are executed on **all the data**
(11.3 million consultations), with their outputs. Notebook 01 ran on a laptop;
notebooks 02 to 04 ran on a secure rented server (16 cores, 128 GB of RAM), 02
and 03 in parallel: 17 h for notebook 02, 19 h for notebook 03 (of which 16 h
for the XGBoost grid search), 30 s for notebook 04.

An earlier pilot on 10 % of the consultations is kept in
[pilot_10pct/](pilot_10pct/) (see [Pilot](#pilot-on-10--of-the-consultations)).

## Results on all the data

Test set: 3,402,784 consultations (30 %, stratified). Metrics averaged over the
two classes (`average='macro'`, as in the conference paper). The threshold of
each model maximises the F1 score on a validation set taken from the training
set; the model kept is the best one on that validation set.

### No-show: [notebooks/02_no_show_prediction.ipynb](notebooks/02_no_show_prediction.ipynb)

No-show rate 3.1 %.

| Model | Test F1 | Precision | Recall | AUC | Accuracy | Train F1 | Threshold |
|---|---|---|---|---|---|---|---|
| **XGBoost** (kept) | **0.690** | 0.721 | 0.667 | 0.869 | 0.967 | 0.780 | 0.51 |
| Random Forest | 0.653 | 0.661 | 0.646 | 0.836 | 0.960 | 0.680 | 0.63 |
| MLP | 0.630 | 0.630 | 0.630 | 0.809 | 0.956 | 0.630 | 0.57 |
| AdaBoost | 0.607 | 0.597 | 0.620 | 0.799 | 0.947 | 0.606 | 0.49 |

The probabilities of XGBoost, calibrated by Platt scaling on the validation set,
give a mean of 3.11 % on the test set, for 3.11 % observed.

### Service-time group: [notebooks/03_service_time_groups.ipynb](notebooks/03_service_time_groups.ipynb)

Two groups by exact 1D K-medians of the service time on the training set: cutoff
22 days (medians 7 and 38 days), 35 % of the consultations are long.

| Model | Test F1 | Precision | Recall | AUC | Accuracy | Train F1 | Threshold |
|---|---|---|---|---|---|---|---|
| **XGBoost** (kept) | **0.667** | 0.666 | 0.668 | 0.746 | 0.697 | 0.857 | 0.51 |
| Random Forest | 0.650 | 0.648 | 0.653 | 0.724 | 0.678 | 0.692 | 0.54 |
| MLP | 0.617 | 0.615 | 0.619 | 0.681 | 0.647 | 0.617 | 0.40 |
| AdaBoost | 0.575 | 0.596 | 0.575 | 0.631 | 0.651 | 0.574 | 0.26 |

XGBoost overfits (train F1 0.857), but is still the best on the test set.
AdaBoost predicts the short group for almost every consultation
(cross-validated F1 0.395); it is shown as it is.

### Scheduling: [notebooks/04_scheduling_ilp.ipynb](notebooks/04_scheduling_ilp.ipynb)

One day of 4 slots of 15 minutes and 6 slots of 30 minutes, 12 patients, short
and long consultations of 15 and 30 minutes, c_I = 1, c_W = 0.5, overtime
c_O = 1. Double-booked slots are placed at the end of the day, for the ILP and
the greedy policy alike. The no-show probabilities and groups are those
predicted for test consultations.

- Table 1 of the conference paper is reproduced (Z* = 65.62), and checked against
  the enumeration of all 16,384 plannings. Every ILP is proven optimal by HiGHS.
- 30 days, mean cost: ILP 27.13 in the model (43.34 simulated), greedy 27.41
  (44.63), single booking 65.33 (82.96, and 10 patients served out of 12).

| ILP compared with | Mean difference (95 % CI) | Days where the ILP is better | Wilcoxon p |
|---|---|---|---|
| Greedy, model cost | 0.28 (0.12 to 0.44) | 26/30 | < 0.001 |
| Greedy, simulated cost | 1.29 (0.02 to 2.56) | 26/30 | < 0.001 |
| Greedy, 100 real test days | 0.6 (-4.8 to 6.0) | 38/100 | 0.67 |
| Single booking, simulated cost | 39.6 (26.8 to 52.4) | 27/30 | < 0.001 |

The ILP beats the greedy policy by a small but significant margin in the model
and in simulation; on real test days the difference is not significant.
Sensitivity to the load (1 to 2 patients per slot) and to c_W/c_I is in the
notebook.

### Files written by the notebooks (not in this repository)

| File | Written by | Content |
|---|---|---|
| `data/data_prep.parquet` | 01 | consultations kept, 14 predictors, targets |
| `data/split.parquet` | 02 | train / validation / test of each consultation |
| `data/test_scores.parquet` | 02 | calibrated no-show probability of each test consultation |
| `data/test_groups.parquet` | 03 | predicted and true group of each test consultation |
| `models/no_show/*.joblib` | 02 | the four trained models (XGBoost: 0.9 GB) |
| `models/service_time_groups/*.joblib` | 03 | the four trained models (XGBoost: 6 GB) |

They hold patient data or are too large, so they stay outside the repository.

## Pilot on 10 % of the consultations

On a laptop, the grid searches took 2 h 40 (notebook 02) and 2 h 20 (notebook
03) on 10 % of the consultations; all the data was estimated at 40 to 60 hours,
hence the rented server. The pilot checked, before that run, that the whole
chain works end to end. Its notebooks are those of `notebooks/` with
`SAMPLE_FRACTION = 0.1` in notebook 02 and their own file paths:
[02](pilot_10pct/02_no_show_prediction_10pct.ipynb),
[03](pilot_10pct/03_service_time_groups_10pct.ipynb),
[04](pilot_10pct/04_scheduling_ilp_10pct.ipynb) (notebook 04 of the pilot uses an
earlier day of 8 slots of 30 minutes). With all the data, the test F1 of
XGBoost rises from 0.662 to 0.690 (no-show) and from 0.636 to 0.667
(service-time group).

## Notebooks

| Notebook | Question |
|---|---|
| [01_data_preparation](notebooks/01_data_preparation.ipynb) | Which consultations and predictors? (same steps and row counts as the paper) |
| [02_no_show_prediction](notebooks/02_no_show_prediction.ipynb) | How well is a no-show predicted? (Random Forest, XGBoost, AdaBoost, MLP) |
| [03_service_time_groups](notebooks/03_service_time_groups.ipynb) | Short or long consultation? (K-medians clustering, then the same four classifiers) |
| [04_scheduling_ilp](notebooks/04_scheduling_ilp.ipynb) | Which patients share a slot, and what does it save? (ILP, simple policies, real test days) |

## Files

```
utils_prep.py      functions of notebook 01 (taken from the paper's code)
utils_ml.py        functions of notebooks 02 and 03 (pipeline, metrics as in the paper)
utils_ilp.py       functions of notebook 04 (cost of a slot, the ILP, simple policies, simulation)
tests/test_ilp.py  the ILP against the enumeration of every planning
pilot_10pct/       notebooks 02 to 04 executed on 10 % of the consultations
references/        stop codes kept (Feyman et al., 2022, eTable 1)
scripts/           one-off conversion of the raw CSV into Parquet chunks
```

## Data

The consultations of the Veterans Health Administration are patient records and
are not in this repository. Notebook 01 reads them from `data/raw_data_parquet/`,
written once from the raw CSV:

```bash
python scripts/csv_to_parquet.py --csv data/consult_waits_2024_03_25.csv \
  --out-dir data/raw_data_parquet --chunksize 500000 --compression snappy
```

## Running

```bash
python3 -m venv ml_env && ./ml_env/bin/pip install -r requirements.txt   # Python 3.11 or later
cd notebooks
for nb in 01_data_preparation 02_no_show_prediction 03_service_time_groups 04_scheduling_ilp; do
  ../ml_env/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
cd .. && ./ml_env/bin/python -m pytest tests
```

The notebooks run in this order: each one reads the files written by the
previous ones in `data/`. Trained models are saved in `models/` and reloaded
when a notebook is run again (`RETRAIN = True` trains them again).

## References

**Data and earlier work**
- Code of the conference paper: https://github.com/GitDeTapha/Data_driven_model_in_AOS
- Feyman Y, Asfaw DA, Griffith KN. Geographic variation in appointment wait
  times for US military veterans. *JAMA Network Open*. 2022;5(8):e2228783.
  doi:10.1001/jamanetworkopen.2022.28783 (stop codes kept: eTable 1). [To find raw data used]

## License

MIT, see [LICENSE](LICENSE).
