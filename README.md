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

- Notebook 01 is executed on all the data.
- Notebooks 02 to 04 have been executed on a **pilot of 10 % of the
  consultations** ([pilot_10pct/](pilot_10pct/)). The run on all the data is
  under way; its executed notebooks will replace the ones in `notebooks/`.

### Why a 10 % pilot

The four classifiers of notebooks 02 and 03 are tuned by a grid search with
3-fold cross-validation. On 10 % of the consultations (1.1 million), this took
2 h 40 for notebook 02 and 2 h 20 for notebook 03 on a laptop. All the data
(11.3 million consultations) is ten times larger: the full run is estimated at
40 to 60 hours on the laptop, so it runs on a secure rented server.

The pilot checks, before that expensive run, that the whole chain works end to
end, and gives a first view of the results. The pilot notebooks are the
notebooks of `notebooks/` with two changes: `SAMPLE_FRACTION = 0.1` in
notebook 02, and the paths of their files.

### Pilot results (10 % of the consultations)

Test set (30 % of the sample), metrics averaged over the two classes
(`average='macro'`, as in the conference paper). The threshold of each model
maximises the F1 score on a validation set taken from the training set. The
training metrics are in the notebooks.

**No-show** ([02_no_show_prediction_10pct](pilot_10pct/02_no_show_prediction_10pct.ipynb)), no-show rate 3.1 %:

| Model | F1 | Precision | Recall | AUC | Accuracy |
|---|---|---|---|---|---|
| XGBoost | 0.662 | 0.679 | 0.649 | 0.845 | 0.963 |
| Random Forest | 0.630 | 0.634 | 0.627 | 0.832 | 0.957 |
| MLP | 0.628 | 0.627 | 0.629 | 0.814 | 0.955 |
| AdaBoost | 0.598 | 0.589 | 0.610 | 0.795 | 0.946 |

XGBoost is kept (best F1 on the validation set). Its probabilities, calibrated
by Platt scaling on the validation set, give a mean of 3.09 % on the test set,
for 3.11 % observed.

**Service-time group** ([03_service_time_groups_10pct](pilot_10pct/03_service_time_groups_10pct.ipynb)):
two groups by exact 1D K-medians on the training set. The cutoff is 22 days
(medians: 7 days for the short group, 38 days for the long group); 35 % of the
consultations are long.

| Model | F1 | Precision | Recall | AUC | Accuracy |
|---|---|---|---|---|---|
| XGBoost | 0.636 | 0.635 | 0.638 | 0.707 | 0.666 |
| Random Forest | 0.632 | 0.630 | 0.636 | 0.700 | 0.657 |
| MLP | 0.612 | 0.610 | 0.615 | 0.674 | 0.640 |
| AdaBoost | 0.561 | 0.601 | 0.567 | 0.630 | 0.658 |

AdaBoost predicts the short group for almost every consultation (cross-validated
F1 0.395); it is shown as it is.

**Scheduling** ([04_scheduling_ilp_10pct](pilot_10pct/04_scheduling_ilp_10pct.ipynb)):
one day of 8 slots of 30 minutes, 12 patients, short and long consultations of
15 and 30 minutes, c_I = 1, c_W = 0.5.

- Table 1 of the conference paper is reproduced (Z* = 65.62). The ILP is
  checked against the enumeration of every planning.
- Every ILP is proven optimal by HiGHS, in about 0.03 s.
- In this setting, the ILP and the greedy policy have the **same expected cost**
  on all 30 days. With identical slots, two lengths and show probabilities close
  to 0.97, there is almost nothing to optimise.
- On 100 real days of the test set, the difference between the ILP and the
  greedy policy is not significant (95 % CI -9.3 to 2.7). Single booking costs
  less, but serves 8 patients out of 12.
- The setting of the full run (slots of different lengths, or a lighter day) is
  being discussed.

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
python3 -m venv ml_env && ./ml_env/bin/pip install -r requirements.txt
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
