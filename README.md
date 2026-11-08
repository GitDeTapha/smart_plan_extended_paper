# Smart Plan: Cluster-Predict-Schedule

Data-driven outpatient appointment scheduling with overbooking. Two predictions
made when an appointment is booked (the probability that the patient does not
show up, and the service-time group of the consultation) feed an integer linear
program that puts at most two patients in a slot and minimises the expected cost
of idle time and waiting time.

This project extends the conference paper whose code is at
https://github.com/GitDeTapha/Data_driven_model_in_AOS. Each notebook is
committed executed: its tables and figures are visible on GitHub, nothing needs
to be run to read the results. Every choice is set in the "Parameters" cell at
the top of each notebook, with its reason.

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
references/        stop codes kept (Feyman et al., 2022, JAMA Netw Open, eTable 1)
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
previous ones in `data/`.

## License

MIT, see [LICENSE](LICENSE).
