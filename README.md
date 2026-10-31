# Smart Plan: Cluster-Predict-Schedule

Data-driven outpatient appointment scheduling. Two predictive models (the
no-show probability of each patient and its service-time group) feed a
stochastic overbooking integer program (ILP) that allows at most two patients
per slot, trading slot idle time against patient waiting time.

This repository is the evidence behind the paper: the code, the executed
notebooks of the runs reported in it, their inputs and outputs, and the
provenance of each number. Results are not restated here; they are read in the
executed notebooks listed below.

## Where to find what

| Question | File |
|---|---|
| How were the models trained, and with which outputs? | [reports/run_full_20261030.ipynb](reports/run_full_20261030.ipynb) (executed notebook, all outputs) |
| How long did each training stage take, with how much memory? | [reports/run_full_20261030/STATUS.md](reports/run_full_20261030/STATUS.md) |
| On which machine and package versions? | [pod_env.txt](reports/run_full_20261030/pod_env.txt), [pod_freeze.txt](reports/run_full_20261030/pod_freeze.txt) |
| Test metrics of the trained models | [test_metrics.csv](reports/run_full_20261030/test_metrics.csv), [tables/](reports/run_full_20261030/tables/) |
| Figures of the training run | [reports/run_full_20261030/figures/](reports/run_full_20261030/figures/) |
| Parameters passed from the models to the ILP | [data/inference/plne_parameters.json](data/inference/plne_parameters.json) |
| How was the ILP validated, and with which outputs? | [reports/ilp_full_20261030/ilp_validation_full_20261030.ipynb](reports/ilp_full_20261030/ilp_validation_full_20261030.ipynb) |
| Key ILP figures for the manuscript | [reports/ilp_full_20261030/RESULTS.md](reports/ilp_full_20261030/RESULTS.md) |
| Raw ILP campaign tables, LaTeX tables and figures | [results/](reports/ilp_full_20261030/results/), [tables/](reports/ilp_full_20261030/tables/), [figures/](reports/ilp_full_20261030/figures/) |
| Which code produced each run? | [PROVENANCE.md](reports/run_full_20261030/PROVENANCE.md), [CAMPAIGN.md](reports/ilp_full_20261030/CAMPAIGN.md) |
| Are the unversioned outputs the ones used? | [SHA256SUMS](reports/run_full_20261030/SHA256SUMS) |

The `.ipynb` files open directly in the GitHub web interface (outputs and
figures included). For a large notebook that GitHub does not render, download
it and open it with Jupyter or VS Code.

## How the pieces fit

```
raw consultations (not versioned)
        │  scripts/csv_to_parquet.py
        ▼
data_driven_plan.ipynb  ── executed on a RunPod pod (16 vCPU, 128 GB) ──►  reports/run_full_20261030.ipynb
        │  export stage
        ├──► data/inference/plne_parameters.json   s_A, s_B, pi_A, no-show rate, provenance (versioned)
        └──► holdout_scores.parquet                q_i and groups of holdout patients (patient level,
        │                                          not versioned; SHA-256 in SHA256SUMS)
        ▼
notebooks/01_ilp_validation.ipynb  ── executed on a laptop (4 cores, 16 GB) ──►  reports/ilp_full_20261030/
```

The training needs a server; the ILP campaign runs on a laptop, which shows that
the scheduling step can be deployed on modest hardware. The ILP campaign reads
nothing but the two exported files, so its results follow from the training run
without any recomputation.

## Repository tree

```
.
├── README.md
├── requirements.txt               pinned dependencies (Python 3.13)
├── data_driven_plan.py            all reusable functions: data, features, models, ILP core, evaluators
├── ilp_campaign.py                ILP campaign: instance generator, benchmarks B1-B5, cache, parallel runner
├── gen_main_nb.py                 GENERATOR of data_driven_plan.ipynb (edit this file, not the notebook)
├── data_driven_plan.ipynb         GENERATED template of the training pipeline, without outputs
├── notebooks/
│   ├── gen_nb.py                  GENERATOR of 01_ilp_validation.ipynb
│   └── 01_ilp_validation.ipynb    GENERATED template of the ILP validation, without outputs
├── scripts/csv_to_parquet.py      converts the raw CSV into chunked Parquet input
├── data/inference/
│   └── plne_parameters.json       the only versioned data file (aggregates, no patient record)
├── tests/                         pytest suite (exactness of the ILP, evaluators, training, campaign)
└── reports/                       EXECUTED runs: the evidence
    ├── run_full_20261030.ipynb        training run on all patients (pod), with outputs
    ├── run_full_20261030/             its status, environment, metrics, tables, figures, checksums
    ├── ilp_full_20261030/             ILP campaign on the outputs of that run (laptop)
    ├── commit_map_old_new.tsv         commit correspondence with the original repository
    ├── fixes_20261029.md              changes made after the audit of the pipeline
    └── run_20260926_2002.ipynb,       earlier run, before the audit (kept for the record,
        recap_data_driven_plan.csv,    superseded by run_full_20261030)
        training_history.csv
```

### Generated templates and executed runs

`data_driven_plan.ipynb` and `notebooks/01_ilp_validation.ipynb` are written by
`gen_main_nb.py` and `notebooks/gen_nb.py`. They are kept without outputs: they
are the programs. Each run executes a copy with papermill, which records the
parameters of the run in a cell tagged `injected-parameters`; that executed copy
is the one stored in `reports/`. To change a notebook, edit its generator and
regenerate (`python3 gen_main_nb.py`, or `cd notebooks && python3 gen_nb.py`).

## Data

The dataset is not part of this repository: it contains patient-level
consultation records. `.gitignore` is an allow-list, so `data/`, `models/`,
`results/`, `figures/`, `tables/` and `runs/` stay local unless a file is
explicitly allowed. All data paths derive from `SMART_PLAN_DATA` (default
`./data`).

## Reproducing the runs

Setup:

```bash
python3 -m venv ml_env
./ml_env/bin/pip install -r requirements.txt
```

Convert the raw CSV once:

```bash
./ml_env/bin/python scripts/csv_to_parquet.py \
  --csv data/consult_waits_2024_03_25.csv \
  --out-dir data/raw_data_parquet --chunksize 500000 --compression snappy
```

Training (the full run took 25.5 h on 16 vCPU and 128 GB; stages resume from
their saved artifacts after an interruption):

```bash
python -m papermill data_driven_plan.ipynb runs/full_executed.ipynb -k python3 \
  -p RUN_ID full_<date> -p MMAP_MATRICES False
```

`-p PATIENT_FRACTION 0.1` runs a pilot on 10 % of the patients; `-p SMOKE True`
a quick check. On a local disk `MMAP_MATRICES` can stay `True`.

ILP validation, from the repository root, on the outputs of a training run:

```bash
python -m papermill notebooks/01_ilp_validation.ipynb runs/ilp_executed.ipynb --cwd . -k python3 \
  -p PLNE_PARAMS_PATH data/inference/plne_parameters.json \
  -p HOLDOUT_SCORES_PATH results/run_<id>/holdout_scores.parquet
```

Tests:

```bash
python -m pytest tests
```

## License

MIT, see [LICENSE](LICENSE).
