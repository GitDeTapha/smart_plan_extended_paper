"""Generate data_driven_plan.ipynb from programmatic cells.

Mirrors the pattern already used by notebooks/gen_nb.py for
01_ilp_validation.ipynb: the notebook is regenerated reliably and reproducibly
instead of being edited cell by cell. Run from the project root.

Note: regenerating overwrites the notebook, and with it the outputs of any
previously executed cell.
"""
import hashlib
import json
import os
import textwrap

os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/inference", exist_ok=True)

OUT = "data_driven_plan.ipynb"

DEFAULT_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"},
}

cells = []


def _cell_id(src):
    """Deterministic cell id: regenerating an unchanged cell keeps its id, so
    the notebook stays byte-stable in git instead of churning on every run."""
    return hashlib.md5(src.encode("utf-8")).hexdigest()[:8]


def md(src):
    cells.append({"cell_type": "markdown", "id": _cell_id(src), "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "id": _cell_id(src),
                  "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})


def code_stage(stage, src):
    """Code cell run only when `stage` must be computed (resumable run, audit 1.7)."""
    code(f'if RUN.compute("{stage}"):\n' + textwrap.indent(src, "    "))


def params(src):
    """Code cell tagged `parameters`: papermill injects overrides right after it."""
    cells.append({"cell_type": "code", "execution_count": None, "id": _cell_id(src),
                  "metadata": {"tags": ["parameters"]}, "outputs": [],
                  "source": src.splitlines(keepends=True)})


# ═══════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════
md("# data_driven_plan, Cluster-Predict-Schedule pipeline\n\n"
   "Data-driven outpatient appointment scheduling: no-show and service-time "
   "group prediction, feeding a stochastic overbooking ILP "
   "(at most k = 2 patients per slot).\n\n"
   "A single CRISP-DM pipeline. All utility functions live in "
   "`data_driven_plan.py`, this notebook holds only the orchestration, the "
   "narrative and the outputs.\n\n"
   "**Changes since run_20260926_2002 (audit of 27/09):** duration groups by exact "
   "K-medians on the training durations (the cutoff moves from about 38.5 to about "
   "21.5 min), fit / valid / calib split inside the training set, calibration and "
   "thresholds chosen outside the test set, ILP parameters exported with their "
   "provenance. See reports/fixes_*.md.")

code("""\
# ── Configuration and reproducibility ─────────────────────────────────────────
import json, os, sys, platform, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data_driven_plan import *

ensure_output_dirs()

print(f"Python   : {sys.version.split()[0]}")
print(f"NumPy    : {np.__version__}")
print(f"Pandas   : {pd.__version__}")
print(f"Platform : {platform.processor()}")
print(f"ILP solver : {SOLVER_NAME}")
""")

params("""\
# ── Authors' decisions (papermill parameters) ────────────────────────────────
DURATION_METHOD = "kmedians"    # B3: method of the manuscript; "kmeans" for the ablation
DURATION_STAT   = "median"      # s_A, s_B = K-medians centroids
C_O_BASE        = 0.0           # V9: no overtime, as in the manuscript (eq. 1, 8, 9); 0.5 for the extension of section 8.5
SLOT_LENGTHS    = (30, 60)      # B4: s_A <= max(SLOT_LENGTHS) is asserted, the run stops otherwise
SPLIT_MODE      = "stratified"  # "patient_group" for the ablation
RANDOM_STATE    = 0
SMOKE           = False         # True: 1% of the patients, minimal budgets
RUN_ID          = None          # None: timestamp; artifacts in models/run_<RUN_ID>/, results/run_<RUN_ID>/
STAGE           = "all"         # "prep" | "durations" | "tune" | "fit" | "calibrate" | "evaluate" | "export" | "all"
N_JOBS          = -1            # CPUs used (-1: all CPUs allotted to the container)
BLAS_THREADS    = None          # BLAS threads of this process; None: N_JOBS (search workers are set by joblib)
SMOKE_PATIENT_FRACTION = 0.01   # SMOKE: share of patients kept, by hashed identifier
PATIENT_FRACTION = None         # pilot run without SMOKE budgets (e.g. 0.10); None: all patients
MIN_FREE_DISK_GB = None         # None: estimated from the share of patients (about 70 GB for a full run)
""")

code("""\
# ── Decisions check: any value outside its allowed set stops the run ─────────
DECISIONS = validate_decisions(DURATION_METHOD, DURATION_STAT, C_O_BASE, SLOT_LENGTHS,
                               SPLIT_MODE, RANDOM_STATE, SMOKE)
np.random.seed(RANDOM_STATE)
import time as _time
RUN_ID = RUN_ID or _time.strftime("%Y%m%d_%H%M")
RUN_RESULTS_DIR = os.path.join(RESULTS_DIR, f"run_{RUN_ID}")
os.makedirs(RUN_RESULTS_DIR, exist_ok=True)
from threadpoolctl import threadpool_limits
# BLAS only: XGBoost's OpenMP threads are set by n_jobs, a global OpenMP cap would override them
N_JOBS = resolve_n_jobs(N_JOBS)
BLAS_THREADS = N_JOBS if BLAS_THREADS is None else int(BLAS_THREADS)
threadpool_limits(limits=BLAS_THREADS, user_api="blas")
FRACTION = SMOKE_PATIENT_FRACTION if SMOKE else PATIENT_FRACTION
FULL_RUN = FRACTION is None   # only a full run may write the versioned plne_parameters.json
RUN = StageRunner(RUN_ID, target=STAGE)
for k, v in {**DECISIONS, "RUN_ID": RUN_ID, "STAGE": STAGE, "N_JOBS": N_JOBS,
             "BLAS_THREADS": BLAS_THREADS, "PATIENT_FRACTION": FRACTION,
             "available CPUs": available_cpus(), "available memory (GB)": round(available_memory() / 2**30, 1)}.items():
    print(f"{k:<16} {v}")
print(f"Status file: {RUN.status_path}")
RUN.configure("prep", decisions=DECISIONS, data=RAW_PARQUET_DIR,
              patient_fraction=FRACTION)
# Artifacts: about 70 GB for all patients (float32 matrices ~31 GB, models and their calibrated
# copies, SMOTE cache). The container disk of the pod is 5 GB: run from the volume.
_need_gb = MIN_FREE_DISK_GB if MIN_FREE_DISK_GB is not None else 70 * (FRACTION or 1.0) + 2
check_free_disk(RUN.dir, _need_gb)
""")

# ═══════════════════════════════════════════════════════════════════════════
# 1. Business Understanding
# ═══════════════════════════════════════════════════════════════════════════
md("## 1. Business Understanding\n\n"
   "**Decision objective:** reduce slot idle time and patient waiting time in "
   "an outpatient clinic by allowing controlled overbooking (at most two "
   "patients per slot), which compensates for no-shows without degrading the "
   "patient experience excessively.\n\n"
   "For each patient to be scheduled, the stochastic ILP (section 7) needs two "
   "predicted quantities:\n"
   "- **q_i**: calibrated show-up probability (1 − P(no-show));\n"
   "- **group**: service-time class (A = long, B = short), which sets the "
   "  deterministic duration `s_i` used in the model.\n\n"
   "Sections 2 to 6 build and validate these two predictive models; section 7 "
   "uses them to solve real ILP instances.")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Data Understanding
# ═══════════════════════════════════════════════════════════════════════════
md("## 2. Data Understanding\n\n"
   "### 2.1 Initial collection\n\n"
   "The raw data (`$SMART_PLAN_DATA/raw_data_parquet/`, derived from "
   "`data/consult_waits_2024_03_25.csv` by `scripts/csv_to_parquet.py`) covers "
   "outpatient consultations with their scheduled and realised times "
   "(`dts`/`dtc`), the attendance status (`disp`) and administrative metadata "
   "(facility, clinic stop code, region).")

code_stage("prep", """\
# ── 2.1 Load the raw data ─────────────────────────────────────────────────────
# Paths come from the module and follow SMART_PLAN_DATA (default: ./data), so
# this notebook runs unchanged wherever the dataset lives.
print(f"Data root: {DATA_DIR}")
data = load_raw_data(RAW_PARQUET_DIR, patient_fraction=FRACTION,
                     key=f"smoke{RANDOM_STATE}")
print(f"Raw shape: {data.shape}")
data.info(memory_usage="deep")
data.head()
""")

md("### 2.2 Target description and class imbalance\n\n"
   "`disp` is decomposed immediately into binary indicators "
   "(`NoShow`/`discontinued`) so that class imbalance can be quantified from "
   "the exploration stage onwards.\n\n"
   "**The `completed` indicator is deliberately not created:** together with "
   "`discontinued` it reconstructs the target "
   "(NoShow ≈ 1 − completed − discontinued) and biases no-show prediction as "
   "soon as it reaches the features. Never producing it removes the risk at "
   "the source rather than relying on a downstream exclusion. Nothing is lost: "
   "`NoShow` already encodes CANCELLED and `discontinued` the third state, so "
   "the COMPLETE/UPDATE category follows from the two.")

code_stage("prep", """\
# ── 2.2 Target decomposition + class imbalance ───────────────────────────────
data = decompose_disp_column(data, disp_column="disp")
plot_target_balance(data, "NoShow", labels={0: "Attended", 1: "No-show"}, prefix="eda")
print(data["NoShow"].value_counts(normalize=True).rename("proportion"))
""")

md("The observed no-show rate (~3%) is far more imbalanced than the initial "
   "85/15 assumption, which drives the choice of imbalance strategy (SMOTE + "
   "class weighting + optimal threshold, section 4a) and makes calibration "
   "(section 5) more important than discrimination alone.")

md("### 2.3 Exploring the numeric and temporal variables\n\n"
   "`dta` (waiting time before the appointment) and `dts`/`dtc` (service "
   "start/end) are explored before cleaning, their raw values contain "
   "inconsistencies (dts > dtc) handled in Data Preparation.")

code_stage("prep", """\
# ── 2.3 Raw numeric distributions ─────────────────────────────────────────────
plot_numeric_histograms(data, ["dta", "dts", "dtc"], bins=[50, 50, 50],
                        xlims=[(0, 100), (0, 250), (0, 500)], prefix="eda")
""")

code_stage("prep", """\
# ── 2.3 (cont.) Missing values ────────────────────────────────────────────────
missing_pct = display_missing_values(data, sentinel_col="state", sentinel_value="*Missing*")
print(missing_pct[missing_pct > 0].sort_values(ascending=False))
""")

md("`AdministrativeFlag` exceeds 55% missing values: reliable imputation is "
   "illusory at that rate, so the column is dropped rather than imputed "
   "(section 3). `zip`/`state` are missing for a marginal share of rows, "
   "listwise deletion is preferred to imputation, with no material impact on a "
   "dataset of this size.")

code_stage("prep", """\
# ── 2.3 (cont.) Outlier diagnostic (IQR), visualisation only ────────────────
visualize_outliers(data, prefix="eda")
""")

code_stage("prep", """\
# ── 2.3 (cont.) Activity patterns (weekday / hour) ───────────────────────────
_ = analyze_temporal_patterns(data, "activitydatetime", prefix="eda")
""")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Data Preparation
# ═══════════════════════════════════════════════════════════════════════════
md("## 3. Data Preparation\n\n"
   "### 3.1 Cleaning\n\n"
   "**Temporal filtering:** data prior to 2016-08-30 is excluded, it predates "
   "a major operational reorganisation (launch of the Veteran Choice Program) "
   "that durably changed waiting times and therefore no-show rates; including "
   "it would bias learning towards an obsolete operating regime.")

code_stage("prep", """\
# ── 3.1 Temporal filtering + calendar features ────────────────────────────────
data = filter_dates_after(data, "activitydatetime", "2016-08-30 00:00:00")
data = add_temporal_features(data, "activitydatetime")
print(f"Shape after temporal filtering: {data.shape}")
""")

code_stage("prep", """\
# ── 3.1 Missing values: column and row removal ───────────────────────────────
data = drop_high_missing_columns(data, ["AdministrativeFlag"])
data = drop_missing_rows(data, ["zip", "state"], missing_sentinels={"state": "*Missing*"})
print(f"Shape after missing-value handling: {data.shape}")
""")

md("**dts/dtc consistency:** a record whose service end time precedes its "
   "start time (`dtc < dts`) reflects a data-entry error, `service_time` "
   "would not be interpretable there. Those rows are removed, including the "
   "rare `NoShow=1` cases with `dts>dtc` (the patient's absence makes `dtc` "
   "meaningless).")

code_stage("prep", """\
# ── 3.1 dts/dtc consistency + duplicates + service_time ──────────────────────
data = filter_dts_dtc_consistency(data, "dts", "dtc", "NoShow")
data = data.drop_duplicates()
data = compute_service_time(data, "dts", "dtc")
print(f"Shape after dts/dtc consistency + deduplication: {data.shape}")
""")

md("### 3.2 Transformation\n\n"
   "**Stop-code filtering:** only clinic codes that have recorded at least one "
   "no-show are kept, a stop code with no no-show history carries no "
   "discriminating signal for this task and dilutes the training sample.")

code_stage("prep", """\
# ── 3.2 Filter out stop codes with no no-show history ────────────────────────
data, _data_without_ns, _codes_dropped = separate_stopcodes_by_no_show(data)
print(f"Stop codes excluded (never any no-show): {len(_codes_dropped)}")
print(f"Shape after stop-code filtering: {data.shape}")
""")

md("**Patient history features:** days since the previous visit (per "
   "patient/stop code) and running averages of prior no-shows and prior "
   "service times (per patient), lagged by one rank to exclude any information "
   "leaking from the current visit.")

code_stage("prep", """\
# ── 3.2 Patient history features ──────────────────────────────────────────────
data = add_patient_history_features(data, "patientsid", "stopcode", "activitydatetime",
                                    "NoShow", "service_time")
print(data[["days_between_visits", "prior_noShow_rate", "prior_service_time_avg"]].describe())
""")

code_stage("prep", """\
# ── 3.2 Final feature selection + checkpoint ─────────────────────────────────
# patientsid is kept until the split (SPLIT_MODE="patient_group" needs it) and
# excluded from the features by prepare_features.
data = data.drop(columns=["activitydatetime"])
print(f"Final preprocessed shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
save_preprocessed_data(data, PREPROCESSED_PATH)
print(f"Saved -> {PREPROCESSED_PATH}")
""")

md("### 3.3 Single partition\n\n"
   "One split of the full data: train 70 % / test 20 % / ILP holdout 10 %, stratified "
   "jointly on `NoShow` and on `service_time` deciles (same seeds as run_20260926_2002, "
   "so test and holdout are unchanged). The training part is then split into **fit** "
   "(80 %), **valid** (10 %) and **calib** (10 %), stratified the same way (audit rec. 13):\n\n"
   "- fit: fitting and hyperparameter search;\n"
   "- valid: model choice inside a family (successive halving, boosting iterations);\n"
   "- calib: probability calibration, choice of the calibration method, of the model and of "
   "the decision thresholds;\n"
   "- test: final metrics only, nothing is fitted or chosen on it;\n"
   "- holdout: never seen by the models, it feeds the ILP.\n\n"
   "`SPLIT_MODE = \"patient_group\"` (ablation) keeps every visit of a patient in the same "
   "partition, from a hash of `patientsid`.")

code_stage("prep", """\
# ── 3.3 Single split (fit / valid / calib / test / ILP holdout) ──────────────
SPLIT = make_master_split(
    data, train_size=0.70, test_size=0.20, inference_size=0.10, inner_sizes=(0.80, 0.10, 0.10),
    split_mode=SPLIT_MODE, noshow_col="NoShow", service_time_col="service_time",
    n_quantile_bins=10, random_state=RANDOM_STATE,
)
df_sample, train_df, test_df, df_inference = SPLIT.data, SPLIT.train, SPLIT.test, SPLIT.inference
""")

md("**Leakage warning:** `discontinued` (section 2.2) is derived from the same "
   "source column `disp` as `NoShow`, and is not known before the appointment "
   "in any case, it is therefore excluded from the features, like the targets "
   "themselves. `completed`, which made the target reconstructible "
   "(NoShow ≈ 1 − completed − discontinued), is no longer produced at all "
   "since section 2.2; it is nevertheless still listed below as a safeguard, "
   "in case the fast resume path (section 3.2bis) reloads a Parquet file "
   "preprocessed by an earlier run.\n\n"
   "**Real-world availability warning:** `dts` (actual service start) and "
   "`dta` (waiting time before the appointment) are known only once the visit "
   "is under way or finished, at the moment the prediction must actually be "
   "used (before the appointment, to decide on overbooking), they do not yet "
   "exist. `dtc` is already dropped after `service_time` is computed "
   "(section 3.1); `dts`/`dta` are therefore excluded from the features too, "
   "even though they remain in `data` for now.")

code_stage("prep", """\
# ── 3.3 Encoding + standardisation (fitted on train only) ────────────────────
X_train, X_test, X_inference, feature_cols, scaler, cat_encodings = prepare_features(
    train_df, test_df, inference_df=df_inference,
    categorical_cols=["stopcode", "sta3n", "non_va", "state", "zip"],
    onehot_max_cardinality=150,   # one-hot below this, integer code above
    excluded_cols={"service_time", "NoShow", "group_label", "cluster",
                   "discontinued", "dts", "dta", "patientsid",
                   "completed"},  # safeguard: no longer produced, except in an older Parquet
    random_state=RANDOM_STATE,
)
# Views of the training matrix on the three inner partitions (same rows, no copy of logic)
X_fit, X_valid, X_calib = (X_train.loc[SPLIT.fit.index], X_train.loc[SPLIT.valid.index],
                           X_train.loc[SPLIT.calib.index])
print(f"X_fit {X_fit.shape} | X_valid {X_valid.shape} | X_calib {X_calib.shape}")
""")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Modeling
# ═══════════════════════════════════════════════════════════════════════════
code_stage("prep", """\
# ── 3.4 Contiguous float32 matrices, cached once (fit | valid | calib rows) ──
_XA = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32))
_XT = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))
_XI = np.ascontiguousarray(X_inference.to_numpy(dtype=np.float32))
del X_train, X_test, X_inference, X_fit, X_valid, X_calib, data
RUN.finish("prep", XA=_XA, X_test=_XT, X_inf=_XI,
           y_ns=train_df["NoShow"].to_numpy().astype(int),
           y_test_ns=test_df["NoShow"].to_numpy().astype(int),
           y_inf_ns=df_inference["NoShow"].to_numpy().astype(int),
           st_train=train_df["service_time"].to_numpy(), st_test=test_df["service_time"].to_numpy(),
           st_inf=df_inference["service_time"].to_numpy(),
           N_FIT=len(SPLIT.fit), N_VALID=len(SPLIT.valid), split_sizes=SPLIT.sizes(), split_mode=SPLIT.mode,
           feature_cols=feature_cols, scaler=scaler, cat_encodings=cat_encodings)
del _XA, _XT, _XI
""")

code("""\\
# ── 3.5 Prep artifacts (memory-mapped matrices) ──────────────────────────────
if RUN.wanted("prep"):
    PREP = RUN.load("prep")
    XA, X_test_np, X_inf_np = PREP["XA"], PREP["X_test"], PREP["X_inf"]
    y_ns, y_test_ns, y_inf_ns = PREP["y_ns"], PREP["y_test_ns"], PREP["y_inf_ns"]
    N_FIT, N_VALID = PREP["N_FIT"], PREP["N_VALID"]
    N_FV = N_FIT + N_VALID
    feature_cols = PREP["feature_cols"]
    display(PREP["split_sizes"])
    print(f"XA {XA.shape} | test {X_test_np.shape} | holdout {X_inf_np.shape} | split {PREP['split_mode']}")
""")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Modeling
# ═══════════════════════════════════════════════════════════════════════════
md("## 4. Modeling\n\n"
   "Two sub-pipelines on the SAME split and features: no-show (4a) and service-time "
   "group (4b). Four families, as in the published study: RandomForest, XGBoost, "
   "AdaBoost, MLP (scikit-learn, xgboost), with the encoding of run_20260926_2002.\n\n"
   "**Protocol (audit 1.3).** Hyperparameters by successive halving "
   "(`HalvingRandomSearchCV`) on the predefined fit / valid split: candidates are trained "
   "on fit and scored on valid, with training rows growing from 500,000 to all of fit. "
   "Criterion: log-loss on valid of the probabilities corrected back to the true base rate "
   "(the models are trained on balanced classes). AdaBoost: one fit per learning rate at "
   "the maximal number of estimators, iterations chosen on valid by `staged_predict_proba`, "
   "in a separate process running in parallel. Final model refitted on fit + valid.\n\n"
   "**Imbalance (audit 1.4, as in the first paper: SMOTE with class weighting).** Every "
   "family is a pipeline applying SMOTE at fit time only, up to a minority / majority "
   "ratio of 0.1, then weights balancing the resampled classes (`class_weight` for the "
   "forest, `scale_pos_weight` for XGBoost, `sample_weight` for AdaBoost and the MLP).")

code("""\\
# ── 4.0 Training settings ───────────────────────────────────────────────────
TRAIN_CFG_NOSHOW = TrainingConfig(
    models=["RandomForest", "XGBoost", "AdaBoost", "ANN"],
    search="halving",
    halving_iterations=3,         # successive halving: 3 iterations ...
    last_candidates=2,            # ... ending with 2 finalists compared on all of fit
    min_resources=500_000,        # fit rows of the first iteration (audit 1.3)
    max_resources=None,           # None: all of fit at the last iteration
    memory_safety=0.6,            # search processes capped to 60 % of the available memory
    smote_sampling_strategy=0.1, smote_k_neighbors=5,
    adaboost_max_estimators=400, adaboost_learning_rates=(0.5, 1.0),
    threshold_metric="youden", n_jobs=N_JOBS, verbose=1, random_state=RANDOM_STATE,
    smote_cache_dir=os.path.join(RUN.dir, "smote_cache"),   # memoised SMOTE (identical results)
)
if SMOKE:   # minimal budgets: the smoke run checks that the code runs, not the results
    TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(
        min_resources=20_000, adaboost_max_estimators=40,
        param_grids={"RandomForest": {"n_estimators": [100], "max_depth": [8, 12]},
                     "XGBoost": {"n_estimators": [100]}},
        base_params={"ANN": {"max_iter": 20}})
TRAIN_CFG_GROUP = TRAIN_CFG_NOSHOW.replace()
display(TRAIN_CFG_NOSHOW.summary())
""")

md("### 4b (first part). Duration groups\n\n"
   "**Exact K-medians, K = 2, on the training durations only** (audit B3). Service times "
   "are integer minutes, so the optimal 1-D K-medians partition is found exactly by "
   "dynamic programming over the distinct values (no initialisation, no randomness). "
   "s_A, s_B are the cluster medians. Groups and cutoff are fitted on train and applied "
   "unchanged to test and holdout. K = 2 is imposed by the ILP (two duration classes); "
   "cost and silhouette for K = 2 to 5 are reported. `DURATION_METHOD = \"kmeans\"` runs "
   "the ablation with the former method.")

code("""\\
# ── 4b. Duration-group settings ─────────────────────────────────────────────
DURATION_KS = [2, 3, 4, 5]    # values of K reported (cost + silhouette)
SIL_SAMPLE = 5_000 if SMOKE else 20_000   # durations used for each silhouette
N_BOOTSTRAP = 10              # bootstrap resamples of the silhouette (local generator)
BOOTSTRAP_SAMPLE = 10_000     # size of each resample
RUN.configure("durations", method=DURATION_METHOD, stat=DURATION_STAT, ks=DURATION_KS,
              sil=(SIL_SAMPLE, N_BOOTSTRAP, BOOTSTRAP_SAMPLE))
""")

code_stage("durations", """\\
# ── 4b. Fit the groups on the training durations, apply them everywhere ─────
_st = np.asarray(PREP["st_train"], dtype=float)
_k_table = evaluate_duration_ks(_st, DURATION_KS, method=DURATION_METHOD, sil_sample=SIL_SAMPLE,
                                random_state=RANDOM_STATE, prefix="servicetime")
_groups = fit_duration_groups(_st, k=2, method=DURATION_METHOD, stat=DURATION_STAT,
                              random_state=RANDOM_STATE)
_sil = bootstrap_silhouette(_st, _groups, n_bootstrap=N_BOOTSTRAP, sample_size=BOOTSTRAP_SAMPLE,
                            random_state=RANDOM_STATE)
_tree, _cut = learn_decision_tree_cutoffs(pd.DataFrame({"service_time": _st, "group_label": _groups.labels(_st)}),
                                          service_time_col="service_time", group_col="group_label",
                                          label_names=["A", "B"])
RUN.finish("durations", groups=_groups, k_table=_k_table, sil=_sil, tree=_tree, tree_cutoff=_cut)
""")

code("""\\
if RUN.wanted("durations"):
    DUR = RUN.load("durations")
    DURATION_GROUPS, cluster_tree = DUR["groups"], DUR["tree"]
    mu_sil, sd_sil = DUR["sil"]
    service_time_by_group = DURATION_GROUPS.s_by_group
    display(DUR["k_table"])
    latex_table(DUR["k_table"], os.path.join(TABLES_DIR, "table_duration_k.tex"),
                caption="Duration groups: cost and silhouette by K (training durations).",
                label="tab:duration-k", decimals=4)
    y_grp = (DURATION_GROUPS.labels(PREP["st_train"]) == "A").astype(int)
    y_test_grp = (DURATION_GROUPS.labels(PREP["st_test"]) == "A").astype(int)
    grp_inf_actual = DURATION_GROUPS.labels(PREP["st_inf"])
    print(f"{DURATION_GROUPS.method}/{DURATION_GROUPS.stat} on {DURATION_GROUPS.n_train:,} training durations: "
          f"s_A = {service_time_by_group['A']:g} min, s_B = {service_time_by_group['B']:g} min, "
          f"cutoff {DURATION_GROUPS.cutoff:g} min (tree root {DUR['tree_cutoff']:g}), "
          f"training share of A {DURATION_GROUPS.shares['A']:.4f}")
    print(f"Bootstrapped silhouette (K=2): mean={mu_sil:.4f}, sd={sd_sil:.4f}")
    assert abs(DUR["tree_cutoff"] - DURATION_GROUPS.cutoff) <= 0.5 + 1e-9
    # Audit B4: a long patient must fit in the longest slot, otherwise stop here
    if service_time_by_group["A"] > max(SLOT_LENGTHS):
        raise ValueError(f"s_A = {service_time_by_group['A']:g} min exceeds the longest slot "
                         f"{max(SLOT_LENGTHS)} min: decision required (SLOT_LENGTHS or DURATION_METHOD)")
""")

md("### 4a, 4b. Hyperparameter search on fit / valid")

code("""\\
RUN.configure("tune", noshow=TRAIN_CFG_NOSHOW.as_dict(), group=TRAIN_CFG_GROUP.as_dict())
""")

code_stage("tune", """\\
# ── 4a / 4b. Tune both tasks (AdaBoost in a separate process) ────────────────
_xa_path = os.path.join(RUN.dir, "prep", "XA.npy")
_y_ns_path, _y_grp_path = RUN.path("tune", "y_ns.npy"), RUN.path("tune", "y_grp.npy")
np.save(_y_ns_path, np.asarray(y_ns)); np.save(_y_grp_path, np.asarray(y_grp))
tuned_ns = tune_binary_classifiers(XA[:N_FV], y_ns[:N_FV], N_FIT, TRAIN_CFG_NOSHOW,
                                   adaboost_files=(_xa_path, _y_ns_path))
tuned_grp = tune_binary_classifiers(XA[:N_FV], y_grp[:N_FV], N_FIT, TRAIN_CFG_GROUP,
                                    adaboost_files=(_xa_path, _y_grp_path))
RUN.finish("tune", tuned_ns=tuned_ns, tuned_grp=tuned_grp)
""")

code("""\\
if RUN.wanted("tune"):
    TUNE = RUN.load("tune")
    tuned_ns, tuned_grp = TUNE["tuned_ns"], TUNE["tuned_grp"]
    for _task, _t in (("no-show", tuned_ns), ("group", tuned_grp)):
        print(f"{_task}: " + "; ".join(f"{m} {b} (log-loss valid {_t['criterion'][m]:.5f})"
                                       for m, b in _t["best"].items()))
    RUN.configure("fit")
""")

code_stage("fit", """\\
# ── 4a / 4b. Refit on fit + valid ───────────────────────────────────────────
noshow_models = fit_binary_classifiers(XA[:N_FV], y_ns[:N_FV], tuned_ns, TRAIN_CFG_NOSHOW)
group_models = fit_binary_classifiers(XA[:N_FV], y_grp[:N_FV], tuned_grp, TRAIN_CFG_GROUP)
RUN.finish("fit", noshow_models=noshow_models, group_models=group_models)
""")

code("""\\
if RUN.wanted("fit"):
    FIT = RUN.load("fit")
    noshow_models, group_models = FIT["noshow_models"], FIT["group_models"]
    display(noshow_models.search_report)
    display(group_models.search_report)
    RUN.configure("calibrate", methods=("sigmoid", "isotonic"))
""")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluation
# ═══════════════════════════════════════════════════════════════════════════
md("## 5. Calibration, model choice and evaluation\n\n"
   "**On calib** (never on test): each family is calibrated (analytic prior correction, "
   "Platt scaling or isotonic regression, fitted on one half of calib); on the other half "
   "the calibrated Brier score chooses the method of each family, then the model; Youden's "
   "threshold is read there, and for the group classifier the threshold that makes the "
   "predicted share of A equal to the observed one (rec. 12). **On test**: AUC-ROC, "
   "AUC-PR, log-loss, Brier, Brier skill score against the training base rate, then at "
   "the calib threshold accuracy (with the majority-class accuracy for reference) and "
   "precision, recall, F1 in macro average and for the positive class (audit P5).")

code_stage("calibrate", """\\
# ── 5.1 Calibration and model choice on calib ───────────────────────────────
cal_ns = calibrate_and_select(noshow_models, XA[N_FV:], y_ns[N_FV:], random_state=RANDOM_STATE)
cal_grp = calibrate_and_select(group_models, XA[N_FV:], y_grp[N_FV:], random_state=RANDOM_STATE)
_thr = share_matching_threshold(cal_grp.p_calib_eval[cal_grp.best_name],
                                target_share=float(cal_grp.y_calib_eval.mean()))
RUN.finish("calibrate", cal_ns=cal_ns, cal_grp=cal_grp, group_threshold=_thr)
""")

code("""\\
if RUN.wanted("calibrate"):
    CAL = RUN.load("calibrate")
    cal_ns, cal_grp, GROUP_THRESHOLD = CAL["cal_ns"], CAL["cal_grp"], CAL["group_threshold"]
    best_noshow_name, best_group_name = cal_ns.best_name, cal_grp.best_name
    calibrated_noshow_model, best_group_model = cal_ns.models[best_noshow_name], cal_grp.models[best_group_name]
    display(cal_ns.table.round(5)); display(cal_grp.table.round(5))
    latex_table(pd.concat([cal_ns.table.assign(Task="no-show"), cal_grp.table.assign(Task="group A")]),
                os.path.join(TABLES_DIR, "table_calibration.tex"),
                caption="Calibration methods on calib (half B): Brier score and log-loss.",
                label="tab:calibration", decimals=5)
    print(f"No-show model: {best_noshow_name} ({cal_ns.methods[best_noshow_name]}) | "
          f"group model: {best_group_name} ({cal_grp.methods[best_group_name]})")
    print(f"Group threshold on calibrated p_A: {GROUP_THRESHOLD:.4f} (Youden {cal_grp.thresholds[best_group_name]:.4f}), "
          f"observed share of A on calib {cal_grp.y_calib_eval.mean():.4f}")
    plot_calibration_curve_(cal_ns.y_calib_eval, cal_ns.p_calib_eval[best_noshow_name],
                            f"{best_noshow_name}_calibrated", n_bins=10, prefix="calib")
    RUN.configure("evaluate", plots=list(ALL_EVAL_PLOTS))
""")

code_stage("evaluate", """\\
# ── 5.2 Test metrics and figures (thresholds from calib) ────────────────────
_tt = np.sort(np.random.default_rng(RANDOM_STATE).choice(N_FV, size=min(200_000, N_FV), replace=False))
results_noshow = evaluate_binary_classification_models(
    cal_ns, X_test_np, y_test_ns, base_rate=noshow_models.prior,
    positive_name="NoShow", reference_name="Attended", feature_cols=feature_cols,
    X_train=XA[_tt], y_tr=y_ns[_tt], prefix="noshow", cache_dir=RUN.path("evaluate", "proba_noshow"))
results_group = evaluate_binary_classification_models(
    cal_grp, X_test_np, y_test_grp, base_rate=group_models.prior,
    positive_name="Group A (long)", reference_name="Group B (short)", feature_cols=feature_cols,
    X_train=XA[_tt], y_tr=y_grp[_tt], prefix="servicetime", cache_dir=RUN.path("evaluate", "proba_group"))
RUN.finish("evaluate", results_noshow=results_noshow, results_group=results_group)
""")

code("""\\
if RUN.wanted("evaluate"):
    EV = RUN.load("evaluate")
    results_noshow, results_group = EV["results_noshow"], EV["results_group"]
    _metrics = pd.concat([results_noshow.assign(Task="no-show"), results_group.assign(Task="group A")])
    display(_metrics.set_index(["Task", "Model"]).round(4))
    save_dataframe(_metrics, os.path.join(RUN_RESULTS_DIR, "test_metrics.csv"))
    latex_table(_metrics[["Task", "Model", "Calibration", AUC_COL, AP_COL, LOGLOSS_COL, BRIER_COL, BSS_COL,
                          ACC_COL, BASE_ACC_COL, F1_COL, F1_POS_COL, THR_COL]],
                os.path.join(TABLES_DIR, "table_test_metrics.tex"),
                caption="Test metrics of the calibrated models (thresholds chosen on calib).",
                label="tab:test-metrics", decimals=4)
    RUN.configure("export", c=(1.0, 0.5, C_O_BASE), slots=SLOT_LENGTHS)
""")

# ═══════════════════════════════════════════════════════════════════════════
# 6-7. Deployment and ILP
# ═══════════════════════════════════════════════════════════════════════════
md("## 6. Deployment and 7. ILP bridge\n\n"
   "Serialisation of the calibrated models, the group threshold and the duration groups; "
   "scoring of the ILP holdout (patients never seen by any model) with `predict_patient`; "
   "export of `plne_parameters.json` (the only input of the ILP campaign, "
   "`notebooks/01_ilp_validation.ipynb`) and of the scored holdout; one real ILP day "
   "solved as a demonstration, with its termination status and gap.")

code_stage("export", """\\
# ── 6.1 Artifacts ───────────────────────────────────────────────────────────
save_artifacts(calibrated_noshow_model, cluster_tree, best_group_model,
               PREP["scaler"], PREP["cat_encodings"], feature_cols, name_prefix="final",
               inference={"group_threshold": GROUP_THRESHOLD, "duration_groups": DURATION_GROUPS,
                          "run_id": RUN_ID, "noshow_model_name": best_noshow_name,
                          "group_model_name": best_group_name})
# ── 7.1 Score the holdout ───────────────────────────────────────────────────
scored_inference = predict_patient(X_inf_np, calibrated_noshow_model, best_group_model,
                                   threshold_A=GROUP_THRESHOLD, s_A=service_time_by_group["A"],
                                   s_B=service_time_by_group["B"])
scored_inference["NoShow_actual"] = y_inf_ns
scored_inference["group_label_actual"] = grp_inf_actual
HOLDOUT_SCORES = pd.DataFrame({"q_i": scored_inference["q_i"], "group_pred": scored_inference["group"],
                               "group_actual": grp_inf_actual, "no_show": y_inf_ns})
# ── 7.2 Export the ILP parameters (run directory; versioned file for a full run)
_paths = [os.path.join(RUN_RESULTS_DIR, "plne_parameters.json")]
if FULL_RUN:
    _paths.append(os.path.join(INFERENCE_DIR, "plne_parameters.json"))
PLNE_PARAMS = export_plne_parameters(_paths, DURATION_GROUPS, HOLDOUT_SCORES, n_train=len(y_ns),
                                     source_run=f"run_{RUN_ID}", smoke=not FULL_RUN, slot_lengths=SLOT_LENGTHS)
export_holdout_scores(os.path.join(RUN_RESULTS_DIR, "holdout_scores.parquet"), HOLDOUT_SCORES)
# ── 7.3 One real ILP day from holdout patients ──────────────────────────────
_day = scored_inference.sample(n=min(30, len(scored_inference)), random_state=RANDOM_STATE)
T_demo, L_demo = [6], [[30.0, 30.0, 30.0, 60.0, 60.0, 60.0]]
real_instance = build_instance_from_patients(
    _day, T=T_demo, L=L_demo, s_A=service_time_by_group["A"], s_B=service_time_by_group["B"],
    c_I=1.0, c_W=0.5, c_O=C_O_BASE, duration_mode="class")   # "expected": s = p_A s_A + (1 - p_A) s_B
real_solution = solve_ilp(real_instance, Config(time_limit=60.0))
RUN.finish("export", plne_params=PLNE_PARAMS, scored_head=scored_inference.head(10),
           share_A=((scored_inference["group"] == "A").mean(), (grp_inf_actual == "A").mean()),
           q_mean=(scored_inference["q_i"].mean(), 1 - y_inf_ns.mean()),
           real_solution={k: getattr(real_solution, k) for k in
                          ("status", "best_bound", "gap", "gap_complete", "Z_complete", "E_idle",
                           "E_wait", "E_OT", "n_served", "n_double", "solve_time")},
           real_c=(real_instance.c_I, real_instance.c_W, real_instance.c_O), real_n=real_instance.n)
""")

code("""\\
if RUN.wanted("export"):
    EX = RUN.load("export")
    print(json.dumps(EX["plne_params"], indent=2))
    display(EX["scored_head"])
    print(f"Share of group A, predicted / actual: {EX['share_A'][0]:.4f} / {EX['share_A'][1]:.4f}")
    print(f"Mean q_i predicted / observed show-up rate: {EX['q_mean'][0]:.4f} / {EX['q_mean'][1]:.4f}")
    _rs = EX["real_solution"]
    print(f"Real ILP day (n={EX['real_n']}): status {_rs['status']}, Z = {_rs['Z_complete']:.4f}, "
          f"gap on complete cost {_rs['gap_complete']:.2e}, gap on reduced objective {_rs['gap']:.2e}, "
          f"{_rs['n_served']} served, {_rs['n_double']} double slots, {_rs['solve_time']:.1f} s")
    from IPython.display import Math
    _obj_tex = objective_latex(*EX["real_c"])
    with open(os.path.join(TABLES_DIR, "objective_solved.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\\\begin{equation}\\n" + _obj_tex + "\\n\\\\end{equation}\\n")
    display(Math(_obj_tex))
""")

md("**Full experimental campaign**: `notebooks/01_ilp_validation.ipynb` (generated by "
   "`notebooks/gen_nb.py`), with `PLNE_PARAMS_PATH` and `HOLDOUT_SCORES_PATH` pointing to "
   "the files written above. It imports the same model core from `data_driven_plan.py`.")

md("## Final summary")

code("""\\
if RUN.wanted("export"):
    _rn, _rg = results_noshow.set_index("Model"), results_group.set_index("Model")
    recap = pd.DataFrame([
        {"Item": "Split", "Value": PREP["split_sizes"].to_dict("records")},
        {"Item": "No-show: selected model", "Value": f"{best_noshow_name} ({cal_ns.methods[best_noshow_name]})"},
        {"Item": "No-show: test AUC-ROC / AUC-PR / Brier / BSS",
         "Value": " / ".join(f"{_rn.loc[best_noshow_name, c]:.4f}" for c in (AUC_COL, AP_COL, BRIER_COL, BSS_COL))},
        {"Item": "Durations", "Value": f"{DURATION_GROUPS.method}/{DURATION_GROUPS.stat}: "
         f"s_A={service_time_by_group['A']:g}, s_B={service_time_by_group['B']:g} min, cutoff {DURATION_GROUPS.cutoff:g} min"},
        {"Item": "Durations: bootstrapped silhouette (K=2)", "Value": f"mean={mu_sil:.4f}, sd={sd_sil:.4f}"},
        {"Item": "Group: selected model", "Value": f"{best_group_name} ({cal_grp.methods[best_group_name]})"},
        {"Item": "Group: test AUC-ROC / Brier",
         "Value": f"{_rg.loc[best_group_name, AUC_COL]:.4f} / {_rg.loc[best_group_name, BRIER_COL]:.4f}"},
        {"Item": "ILP day: status, Z", "Value": f"{EX['real_solution']['status']}, {EX['real_solution']['Z_complete']:.4f}"},
    ])
    display(recap)
    save_dataframe(recap, os.path.join(RUN_RESULTS_DIR, "recap_data_driven_plan.csv"))
""")

# ═══════════════════════════════════════════════════════════════════════════
# Keep the metadata of an existing notebook, but always pin a portable
# kernelspec: a locally named kernel (a venv registered as "ml_env", say) does
# not exist on any other machine, and executing the notebook there fails with
# NoSuchKernel before a single cell runs.
metadata = dict(DEFAULT_METADATA)
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        existing = json.load(f).get("metadata") or {}
    metadata = {**existing, "kernelspec": DEFAULT_METADATA["kernelspec"]}

nb = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Notebook written: {OUT}  ({len(cells)} cells)")
