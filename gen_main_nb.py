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
C_O_BASE        = 0.5           # V9: overtime kept (manuscript to update); 0 for the ablation
SLOT_LENGTHS    = (30, 60)      # B4: s_A <= max(SLOT_LENGTHS) is asserted, the run stops otherwise
SPLIT_MODE      = "stratified"  # "patient_group" for the ablation
RANDOM_STATE    = 0
SMOKE           = False         # True: 1% of the patients, minimal budgets
RUN_ID          = None          # None: timestamp of the run; artifacts under models/ and results/run_<RUN_ID>/
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
for k, v in {**DECISIONS, "RUN_ID": RUN_ID}.items():
    print(f"{k:<16} {v}")
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

code("""\
# ── 2.1 Load the raw data ─────────────────────────────────────────────────────
# Paths come from the module and follow SMART_PLAN_DATA (default: ./data), so
# this notebook runs unchanged wherever the dataset lives.
print(f"Data root: {DATA_DIR}")
data = pd.read_parquet(RAW_PARQUET_DIR, engine="pyarrow")
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

code("""\
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

code("""\
# ── 2.3 Raw numeric distributions ─────────────────────────────────────────────
plot_numeric_histograms(data, ["dta", "dts", "dtc"], bins=[50, 50, 50],
                        xlims=[(0, 100), (0, 250), (0, 500)], prefix="eda")
""")

code("""\
# ── 2.3 (cont.) Missing values ────────────────────────────────────────────────
missing_pct = display_missing_values(data, sentinel_col="state", sentinel_value="*Missing*")
print(missing_pct[missing_pct > 0].sort_values(ascending=False))
""")

md("`AdministrativeFlag` exceeds 55% missing values: reliable imputation is "
   "illusory at that rate, so the column is dropped rather than imputed "
   "(section 3). `zip`/`state` are missing for a marginal share of rows, "
   "listwise deletion is preferred to imputation, with no material impact on a "
   "dataset of this size.")

code("""\
# ── 2.3 (cont.) Outlier diagnostic (IQR), visualisation only ────────────────
visualize_outliers(data, prefix="eda")
""")

code("""\
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

code("""\
# ── 3.1 Temporal filtering + calendar features ────────────────────────────────
data = filter_dates_after(data, "activitydatetime", "2016-08-30 00:00:00")
data = add_temporal_features(data, "activitydatetime")
print(f"Shape after temporal filtering: {data.shape}")
""")

code("""\
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

code("""\
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

code("""\
# ── 3.2 Filter out stop codes with no no-show history ────────────────────────
data, _data_without_ns, _codes_dropped = separate_stopcodes_by_no_show(data)
print(f"Stop codes excluded (never any no-show): {len(_codes_dropped)}")
print(f"Shape after stop-code filtering: {data.shape}")
""")

md("**Patient history features:** days since the previous visit (per "
   "patient/stop code) and running averages of prior no-shows and prior "
   "service times (per patient), lagged by one rank to exclude any information "
   "leaking from the current visit.")

code("""\
# ── 3.2 Patient history features ──────────────────────────────────────────────
data = add_patient_history_features(data, "patientsid", "stopcode", "activitydatetime",
                                    "NoShow", "service_time")
print(data[["days_between_visits", "prior_noShow_rate", "prior_service_time_avg"]].describe())
""")

code("""\
# ── 3.2 Final feature selection + checkpoint ─────────────────────────────────
# patientsid is kept until the split (SPLIT_MODE="patient_group" needs it) and
# excluded from the features by prepare_features.
data = data.drop(columns=["activitydatetime"])
print(f"Final preprocessed shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
save_preprocessed_data(data, PREPROCESSED_PATH)
print(f"Saved -> {PREPROCESSED_PATH}")
""")

md("### 3.2bis Fast resume point (optional)\n\n"
   "Section 2 (Data Understanding) and sections 3.1-3.2 (cleaning + feature "
   "engineering) are expensive on the 40M+ raw rows. If "
   "the preprocessed Parquet file already exists from a "
   "previous run, the cell below skips them entirely when resuming: run only "
   "the configuration cell at the very top, uncomment and run this one, then "
   "continue normally from section 3.3. On a full run from the raw data, leave "
   "it commented out (the `data` already in memory is identical).\n\n"
   "**Parquet rather than CSV:** it preserves the optimised dtypes (category, "
   "uint32, float32...), a CSV round-trip would restore them as "
   "object/int64/float64, multiplying the memory footprint of a dataset of "
   "this size.")

code("""\
# ── 3.2bis Fast resume: reload the preprocessed data instead of sections 2-3.2 ──
# data = load_preprocessed_data(PREPROCESSED_PATH)
# print(f"Preprocessed data reloaded: {data.shape}")
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

code("""\
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

code("""\
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
md("## 4. Modeling\n\n"
   "Two sub-pipelines on the SAME split and features: no-show (4a) and service-time "
   "group (4b). Four families, as in the published study: RandomForest, XGBoost, "
   "AdaBoost, MLP (scikit-learn, xgboost), with the encoding of run_20260926_2002.\n\n"
   "**Protocol (audit 1.3).** Hyperparameters by successive halving "
   "(`HalvingRandomSearchCV`) on the predefined fit / valid split: candidates are trained "
   "on fit and scored on valid, with training rows growing from 500,000 to all of fit. "
   "Criterion: log-loss on valid of the probabilities corrected back to the true base rate "
   "(the models are trained on balanced classes). AdaBoost: one fit per learning rate at "
   "the maximal number of estimators, iterations chosen on valid by `staged_predict_proba`. "
   "Final model refitted on fit + valid.\n\n"
   "**Imbalance (audit 1.4, as in the first paper: SMOTE with class weighting).** Every "
   "family is a pipeline applying SMOTE at fit time only, up to a minority / majority "
   "ratio of 0.1, then weights balancing the resampled classes (`class_weight` for the "
   "forest, `scale_pos_weight` for XGBoost, `sample_weight` for AdaBoost and the MLP).")

code("""\
# ── 4.0 Training settings ───────────────────────────────────────────────────
TRAIN_CFG_NOSHOW = TrainingConfig(
    models=["RandomForest", "XGBoost", "AdaBoost", "ANN"],
    search="halving", n_candidates=12, halving_factor=3,
    min_resources=500_000,        # fit rows of the first halving iteration
    max_resources=None,           # None: all of fit at the last iteration
    smote_sampling_strategy=0.1, smote_k_neighbors=5,
    adaboost_max_estimators=400, adaboost_learning_rates=(0.5, 1.0),
    threshold_metric="youden", n_jobs=-1, verbose=1, random_state=RANDOM_STATE,
)
if SMOKE:   # minimal budgets: the smoke run checks that the code runs, not the results
    TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(
        n_candidates=4, min_resources=20_000, adaboost_max_estimators=40,
        base_params={"ANN": {"max_iter": 20}})
TRAIN_CFG_GROUP = TRAIN_CFG_NOSHOW.replace()
display(TRAIN_CFG_NOSHOW.summary())
""")

code("""\
# ── 4.0 Contiguous float32 matrices: fit | valid | calib are row slices (views) ─
XA = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32))
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))
X_inf_np = np.ascontiguousarray(X_inference.to_numpy(dtype=np.float32))
N_FIT, N_VALID = len(SPLIT.fit), len(SPLIT.valid)
N_FV = N_FIT + N_VALID
del X_train, X_fit, X_valid, X_calib
y_ns = train_df["NoShow"].to_numpy().astype(int)
y_test_ns = test_df["NoShow"].to_numpy().astype(int)
print(f"XA {XA.shape} ({XA.nbytes/1e9:.1f} GB) | test {X_test_np.shape} | holdout {X_inf_np.shape}")
""")

md("### 4a. No-show prediction")

code("""\
# ── 4a. Tune on fit / valid, refit on fit + valid ──────────────────────────
noshow_models = train_binary_classifiers(XA[:N_FV], y_ns[:N_FV], N_FIT, TRAIN_CFG_NOSHOW)
display(noshow_models.search_report)
save_model_bundle({"models": noshow_models, "config": TRAIN_CFG_NOSHOW}, name="noshow_training_state")
""")

md("### 4b. Service time, duration groups then classification\n\n"
   "**Exact K-medians, K = 2, on the training durations only** (audit B3). Service times "
   "are integer minutes, so the optimal 1-D K-medians partition is found exactly by "
   "dynamic programming over the distinct values (no initialisation, no randomness). "
   "The group durations s_A, s_B are the cluster medians. The groups and their cutoff "
   "are fitted on the training partition and applied unchanged to test and holdout. "
   "K = 2 is imposed by the ILP (two duration classes); the cost and silhouette for "
   "K = 2 to 5 are reported for reference. `DURATION_METHOD = \"kmeans\"` runs the "
   "ablation with the former method.")

code("""\
# ── 4b. Duration-group settings ─────────────────────────────────────────────
DURATION_KS = [2, 3, 4, 5]    # values of K reported (cost + silhouette)
SIL_SAMPLE = 20_000           # durations used for each silhouette
N_BOOTSTRAP = 10              # bootstrap resamples of the silhouette (local generator)
BOOTSTRAP_SAMPLE = 10_000     # size of each resample
""")

code("""\
# ── 4b. Fit the groups on the training durations, apply them everywhere ─────
_train_st = train_df["service_time"].to_numpy()
duration_k_table = evaluate_duration_ks(_train_st, DURATION_KS, method=DURATION_METHOD,
                                        sil_sample=SIL_SAMPLE, random_state=RANDOM_STATE,
                                        prefix="servicetime")
display(duration_k_table)
DURATION_GROUPS = fit_duration_groups(_train_st, k=2, method=DURATION_METHOD, stat=DURATION_STAT,
                                      random_state=RANDOM_STATE)
best_k = 2
mu_sil, sd_sil = bootstrap_silhouette(_train_st, DURATION_GROUPS, n_bootstrap=N_BOOTSTRAP,
                                      sample_size=BOOTSTRAP_SAMPLE, random_state=RANDOM_STATE)
df_sample["group_label"] = DURATION_GROUPS.labels(df_sample["service_time"])
train_df["group_label"] = df_sample.loc[train_df.index, "group_label"]
service_time_by_group = DURATION_GROUPS.s_by_group
label_names = ["A", "B"]
print(f"Method {DURATION_GROUPS.method} / {DURATION_GROUPS.stat}, fitted on {DURATION_GROUPS.n_train:,} "
      f"training durations")
print(f"s_A = {service_time_by_group['A']:g} min, s_B = {service_time_by_group['B']:g} min, "
      f"cutoff {DURATION_GROUPS.cutoff:g} min, training share of A {DURATION_GROUPS.shares['A']:.4f}")
print(f"Bootstrapped silhouette (K=2): mean={mu_sil:.4f}, sd={sd_sil:.4f}")
# Audit B4: a long patient must fit in the longest slot, otherwise stop here
if service_time_by_group["A"] > max(SLOT_LENGTHS):
    raise ValueError(f"s_A = {service_time_by_group['A']:g} min exceeds the longest slot "
                     f"{max(SLOT_LENGTHS)} min: decision required (SLOT_LENGTHS or DURATION_METHOD)")
""")

code("""\
# ── 4b. Interpretable cutoff (decision tree on the training labels) ─────────
cluster_tree, primary_cutoff = learn_decision_tree_cutoffs(
    train_df, service_time_col="service_time", group_col="group_label", label_names=label_names,
)
assert abs(primary_cutoff - DURATION_GROUPS.cutoff) <= 0.5 + 1e-9, \\
    f"tree root {primary_cutoff} and group cutoff {DURATION_GROUPS.cutoff} disagree"
""")

md("**Group classifier.** Same function and protocol; the A/B target (about 35 % of "
   "A) is above the SMOTE ratio of 0.1, so only the class weighting applies.")

code("""\
# ── 4b. Group classifier for new patients (target: training-fitted groups) ──
y_grp = (df_sample.loc[train_df.index, "group_label"] == "A").to_numpy().astype(int)
y_test_grp = (df_sample.loc[test_df.index, "group_label"] == "A").to_numpy().astype(int)
group_models = train_binary_classifiers(XA[:N_FV], y_grp[:N_FV], N_FIT, TRAIN_CFG_GROUP)
display(group_models.search_report)
save_model_bundle({"models": group_models, "config": TRAIN_CFG_GROUP}, name="group_training_state")
""")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluation
# ═══════════════════════════════════════════════════════════════════════════
md("## 5. Calibration, model choice and evaluation\n\n"
   "**On calib** (never on test): each family is calibrated (analytic prior correction, "
   "Platt scaling or isotonic regression, fitted on one half of calib); on the other half "
   "the calibrated Brier score chooses the method of each family, then the model; Youden's "
   "threshold is read there too. **On test**: AUC-ROC, AUC-PR, log-loss, Brier, Brier skill "
   "score against the training base rate, then at the calib threshold accuracy (with the "
   "majority-class accuracy for reference) and precision, recall, F1 in macro average and "
   "for the positive class (audit P5).")

code("""\
# ── 5.1 No-show: calibration and model choice on calib ──────────────────────
cal_ns = calibrate_and_select(noshow_models, XA[N_FV:], y_ns[N_FV:], random_state=RANDOM_STATE)
display(cal_ns.table.round(5))
best_noshow_name = cal_ns.best_name
calibrated_noshow_model = cal_ns.models[best_noshow_name]
print(f"Selected no-show model (calibrated Brier on calib): {best_noshow_name} "
      f"({cal_ns.methods[best_noshow_name]})")
""")

code("""\
# ── 5.2 No-show: reliability diagram of the selected model (calib half B) ───
plot_calibration_curve_(cal_ns.y_calib_eval, cal_ns.p_calib_eval[best_noshow_name],
                        f"{best_noshow_name}_calibrated", n_bins=10, prefix="calib")
""")

code("""\
# ── 5.3 No-show: test metrics ───────────────────────────────────────────────
_rng_tt = np.random.default_rng(RANDOM_STATE)
_tt = _rng_tt.choice(N_FV, size=min(200_000, N_FV), replace=False)
results_noshow = evaluate_binary_classification_models(
    cal_ns, X_test_np, y_test_ns, base_rate=noshow_models.prior,
    positive_name="NoShow", reference_name="Attended", feature_cols=feature_cols,
    X_train=XA[np.sort(_tt)], y_tr=y_ns[np.sort(_tt)], prefix="noshow",
    cache_dir=os.path.join(MODELS_DIR, f"run_{RUN_ID}", "test_predictions_noshow"))
record_training_run("no-show", results_noshow)
""")

code("""\
# ── 5.4 Group: calibration, model choice, thresholds on calib ───────────────
cal_grp = calibrate_and_select(group_models, XA[N_FV:], y_grp[N_FV:], random_state=RANDOM_STATE)
display(cal_grp.table.round(5))
best_group_name = cal_grp.best_name
best_group_model = cal_grp.models[best_group_name]
print(f"Selected group model (calibrated Brier on calib): {best_group_name} ({cal_grp.methods[best_group_name]})")
""")

code("""\
# ── 5.5 Group: test metrics ─────────────────────────────────────────────────
results_group = evaluate_binary_classification_models(
    cal_grp, X_test_np, y_test_grp, base_rate=group_models.prior,
    positive_name="Group A (long)", reference_name="Group B (short)", feature_cols=feature_cols,
    X_train=XA[np.sort(_tt)], y_tr=y_grp[np.sort(_tt)], prefix="servicetime",
    cache_dir=os.path.join(MODELS_DIR, f"run_{RUN_ID}", "test_predictions_group"))
record_training_run("A/B group", results_group)
history = training_history_table()
save_dataframe(history, os.path.join(RUN_RESULTS_DIR, "test_metrics.csv"))
""")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Deployment
# ═══════════════════════════════════════════════════════════════════════════
md("## 6. Deployment\n\n"
   "Serialisation of the three inference artifacts (calibrated no-show model, "
   "service_time→group tree for interpretability, group classifier for "
   "inference on new patients) and of the preprocessing objects fitted on the "
   "training set. `predict_patient` is the single inference entry point, used "
   "as-is by section 7.")

code("""\
# ── 6.1 Serialise the artifacts ───────────────────────────────────────────────
save_artifacts(
    calibrated_noshow_model, cluster_tree, best_group_model,
    scaler, cat_encodings, feature_cols, name_prefix="final",
)
""")

code("""\
# ── 6.2 predict_patient demonstration on a test-set sample ──────────────────
demo_scores = predict_patient(
    X_test.head(10), calibrated_noshow_model, best_group_model,
    positive_group_label="A", reference_group_label="B",
)
demo_scores["NoShow_actual"] = test_df["NoShow"].head(10).values
display(demo_scores)
""")

# ═══════════════════════════════════════════════════════════════════════════
# 7. Solving the ILP
# ═══════════════════════════════════════════════════════════════════════════
md("## 7. Solving the ILP (outside CRISP-DM)\n\n"
   "The inference holdout (`data/inference/inference_set_raw.csv`, "
   "`df_inference`) was never seen by any model of sections 4-6. It is used "
   "here to demonstrate the end-to-end bridge: serialised artifacts → "
   "`predict_patient` → real ILP instance → `solve_ilp`.\n\n"
   "`Instance`/`Config`/`Solution`/`compute_coefficients`/`solve_ilp` are "
   "imported from `data_driven_plan.py`, the **same** model core as "
   "`01_ilp_validation.ipynb` (regenerated by `notebooks/gen_nb.py`), which "
   "carries the full experimental campaign (illustrative example, design of "
   "experiments, benchmarks B1-B4, out-of-sample simulation, sensitivity "
   "analysis). This section does not duplicate it: it adds the missing step, "
   "building an instance from **real patients never used in training**, rather "
   "than from synthetic patients drawn from distributions calibrated on "
   "aggregate statistics.")

code("""\
# ── 7.1 Load the serialised artifacts ─────────────────────────────────────────
artifacts = load_artifacts(name_prefix="final")
print("Artifacts loaded:", list(artifacts.keys()))
""")

code("""\
# ── 7.2 Inference on the holdout patients (never trained on) ────────────────
scored_inference = predict_patient(
    X_inference, artifacts["noshow_model"], artifacts["group_classifier"],
    positive_group_label="A", reference_group_label="B",
)
scored_inference["NoShow_actual"] = df_inference["NoShow"].values
scored_inference["group_label_actual"] = df_sample.loc[df_inference.index, "group_label"].values

print("Predictions on patients never seen during training:")
display(scored_inference.head(10))
print(f"\\nMean predicted show-up rate (q_i): {scored_inference['q_i'].mean():.3f}"
      f"   |   observed (1-NoShow): {(1 - scored_inference['NoShow_actual']).mean():.3f}")
print(f"Predicted group distribution:\\n{scored_inference['group'].value_counts()}")
print(f"Actual group distribution   :\\n{scored_inference['group_label_actual'].value_counts()}")
""")

code("""\
# ── 7.2b Export the ILP parameters and the scored holdout ───────────────────
# plne_parameters.json is the only input of notebooks/01_ilp_validation.ipynb.
# A smoke run writes it to its run directory only; a full run also writes the
# versioned data/inference/plne_parameters.json.
HOLDOUT_SCORES = pd.DataFrame({
    "q_i": scored_inference["q_i"].to_numpy(),
    "group_pred": scored_inference["group"].to_numpy(),
    "group_actual": scored_inference["group_label_actual"].to_numpy(),
    "no_show": scored_inference["NoShow_actual"].astype(int).to_numpy()})
_param_paths = [os.path.join(RUN_RESULTS_DIR, "plne_parameters.json")]
if not SMOKE:
    _param_paths.append(os.path.join(INFERENCE_DIR, "plne_parameters.json"))
PLNE_PARAMS = export_plne_parameters(_param_paths, DURATION_GROUPS, HOLDOUT_SCORES,
                                     n_train=len(train_df), source_run=f"run_{RUN_ID}",
                                     smoke=SMOKE, slot_lengths=SLOT_LENGTHS)
export_holdout_scores(os.path.join(RUN_RESULTS_DIR, "holdout_scores.parquet"), HOLDOUT_SCORES)
print(json.dumps(PLNE_PARAMS, indent=2))
print("Written:", _param_paths, "and holdout_scores.parquet")
""")

code("""\
# ── 7.3 Build a real ILP instance (one clinic day) ──────────────────────────
N_DEMO = 30  # size of an illustrative day, at realistic scale
demo_patients = scored_inference.sample(n=min(N_DEMO, len(scored_inference)), random_state=RANDOM_STATE)

T_demo = [6]
L_demo = [[30.0, 30.0, 30.0, 60.0, 60.0, 60.0]]
s_A_real = service_time_by_group["A"]
s_B_real = service_time_by_group["B"]

real_instance = build_instance_from_patients(
    demo_patients, T=T_demo, L=L_demo, s_A=s_A_real, s_B=s_B_real,
    c_I=1.0, c_W=0.5, c_O=C_O_BASE, group_col="group", q_col="q_i", positive_group="A",
)
print(f"Real instance: n={real_instance.n} patients, {sum(T_demo)} slots "
      f"(max capacity {2*sum(T_demo)} seats), s_A={s_A_real:.1f} min, s_B={s_B_real:.1f} min")
""")

code("""\
# ── 7.4 Solve ─────────────────────────────────────────────────────────────────
real_solution = solve_ilp(real_instance, Config(time_limit=60.0))
print(f"Status: {real_solution.status}   |   solve time: {real_solution.solve_time:.3f} s")
print(f"Best bound (reduced): {real_solution.best_bound:.4f}   |   "
      f"gap on complete cost: {real_solution.gap_complete:.2e}   |   "
      f"gap on reduced objective: {real_solution.gap:.2e}")
print(f"Z_reduced = {real_solution.Z_reduced:.4f}   Z_complete = {real_solution.Z_complete:.4f}")
print(f"E[Idle]={real_solution.E_idle:.3f}  E[Wait]={real_solution.E_wait:.3f}  E[OT]={real_solution.E_OT:.3f}")
print(f"Patients served: {real_solution.n_served}/{real_instance.n}   "
      f"Double-booked slots: {real_solution.n_double}/{sum(T_demo)}")
""")

code("""\
# ── 7.5 Objective actually solved (costs passed to the solver) ──────────────
from IPython.display import Math, display
_obj_tex = objective_latex(real_instance.c_I, real_instance.c_W, real_instance.c_O)
import os
with open(os.path.join(TABLES_DIR, "objective_solved.tex"), "w", encoding="utf-8") as fh:
    fh.write("\\\\begin{equation}\\n" + _obj_tex + "\\n\\\\end{equation}\\n")
display(Math(_obj_tex))
print(_obj_tex)
""")

md("**Full experimental campaign** (design of experiments with distributional "
   "notation and factor levels, benchmarks B1-B4, out-of-sample simulation, "
   "sensitivity analysis to the no-show rate / variance / load with percentage "
   "deviation and solve time): see the companion notebook "
   "**`notebooks/01_ilp_validation.ipynb`**, regenerated by `notebooks/gen_nb.py`. "
   "It imports its model core "
   "(`Instance`/`Config`/`solve_ilp`) from `data_driven_plan.py`, so both "
   "solve exactly the same model.")

# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════
md("## Final summary")

code("""\
# ── Summary table ─────────────────────────────────────────────────────────────
_rn, _rg = results_noshow.set_index("Model"), results_group.set_index("Model")
recap = pd.DataFrame([
    {"Item": "Split", "Value": f"{SPLIT.mode}: fit={len(SPLIT.fit):,} valid={len(SPLIT.valid):,} "
     f"calib={len(SPLIT.calib):,} test={len(SPLIT.test):,} holdout={len(SPLIT.inference):,}"},
    {"Item": "No-show: selected model (calibrated Brier on calib)",
     "Value": f"{best_noshow_name} ({cal_ns.methods[best_noshow_name]})"},
    {"Item": "No-show: test AUC-ROC / AUC-PR / Brier / BSS",
     "Value": " / ".join(f"{_rn.loc[best_noshow_name, c]:.4f}" for c in (AUC_COL, AP_COL, BRIER_COL, BSS_COL))},
    {"Item": "Durations", "Value": f"{DURATION_GROUPS.method}/{DURATION_GROUPS.stat}: s_A={service_time_by_group['A']:g}, "
     f"s_B={service_time_by_group['B']:g} min, cutoff {DURATION_GROUPS.cutoff:g} min"},
    {"Item": "Durations: bootstrapped silhouette (K=2)", "Value": f"mean={mu_sil:.4f}, sd={sd_sil:.4f}"},
    {"Item": "Group: selected model", "Value": f"{best_group_name} ({cal_grp.methods[best_group_name]})"},
    {"Item": "Group: test AUC-ROC / Brier",
     "Value": f"{_rg.loc[best_group_name, AUC_COL]:.4f} / {_rg.loc[best_group_name, BRIER_COL]:.4f}"},
    {"Item": "ILP: real instance (n patients)", "Value": str(real_instance.n)},
    {"Item": "ILP: status, Z_complete", "Value": f"{real_solution.status}, {real_solution.Z_complete:.4f}"},
])
display(recap)
save_dataframe(recap, "results/recap_data_driven_plan.csv")
print("Summary saved -> results/recap_data_driven_plan.csv")
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
