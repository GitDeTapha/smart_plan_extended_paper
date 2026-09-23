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


# ═══════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════
md("# data_driven_plan — Cluster-Predict-Schedule pipeline\n\n"
   "Data-driven outpatient appointment scheduling: no-show and service-time "
   "group prediction, feeding a stochastic overbooking ILP "
   "(at most k = 2 patients per slot).\n\n"
   "A single CRISP-DM pipeline. All utility functions live in "
   "`data_driven_plan.py` — this notebook holds only the orchestration, the "
   "narrative and the outputs.\n\n"
   "**Previously validated results, reproduced identically:** K=2 "
   "(service_time clustering, bootstrapped silhouette), cutoff ~38.5 min "
   "(decision tree), XGBoost as best no-show classifier.")

code("""\
# ── Configuration and reproducibility ─────────────────────────────────────────
import sys, platform, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data_driven_plan import *

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
ensure_output_dirs()

print(f"Python   : {sys.version.split()[0]}")
print(f"NumPy    : {np.__version__}")
print(f"Pandas   : {pd.__version__}")
print(f"Platform : {platform.processor()}")
print(f"ILP solver : {SOLVER_NAME}")
print(f"Native XGBoost: {XGB_AVAILABLE} | SMOTE available: {SMOTE_AVAILABLE}")
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
   "85/15 assumption — which drives the choice of imbalance strategy (SMOTE + "
   "class weighting + optimal threshold, section 4a) and makes calibration "
   "(section 5) more important than discrimination alone.")

md("### 2.3 Exploring the numeric and temporal variables\n\n"
   "`dta` (waiting time before the appointment) and `dts`/`dtc` (service "
   "start/end) are explored before cleaning — their raw values contain "
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
   "(section 3). `zip`/`state` are missing for a marginal share of rows — "
   "listwise deletion is preferred to imputation, with no material impact on a "
   "dataset of this size.")

code("""\
# ── 2.3 (cont.) Outlier diagnostic (IQR) — visualisation only ────────────────
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
   "**Temporal filtering:** data prior to 2016-08-30 is excluded — it predates "
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
   "start time (`dtc < dts`) reflects a data-entry error — `service_time` "
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
   "no-show are kept — a stop code with no no-show history carries no "
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
data = data.drop(columns=["activitydatetime", "patientsid"])
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
   "uint32, float32...) — a CSV round-trip would restore them as "
   "object/int64/float64, multiplying the memory footprint of a dataset of "
   "this size.")

code("""\
# ── 3.2bis Fast resume: reload the preprocessed data instead of sections 2-3.2 ──
# data = load_preprocessed_data(PREPROCESSED_PATH)
# print(f"Preprocessed data reloaded: {data.shape}")
""")

md("### 3.3 Single partition\n\n"
   "One split, stratified jointly on `NoShow` (strong imbalance) and on "
   "`service_time` quantiles, guarantees that BOTH modelling pipelines "
   "(no-show and service time) inherit a representative train/test split, with "
   "no leakage between them or with the ILP holdout — reserved exclusively for "
   "patients never seen during training (section 7).\n\n"
   "**The full set of rows is used** (no subsampling at this stage): the split "
   "yields exactly 70% train / 20% test / 10% inference holdout of the total. "
   "Subsampling to 800k rows, consistent with the already validated clustering "
   "(K=2, bootstrapped silhouette), happens later, when the models are trained "
   "(section 4), to keep AdaBoost/ANN tractable.")

code("""\
# ── 3.3 Single split (train / test / ILP holdout) ────────────────────────────
df_sample, train_df, test_df, df_inference = make_master_split(
    data, train_size=0.70, test_size=0.20, inference_size=0.10,
    noshow_col="NoShow", service_time_col="service_time",
    n_quantile_bins=10, random_state=RANDOM_STATE,
)
""")

md("**Leakage warning:** `discontinued` (section 2.2) is derived from the same "
   "source column `disp` as `NoShow`, and is not known before the appointment "
   "in any case — it is therefore excluded from the features, like the targets "
   "themselves. `completed`, which made the target reconstructible "
   "(NoShow ≈ 1 − completed − discontinued), is no longer produced at all "
   "since section 2.2; it is nevertheless still listed below as a safeguard, "
   "in case the fast resume path (section 3.2bis) reloads a Parquet file "
   "preprocessed by an earlier run.\n\n"
   "**Real-world availability warning:** `dts` (actual service start) and "
   "`dta` (waiting time before the appointment) are known only once the visit "
   "is under way or finished — at the moment the prediction must actually be "
   "used (before the appointment, to decide on overbooking), they do not yet "
   "exist. `dtc` is already dropped after `service_time` is computed "
   "(section 3.1); `dts`/`dta` are therefore excluded from the features too, "
   "even though they remain in `data` for now.")

code("""\
# ── 3.3 Encoding + standardisation (fitted on train only) ────────────────────
X_train, X_test, X_inference, feature_cols, scaler, cat_encodings = prepare_features(
    train_df, test_df, inference_df=df_inference,
    categorical_cols=["stopcode", "sta3n", "non_va", "state", "zip"],
    excluded_cols={"service_time", "NoShow", "group_label", "cluster",
                   "discontinued", "dts", "dta",
                   "completed"},  # safeguard: no longer produced, except in an older Parquet
    random_state=RANDOM_STATE,
)
""")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Modeling
# ═══════════════════════════════════════════════════════════════════════════
md("## 4. Modeling\n\n"
   "Two parallel sub-pipelines, on the SAME split and the SAME features "
   "(section 3.3): no-show prediction (4a) and service-time group assignment "
   "(4b). Both reuse the single function `train_binary_classifiers` "
   "(RandomForest, XGBoost, AdaBoost, ANN — SMOTE + class weighting + optimal "
   "threshold).\n\n"
   "`train_df` (70% of the full dataset, potentially tens of millions of rows) "
   "acts as a reservoir, but the final model fit runs on a bounded subsample "
   "`fit_sample` drawn from `train_df` — AdaBoost and the scikit-learn MLP "
   "(sequential, non-distributed implementations) would be impractical on the "
   "full training set. The tuning subsample (`search_sample`) is itself drawn "
   "from that fit subsample.\n\n"
   "**No hyperparameter is hard-coded in the module:** section 4.0 below "
   "gathers every training lever into an editable `TrainingConfig`, and "
   "section 4c provides the tuning loop (variant → re-evaluation → "
   "comparison).")

md("### 4.0 Training control panel\n\n"
   "Each lever, what it changes, and when to touch it:\n\n"
   "| Setting | Effect | Tuning hint |\n"
   "|---|---|---|\n"
   "| `models` | families trained | keep only one (e.g. `[\"XGBoost\"]`) to iterate fast, then reopen |\n"
   "| `search` | `grid` exhaustive / `random` (n_iter draws) / `none` (direct fit) | `random` affords far wider grids at equal budget |\n"
   "| `n_iter` | budget of `search=\"random\"` | 40-60 explores a wide space seriously |\n"
   "| `cv` | cross-validation folds of the tuning | 5 for a more stable choice, 2 to go fast |\n"
   "| `scoring` | score optimised by the search | `average_precision` suits ~3% positives better than `roc_auc`; `recall` when missing a no-show is costly |\n"
   "| `search_sample` | tuning rows | raising it makes hyperparameter selection more reliable (roughly linear cost) |\n"
   "| `fit_sample` | final fit rows | first lever on raw performance; `None` = the whole `train_df` (very slow for AdaBoost/ANN) |\n"
   "| `use_smote` / `smote_params` | synthetic oversampling of the tuning subset | `{\"sampling_strategy\": 0.3}` avoids the overcorrection of full SMOTE |\n"
   "| `balance_classes` / `scale_pos_weight` | class weighting | three imbalance mechanisms stack up: switching one off can improve precision |\n"
   "| `param_grids` / `base_params` | grids explored / fixed hyperparameters | `DEFAULT_PARAM_GRIDS` is the starting point, printed below |\n"
   "| `threshold_metric` | decision-threshold criterion | `f1_positive` or `youden` favour no-show recall — **without retraining** |\n\n"
   "Reported metrics are fixed: **AUC, accuracy, precision, recall, F1-score**, "
   "the last three macro-averaged over both classes.\n\n"
   "**On the ANN.** `MLPClassifier` accepts neither `class_weight` nor "
   "`sample_weight`, so it was the only family left untreated against a 3% "
   "positive rate and collapsed onto the majority class. It is now fitted "
   "inside an imbalanced-learn pipeline that applies SMOTE during training "
   "only, and searched over a wider space (three architectures, three "
   "regularisation levels, two learning rates).\n\n"
   "The split (section 3.3), clustering (4b) and calibration (5.2) settings "
   "stay in their own cells.")

code("""\
# ── 4.0 Training settings — no-show ───────────────────────────────────────────
# Edit here, rerun 4a, compare with training_history_table() (4c).
TRAIN_CFG_NOSHOW = TrainingConfig(
    models=["RandomForest", "XGBoost", "AdaBoost", "ANN"],  # families to train
    search="grid",                  # "grid" | "random" | "none"
    n_iter=20,                      # draws when search="random"
    cv=3,                           # cross-validation folds of the tuning
    scoring="roc_auc",              # e.g. "average_precision", "f1_macro", "recall"
    search_sample=80_000,           # tuning rows     (None = all of X_fit)
    fit_sample=800_000,             # final fit rows  (None = all of X_train)
    use_smote=True,                 # SMOTE on the tuning subset only
    smote_params={},                # e.g. {"sampling_strategy": 0.3, "k_neighbors": 3}
    balance_classes=True,           # class_weight="balanced" / scale_pos_weight
    scale_pos_weight=None,          # None = n_neg/n_pos; a number to damp it
    param_grids={},                 # per-model grid overrides
    base_params={},                 # per-model fixed hyperparameters
    threshold_metric="f1_macro",    # "f1_positive" | "balanced_accuracy" | "youden" | "fixed"
    threshold_range=(0.05, 0.96, 0.01),
    fixed_threshold=0.5,            # used when threshold_metric="fixed"
    n_jobs=-1, verbose=1, random_state=RANDOM_STATE,
)

print("Module default grids (starting point for param_grids):")
for _m, _g in DEFAULT_PARAM_GRIDS.items():
    print(f"  {_m:<13} {_g}")
print("\\nDefault fixed hyperparameters (starting point for base_params):")
for _m, _b in DEFAULT_BASE_PARAMS.items():
    print(f"  {_m:<13} {_b}")
print()
display(TRAIN_CFG_NOSHOW.summary())
""")

md("**Ready-made variants.** Uncomment ONE line (or write another on the same "
   "pattern) then rerun the training cell 4a. `.replace()` returns a copy, so "
   "the reference configuration `TRAIN_CFG_NOSHOW` stays intact.")

code("""\
# ── 4.0 (variants) Tuning recipes — uncomment the one to test ────────────────
# Fast iteration (one family, reduced grid, less data):
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(
#     models=["XGBoost"], search_sample=20_000, fit_sample=200_000, cv=2)

# Wide exploration at controlled budget (random search over an extended grid):
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(
#     search="random", n_iter=40, scoring="average_precision",
#     param_grids={"XGBoost": {"n_estimators": [200, 400, 800],
#                              "max_depth": [3, 4, 6, 8, 10],
#                              "learning_rate": [0.01, 0.03, 0.05, 0.1],
#                              "subsample": [0.6, 0.8, 1.0],
#                              "colsample_bytree": [0.6, 0.8, 1.0],
#                              "min_child_weight": [1, 5, 20],
#                              "reg_lambda": [1.0, 5.0, 20.0]}})

# Favour no-show recall (the ILP overbooks on that signal):
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(
#     scoring="average_precision", threshold_metric="f1_positive")

# Test the threefold imbalance strategy by removing one component:
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(use_smote=False)
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(balance_classes=False)
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(smote_params={"sampling_strategy": 0.3})

# Regularise the ANN / widen the network:
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(
#     param_grids={"ANN": {"hidden_layer_sizes": [(128, 64), (256, 128, 64)],
#                          "alpha": [1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-3, 3e-4]}},
#     base_params={"ANN": {"max_iter": 800, "n_iter_no_change": 25}})

# Imposed hyperparameters, no search at all (reproduce a known setting):
# TRAIN_CFG_NOSHOW = TRAIN_CFG_NOSHOW.replace(
#     search="none", models=["XGBoost"],
#     base_params={"XGBoost": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05}})

display(TRAIN_CFG_NOSHOW.summary())
""")

md("### 4a. No-show prediction")

code("""\
# ── 4a. Train the no-show classifiers ─────────────────────────────────────────
noshow_models, y_train_ns, y_test_ns, X_fit_ns, y_fit_ns = train_binary_classifiers(
    X_train, train_df["NoShow"], X_test, test_df["NoShow"],
    feature_cols, positive_label=1, config=TRAIN_CFG_NOSHOW,
)
""")

code("""\
# ── 4a. What the search actually selected ─────────────────────────────────────
# Grid explored, winning hyperparameters, CV score and timings per family.
display(noshow_models.search_report)
""")

code("""\
# ── 4a. Save the training state (resume without retraining) ─────────────────
save_model_bundle(
    {"models": noshow_models, "y_train": y_train_ns, "y_test": y_test_ns,
     "X_fit": X_fit_ns, "y_fit": y_fit_ns, "config": TRAIN_CFG_NOSHOW},
    name="noshow_training_state",
)
""")

md("**Fast resume point (optional):** tuning four model families plus the fit "
   "on `fit_sample=800_000` rows can take a long time. To skip them when "
   "resuming, do NOT run the training cell above: uncomment and run this one "
   "instead, then continue normally. The configuration used is reloaded along "
   "with the models, which keeps the settings of the saved run on record.")

code("""\
# ── 4a. Fast resume: reload the models instead of retraining ────────────────
# _state_ns = load_model_bundle("noshow_training_state")
# noshow_models, y_train_ns, y_test_ns, X_fit_ns, y_fit_ns = (
#     _state_ns["models"], _state_ns["y_train"], _state_ns["y_test"],
#     _state_ns["X_fit"], _state_ns["y_fit"])
# TRAIN_CFG_NOSHOW = _state_ns.get("config", TRAIN_CFG_NOSHOW)
""")

code("""\
# ── 4a. Comparative evaluation (+ train vs test overfitting check) ───────────
# config=... : the decision threshold follows the configuration's threshold_metric.
# plots=[...] : restricting the figures speeds up tuning iterations
#   (e.g. plots=["metrics"]); eval_sample=200_000 evaluates on a test extract.
results_noshow = evaluate_binary_classification_models(
    noshow_models, X_test, y_test_ns, positive_name="NoShow", reference_name="Attended",
    feature_cols=feature_cols, X_train=X_fit_ns, y_tr=y_fit_ns, prefix="noshow",
    config=TRAIN_CFG_NOSHOW, plots=ALL_EVAL_PLOTS, perm_sample=20_000, eval_sample=None,
)
""")

code("""\
# ── 4a. Record the run (to compare successive settings) ─────────────────────
# Rename the label for each run: it is the lookup key in the comparison table
# of section 4c.
record_training_run("no-show — reference config", results_noshow, TRAIN_CFG_NOSHOW,
                    notes="grid / roc_auc / SMOTE / f1_macro threshold")
""")

md("### 4b. Service time — clustering then classification\n\n"
   "**K-means (K=2):** validated by the elbow method (SAE) and the silhouette "
   "coefficient on a 200k-row sample, confirmed by bootstrap (silhouette "
   "stability across resamples).\n\n"
   "The clustering parameters (candidate K, sample size, bootstrap) are "
   "gathered in the cell below, like the training ones.")

code("""\
# ── 4b. Clustering settings ───────────────────────────────────────────────────
CLUSTER_KS = [2, 3, 4, 5]     # values of K compared (elbow + silhouette)
CLUSTER_SAMPLE = 200_000      # rows used to evaluate K
N_BOOTSTRAP = 10              # resamples validating the silhouette
BOOTSTRAP_SAMPLE = 10_000     # size of each resample
KMEANS_N_INIT = 10            # K-means restarts (solution stability)
FORCE_K = None                # impose K (e.g. 2) instead of the maximum silhouette
""")

code("""\
# ── 4b. Selecting K (elbow + silhouette) ──────────────────────────────────────
sae_results, sil_results, service_time_std, cluster_scaler = evaluate_clustering_ks(
    df_sample, service_time_col="service_time", candidate_ks=CLUSTER_KS,
    sample_size=CLUSTER_SAMPLE, random_state=RANDOM_STATE,
)
best_k = FORCE_K or max(sil_results, key=sil_results.get)
print(f"K selected ({'imposed' if FORCE_K else 'maximum silhouette'}): {best_k}")
""")

code("""\
# ── 4b. Bootstrap validation + A/B labelling ──────────────────────────────────
from sklearn.cluster import KMeans as _KMeans
_labels_full = _KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT).fit_predict(service_time_std)
mu_sil, sd_sil = bootstrap_silhouette(service_time_std, _labels_full,
                                      n_bootstrap=N_BOOTSTRAP, sample_size=BOOTSTRAP_SAMPLE)
print(f"Bootstrapped silhouette (K={best_k}): mean={mu_sil:.4f}, sd={sd_sil:.4f}")

df_sample, cluster_profile, label_map, label_names = fit_clustering_and_label(
    df_sample, best_k, service_time_std, service_time_col="service_time", random_state=RANDOM_STATE,
)
service_time_by_group = df_sample.groupby("group_label")["service_time"].mean().to_dict()
print("Mean service time per group:", service_time_by_group)
""")

code("""\
# ── 4b. Interpretable cutoff (decision tree) ─────────────────────────────────
cluster_tree, primary_cutoff = learn_decision_tree_cutoffs(
    df_sample, service_time_col="service_time", group_col="group_label", label_names=label_names,
)
""")

md("**Group classifier.** Same training function, hence the same levers: "
   "`TRAIN_CFG_GROUP` starts from the no-show configuration and changes only "
   "what needs changing. The A/B target being far less imbalanced than "
   "no-show, SMOTE and class weighting matter less here — that is the setting "
   "to test first.")

code("""\
# ── 4b. Training settings — A/B group classifier ─────────────────────────────
TRAIN_CFG_GROUP = TRAIN_CFG_NOSHOW.replace(
    # e.g. models=["XGBoost", "RandomForest"], use_smote=False, scoring="f1_macro"
)
display(TRAIN_CFG_GROUP.summary())
""")

code("""\
# ── 4b. Group assignment classifier for new patients ─────────────────────────
y_train_grp, y_test_grp = get_group_labels_for_split(df_sample, X_train, X_test)
group_models, y_train_grp_bin, y_test_grp_bin, X_fit_grp, y_fit_grp = train_binary_classifiers(
    X_train, y_train_grp, X_test, y_test_grp,
    feature_cols, positive_label="A", config=TRAIN_CFG_GROUP,
)
display(group_models.search_report)
""")

code("""\
# ── 4b. Save the training state (resume without retraining) ─────────────────
save_model_bundle(
    {"models": group_models, "y_train": y_train_grp_bin, "y_test": y_test_grp_bin,
     "X_fit": X_fit_grp, "y_fit": y_fit_grp, "config": TRAIN_CFG_GROUP},
    name="group_training_state",
)
""")

md("**Fast resume point (optional):** same logic as in 4a — to skip retraining "
   "when resuming, do NOT run the training cell above: uncomment and run this "
   "one instead.")

code("""\
# ── 4b. Fast resume: reload the models instead of retraining ────────────────
# _state_grp = load_model_bundle("group_training_state")
# group_models, y_train_grp_bin, y_test_grp_bin, X_fit_grp, y_fit_grp = (
#     _state_grp["models"], _state_grp["y_train"], _state_grp["y_test"],
#     _state_grp["X_fit"], _state_grp["y_fit"])
# TRAIN_CFG_GROUP = _state_grp.get("config", TRAIN_CFG_GROUP)
""")

code("""\
# ── 4b. Comparative evaluation (+ train vs test overfitting check) ───────────
results_group = evaluate_binary_classification_models(
    group_models, X_test, y_test_grp_bin, positive_name="Group A (long)", reference_name="Group B (short)",
    feature_cols=feature_cols, X_train=X_fit_grp, y_tr=y_fit_grp, prefix="servicetime",
    config=TRAIN_CFG_GROUP, plots=ALL_EVAL_PLOTS, eval_sample=None,
)
record_training_run("A/B group — reference config", results_group, TRAIN_CFG_GROUP)
""")

md("### 4c. Tuning loop — comparing runs\n\n"
   "Three ways to improve the results, from fastest to heaviest:\n\n"
   "1. **Change the decision threshold alone** — no retraining: rerun the "
   "evaluation with `threshold_metric=\"f1_positive\"`.\n"
   "2. **Retrain a single family** with imposed hyperparameters — "
   "`train_single_model(...)`, a few minutes, and the model joins the dict "
   "under an alias so it can be compared with the others.\n"
   "3. **Rerun a full campaign** — edit `TRAIN_CFG_NOSHOW` (4.0) and rerun 4a.\n\n"
   "In all three cases `record_training_run(...)` archives the result and "
   "`training_history_table()` puts the runs side by side. The history is "
   "saved at the end of the section so it can be picked up across sessions.")

code("""\
# ── 4c. (1) Effect of the decision threshold alone — no retraining ──────────
# Compares the recall/precision trade-offs reachable without touching the model.
for _metric in ["f1_macro", "f1_positive", "balanced_accuracy", "youden"]:
    _res = compute_classification_metrics(
        noshow_models, X_test, y_test_ns, "NoShow", "Attended", threshold_metric=_metric)[0]
    record_training_run(f"no-show — {_metric} threshold", _res, TRAIN_CFG_NOSHOW,
                        notes="same models, threshold only")
""")

code("""\
# ── 4c. (2) Single-model variant, imposed hyperparameters ───────────────────
# Adds "XGBoost_v2" to the no-show model dict: the next evaluation compares it
# directly with the models produced by the search.
# train_single_model(
#     "XGBoost", X_fit_ns, y_fit_ns,
#     params={"n_estimators": 600, "max_depth": 6, "learning_rate": 0.03,
#             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10},
#     config=TRAIN_CFG_NOSHOW, models_dict=noshow_models, alias="XGBoost_v2",
# )
# results_noshow = evaluate_binary_classification_models(
#     noshow_models, X_test, y_test_ns, positive_name="NoShow", reference_name="Attended",
#     feature_cols=feature_cols, prefix="noshow", config=TRAIN_CFG_NOSHOW, plots=["metrics"],
# )
# record_training_run("no-show — hand-tuned XGBoost_v2", results_noshow, TRAIN_CFG_NOSHOW)
""")

code("""\
# ── 4c. Comparison table of every recorded run ──────────────────────────────
history = training_history_table(sort_by="AUC")
display(history)
save_dataframe(history, "results/training_history.csv")
print("History saved -> results/training_history.csv")
""")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluation
# ═══════════════════════════════════════════════════════════════════════════
md("## 5. Evaluation\n\n"
   "Metrics on the **test set only** (never used for training or "
   "hyperparameter tuning). The ILP consumes q_i = 1 − P(NoShow) directly: a "
   "well-discriminating but poorly calibrated probability would bias the "
   "optimisation parameters systematically. Calibration (Platt/isotonic) is "
   "therefore evaluated explicitly through the Brier score, on top of the "
   "AUC, precision, recall, F1-score and accuracy already obtained in "
   "section 4.")

code("""\
# ── 5.0 Selection and calibration settings ───────────────────────────────────
SELECTION_METRIC = "AUC"       # criterion for picking the best model, among the
                               # reported metrics: "AUC", "Accuracy", "Precision",
                               # "Recall" (when missing a no-show is costly), "F1-score"
FORCE_NOSHOW_MODEL = None      # impose a model (e.g. "XGBoost_v2") instead of the best one
FORCE_GROUP_MODEL = None
CALIB_METHODS = ("sigmoid", "isotonic")  # calibration methods compared (Brier)
CALIB_FRACTION = 0.5           # share of the test set used to FIT the calibration
CALIB_BINS = 10                # bins of the reliability curves
""")

code("""\
# ── 5.1 Selected no-show model ────────────────────────────────────────────────
best_noshow_name = FORCE_NOSHOW_MODEL or results_noshow.set_index("Model")[SELECTION_METRIC].idxmax()
best_noshow_model = noshow_models[best_noshow_name]
print(f"Best no-show model (test {SELECTION_METRIC}): {best_noshow_name}")
display(results_noshow.set_index("Model").loc[[best_noshow_name]])
""")

code("""\
# ── 5.2 Calibration of the selected no-show model ────────────────────────────
# The test set is split into a calibration subset (fit) and a disjoint
# evaluation subset, to avoid calibrating and evaluating on the same data
# (see the docstring of calibrate_and_select).
calibrated_noshow_model, calib_comparison, X_calib_eval, y_calib_eval = calibrate_and_select(
    best_noshow_model, X_test, y_test_ns, methods=CALIB_METHODS,
    calib_fraction=CALIB_FRACTION, random_state=RANDOM_STATE,
)
raw_prob_eval = best_noshow_model.predict_proba(X_calib_eval)[:, 1]
plot_calibration_curve_(y_calib_eval, raw_prob_eval, f"{best_noshow_name}_raw", n_bins=CALIB_BINS, prefix="calib")

calibrated_prob_eval = calibrated_noshow_model.predict_proba(X_calib_eval)[:, 1]
plot_calibration_curve_(y_calib_eval, calibrated_prob_eval, f"{best_noshow_name}_calibrated", n_bins=CALIB_BINS, prefix="calib")
""")

code("""\
# ── 5.3 Selected group model ──────────────────────────────────────────────────
best_group_name = FORCE_GROUP_MODEL or results_group.set_index("Model")[SELECTION_METRIC].idxmax()
best_group_model = group_models[best_group_name]
print(f"Best group classifier (test {SELECTION_METRIC}): {best_group_name}")
display(results_group.set_index("Model").loc[[best_group_name]])
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
   "imported from `data_driven_plan.py` — the **same** model core as "
   "`01_ilp_validation.ipynb` (regenerated by `notebooks/gen_nb.py`), which "
   "carries the full experimental campaign (illustrative example, design of "
   "experiments, benchmarks B1-B4, out-of-sample simulation, sensitivity "
   "analysis). This section does not duplicate it: it adds the missing step — "
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
# ── 7.3 Build a real ILP instance (one clinic day) ──────────────────────────
N_DEMO = 30  # size of an illustrative day, at realistic scale
demo_patients = scored_inference.sample(n=min(N_DEMO, len(scored_inference)), random_state=RANDOM_STATE)

T_demo = [6]
L_demo = [[30.0, 30.0, 30.0, 60.0, 60.0, 60.0]]
s_A_real = service_time_by_group["A"]
s_B_real = service_time_by_group["B"]

real_instance = build_instance_from_patients(
    demo_patients, T=T_demo, L=L_demo, s_A=s_A_real, s_B=s_B_real,
    c_I=1.0, c_W=0.5, c_O=0.5, group_col="group", q_col="q_i", positive_group="A",
)
print(f"Real instance: n={real_instance.n} patients, {sum(T_demo)} slots "
      f"(max capacity {2*sum(T_demo)} seats), s_A={s_A_real:.1f} min, s_B={s_B_real:.1f} min")
""")

code("""\
# ── 7.4 Solve ─────────────────────────────────────────────────────────────────
real_solution = solve_ilp(real_instance, Config(time_limit=60.0))
print(f"Solve time: {real_solution.solve_time:.3f} s   |   MIP gap: {real_solution.gap:.2e}")
print(f"Z_reduced = {real_solution.Z_reduced:.4f}   Z_complete = {real_solution.Z_complete:.4f}")
print(f"E[Idle]={real_solution.E_idle:.3f}  E[Wait]={real_solution.E_wait:.3f}  E[OT]={real_solution.E_OT:.3f}")
print(f"Patients served: {real_solution.n_served}/{real_instance.n}   "
      f"Double-booked slots: {real_solution.n_double}/{sum(T_demo)}")
""")

md("**Full experimental campaign** (design of experiments with distributional "
   "notation and factor levels, benchmarks B1-B4, out-of-sample simulation, "
   "sensitivity analysis to the no-show rate / variance / load with percentage "
   "deviation and solve time): see the companion notebook "
   "**`01_ilp_validation.ipynb`**, regenerated by `notebooks/gen_nb.py` and "
   "not included in this repository. It imports its model core "
   "(`Instance`/`Config`/`solve_ilp`) from `data_driven_plan.py`, so both "
   "solve exactly the same model.")

# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════
md("## Final summary")

code("""\
# ── Summary table ─────────────────────────────────────────────────────────────
recap = pd.DataFrame([
    {"Item": "Split", "Value": f"{len(df_sample):,} rows (full dataset) | "
     f"train={len(train_df):,} test={len(test_df):,} inference={len(df_inference):,}"},
    {"Item": "No-show — best model", "Value": best_noshow_name},
    {"Item": "No-show — AUC (test)",
     "Value": f"{results_noshow.set_index('Model').loc[best_noshow_name, 'AUC']:.4f}"},
    {"Item": "No-show — Brier (calibrated)",
     "Value": f"{calib_comparison.set_index('Method').loc[calib_comparison.iloc[0]['Method'], 'Brier']:.4f}"},
    {"Item": "service_time clustering — selected K", "Value": str(best_k)},
    {"Item": "Clustering — bootstrapped silhouette", "Value": f"mean={mu_sil:.4f}, sd={sd_sil:.4f}"},
    {"Item": "Cutoff (decision tree)", "Value": f"{primary_cutoff:.1f} min"},
    {"Item": "Group — best classifier", "Value": best_group_name},
    {"Item": "Group — AUC (test)",
     "Value": f"{results_group.set_index('Model').loc[best_group_name, 'AUC']:.4f}"},
    {"Item": "ILP — real instance (n patients)", "Value": str(real_instance.n)},
    {"Item": "ILP — Z_complete", "Value": f"{real_solution.Z_complete:.4f}"},
    {"Item": "ILP — solve time", "Value": f"{real_solution.solve_time:.3f} s"},
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
