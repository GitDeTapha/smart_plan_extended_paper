"""
data_driven_plan.py

Reusable functions for the Cluster-Predict-Schedule pipeline (data-driven
outpatient appointment scheduling).

Covers the full workflow in eight sections, following the CRISP-DM order of
data_driven_plan.ipynb: exploration, data preparation, no-show prediction,
service-time clustering and classification, probability calibration,
deployment artifacts, and the stochastic overbooking ILP.

Usage: from data_driven_plan import *
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, silhouette_score, brier_score_loss,
    f1_score, balanced_accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, classification_report,
    accuracy_score, precision_score, recall_score,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print

# xgboost and imbalanced-learn are hard requirements: the published study (Fall et al.)
# uses these libraries, and a silent substitute (HistGradientBoosting, no SMOTE) would
# change the models being reported without saying so.
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import pyomo.environ as pyo

warnings.filterwarnings("ignore")

# ============================================================================
# 0. Configuration, reproducibility, plotting style
# ============================================================================

RANDOM_STATE = 0

FIGURES_DIR = "figures"
MODELS_DIR = "models"
TABLES_DIR = "tables"
RESULTS_DIR = "results"

# Data root, overridable so the same notebook runs unchanged on a workstation
# and on a remote machine where the dataset sits elsewhere:
#     export SMART_PLAN_DATA=/workspace/data
DATA_DIR = os.environ.get("SMART_PLAN_DATA", "data")
RAW_PARQUET_DIR = os.path.join(DATA_DIR, "raw_data_parquet")
PREPROCESSED_PATH = os.path.join(DATA_DIR, "preprocessed", "preprocessed_data.parquet")
INFERENCE_DIR = os.path.join(DATA_DIR, "inference")

# ----------------------------------------------------------------------------
# Figure typography, single source of truth
# ----------------------------------------------------------------------------
# Every figure is meant for a journal submission, so sizes are set once here and
# never hard-coded inside a plotting function: annotations that used to be fixed
# at 9 pt became unreadable once a figure was scaled down to column width.
# Sizes are expressed relative to FONT_BASE, so the whole set scales together.
FONT_BASE = 15          # axis tick labels and general text
FS_TITLE = 17           # axes titles
FS_LABEL = 16           # axis labels
FS_TICK = 14            # tick labels
FS_LEGEND = 14          # legend entries
FS_ANNOT = 13           # in-figure annotations (bar values, matrix cells)
FS_ANNOT_SMALL = 11     # secondary annotations (counts under percentages)
FS_TREE = 12            # decision tree node text

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    # DejaVu Sans first: it ships with matplotlib, so figures render identically
    # on a workstation and on a headless server (no silent font substitution).
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "Helvetica"],
    "font.size": FONT_BASE,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND,
    "figure.titlesize": FS_TITLE + 1,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.35,
    "text.color": "#000000",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # TrueType rather than Type 3: several journals reject Type 3 fonts, and
    # Type 3 text cannot be searched or copied in the published PDF.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def ensure_output_dirs() -> None:
    """Create the standard output directories if they do not exist."""
    for d in (FIGURES_DIR, MODELS_DIR, TABLES_DIR, RESULTS_DIR, INFERENCE_DIR):
        os.makedirs(d, exist_ok=True)


def _savefig(name: str, prefix: str = "") -> None:
    """Save the current figure to figures/ as vector PDF and 300 dpi PNG."""
    ensure_output_dirs()
    stem = f"{prefix}_{name}" if prefix else name
    plt.savefig(f"{FIGURES_DIR}/{stem}.pdf", bbox_inches="tight")
    plt.savefig(f"{FIGURES_DIR}/{stem}.png", dpi=300, bbox_inches="tight")


print(f"xgboost {__import__('xgboost').__version__} | imbalanced-learn {__import__('imblearn').__version__}")


# ----------------------------------------------------------------------------
# Authors' decisions (declared at the top of both notebooks, papermill-overridable)
# ----------------------------------------------------------------------------
ALLOWED_DECISIONS = {
    "DURATION_METHOD": ("kmedians", "kmeans"),   # kmedians: manuscript; kmeans: ablation
    "DURATION_STAT": ("median", "mean"),          # centroid statistic giving s_A, s_B
    "SPLIT_MODE": ("stratified", "patient_group"),
}


def validate_decisions(DURATION_METHOD: str, DURATION_STAT: str, C_O_BASE: float,
                       SLOT_LENGTHS: Sequence[float], SPLIT_MODE: str, RANDOM_STATE: int,
                       SMOKE: bool) -> dict:
    """Check the authors' decisions and return them as a dict for logging.

    Raises:
        ValueError: on any value outside its allowed set. There is no default to
            fall back to: a typo must stop the run, not change the method silently.
    """
    given = {"DURATION_METHOD": DURATION_METHOD, "DURATION_STAT": DURATION_STAT,
             "SPLIT_MODE": SPLIT_MODE}
    for key, value in given.items():
        if value not in ALLOWED_DECISIONS[key]:
            raise ValueError(f"{key}={value!r} not in {ALLOWED_DECISIONS[key]}")
    if DURATION_METHOD == "kmedians" and DURATION_STAT != "median":
        raise ValueError("K-medians centroids are medians: DURATION_STAT must be 'median'")
    if C_O_BASE < 0:
        raise ValueError(f"C_O_BASE must be non-negative, got {C_O_BASE}")
    slots = tuple(float(v) for v in SLOT_LENGTHS)
    if len(slots) != 2 or min(slots) <= 0 or slots[0] == slots[1]:
        raise ValueError(f"SLOT_LENGTHS must be two distinct positive lengths (short, long), got {SLOT_LENGTHS}")
    if not isinstance(RANDOM_STATE, (int, np.integer)):
        raise ValueError(f"RANDOM_STATE must be an integer, got {RANDOM_STATE!r}")
    return {**given, "C_O_BASE": float(C_O_BASE), "SLOT_LENGTHS": slots,
            "RANDOM_STATE": int(RANDOM_STATE), "SMOKE": bool(SMOKE)}


# ============================================================================
# 1. Generic utilities: I/O, master split, feature preparation
# ============================================================================

def load_preprocessed_data(path: str) -> pd.DataFrame:
    """Load the preprocessed Parquet file produced by the Data Preparation phase.

    Parquet rather than CSV preserves the optimised dtypes (category, uint32,
    float32...): a CSV round-trip would restore them as object/int64/float64,
    multiplying the memory footprint of a 40M+ row dataset.

    Args:
        path: path to the Parquet file.

    Returns:
        Loaded DataFrame.
    """
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")


def save_preprocessed_data(df: pd.DataFrame, path: str) -> None:
    """Save the preprocessed DataFrame as Parquet, creating the parent directory
    if needed (see `load_preprocessed_data` for why Parquet).

    Args:
        df: DataFrame to save.
        path: output path (directory created automatically).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as CSV, creating the parent directory if needed.

    Intended for small exports meant to be read as-is (summary tables,
    aggregated results), use `save_preprocessed_data` (Parquet) to checkpoint
    the large preprocessed DataFrame.

    Args:
        df: DataFrame to save.
        path: output path (directory created automatically).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def save_model_bundle(bundle: dict, name: str) -> None:
    """Serialise a dict of arbitrary Python objects (trained models, fit
    subsample, binarised targets) into a single joblib file, so the notebook
    can be resumed without retraining everything.

    Args:
        bundle: dict {key: object}, serialised as-is.
        name: file name (without extension) inside MODELS_DIR.
    """
    ensure_output_dirs()
    path = f"{MODELS_DIR}/{name}.joblib"
    joblib.dump(bundle, path)
    print(f"Saved -> {path}  ({list(bundle.keys())})")


def load_model_bundle(name: str) -> dict:
    """Reload a bundle serialised by `save_model_bundle`.

    Args:
        name: file name (without extension) inside MODELS_DIR.

    Returns:
        dict {key: object}, identical to the one passed to `save_model_bundle`.
    """
    path = f"{MODELS_DIR}/{name}.joblib"
    bundle = joblib.load(path)
    print(f"Loaded <- {path}  ({list(bundle.keys())})")
    return bundle


def inspect_dataframe(df: pd.DataFrame, target_col: str) -> None:
    """Print a standard DataFrame summary (shape, dtypes, describe, missing values).

    Args:
        df: DataFrame to inspect.
        target_col: target column described in more detail (histogram + boxplot).
    """
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    display(df.head())
    df.info()
    print(df[target_col].describe())
    print("Missing values:\n", df.isna().sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.histplot(df[target_col], bins=80, kde=True, ax=axes[0], color=PLOT_COLORS[0])
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Frequency")
    sns.boxplot(x=df[target_col], ax=axes[1], color=PLOT_COLORS[1])
    axes[1].set_xlabel(target_col)
    sns.despine()
    _savefig(f"distribution_{target_col}")
    plt.show()


def make_master_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    test_size: float = 0.20,
    inference_size: float = 0.10,
    sample_size: Optional[int] = None,
    noshow_col: str = "NoShow",
    service_time_col: str = "service_time",
    n_quantile_bins: int = 10,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single leak-proof partition, shared by both ML pipelines.

    By default the whole DataFrame is partitioned (no row discarded) into three
    disjoint sets whose sizes match train_size/test_size/inference_size of the
    total EXACTLY: an inference holdout (never seen by the models, reserved for
    the ILP), a train set (training + cross-validation) and a test set (final
    evaluation). Stratification combines the no-show target (strongly
    imbalanced, ~3%) with quantile bins of service time, so that BOTH pipelines
    (no-show and service time) inherit a representative split without either
    seeing the other's data.

    Args:
        df: full preprocessed DataFrame.
        train_size: fraction of the total assigned to training.
        test_size: fraction of the total assigned to testing.
        inference_size: fraction of the total reserved for the ILP holdout.
        sample_size: if set, a random subsample of this size is drawn from df
            before splitting (useful during development/debugging). Left at
            None, the whole of df is used.
        noshow_col: name of the binary no-show target column.
        service_time_col: name of the continuous service-time target column.
        n_quantile_bins: number of service_time quantile bins used in the
            combined stratification key.
        random_state: seed for reproducibility.

    Returns:
        (df_sample, train_df, test_df, df_inference), three disjoint DataFrames
        (train/test/inference), df_sample being their union.
    """
    if abs((train_size + test_size + inference_size) - 1.0) > 1e-9:
        raise ValueError(
            f"train_size + test_size + inference_size must equal 1.0 "
            f"(got {train_size} + {test_size} + {inference_size} = "
            f"{train_size + test_size + inference_size})")

    full_n = len(df)
    if sample_size is not None and sample_size < full_n:
        df_s = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"Working sample: {sample_size:,} / {full_n:,} rows (~{sample_size/full_n:.1%})")
    else:
        # reset_index() without inplace already builds a new DataFrame: a prior
        # .copy() would needlessly duplicate the 40M+ rows in memory at the
        # most memory-constrained point of the pipeline.
        df_s = df.reset_index(drop=True)
        print(f"Full dataset used as working sample: {full_n:,} rows")

    st_bin = pd.qcut(df_s[service_time_col], q=n_quantile_bins, duplicates="drop").astype(str)
    strat_key = df_s[noshow_col].astype(str) + "_" + st_bin

    df_inf, df_mod = train_test_split(
        df_s, test_size=(1.0 - inference_size), random_state=random_state, stratify=strat_key)
    strat_mod = strat_key.loc[df_mod.index]
    # test_size/train_size are fractions of the TOTAL df_s; convert them into a
    # fraction of the post-holdout remainder df_mod to hit the target
    # proportions exactly, instead of a cascading approximation (e.g. 72/18
    # instead of 70/20).
    test_frac_of_remainder = test_size / (train_size + test_size)
    train_df, test_df = train_test_split(
        df_mod, test_size=test_frac_of_remainder, random_state=random_state, stratify=strat_mod)

    n_inf, n_tr, n_te = len(df_inf), len(train_df), len(test_df)
    total = n_inf + n_tr + n_te
    print(f"  Inference holdout : {n_inf:>8,}  ({n_inf/total:.1%})  -- never seen by the models")
    print(f"  Train             : {n_tr:>8,}  ({n_tr/total:.1%})")
    print(f"  Test              : {n_te:>8,}  ({n_te/total:.1%})")
    for name, d in [("Inference", df_inf), ("Train", train_df), ("Test", test_df)]:
        rate = d[noshow_col].mean()
        print(f"  NoShow rate ({name:9s}) : {rate:.4f}  |  median service_time: {d[service_time_col].median():.1f} min")

    ensure_output_dirs()
    inf_path = f"{INFERENCE_DIR}/inference_set_raw.csv"
    df_inf.to_csv(inf_path, index=False)
    print(f"  Inference holdout saved -> {inf_path}  (seed={random_state}, stratified NoShow x service_time)")

    # Indices of train_df/test_df/df_inf are deliberately KEPT (subsets of the
    # df_s index, not reset) so rows can later be aligned with df_s via .loc[]
    # (e.g. get_group_labels_for_split).
    return df_s, train_df, test_df, df_inf
def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inference_df: Optional[pd.DataFrame] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    excluded_cols: Optional[set[str]] = None,
    random_state: int = RANDOM_STATE,
    onehot_max_cardinality: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], list[str], StandardScaler, dict]:
    """Encoding and standardisation shared by both ML pipelines.

    The scaler and categorical encodings are fitted on the training set ONLY,
    then applied to test and inference, no leakage between sets. The targets
    (NoShow, service_time, group_label) are excluded from the features and
    extracted separately by each pipeline. `discontinued` is excluded too:
    derived from the same source column `disp` as `NoShow`
    (`decompose_disp_column`), it is not available before the appointment.
    `completed` is no longer produced at all by `decompose_disp_column`, it
    was what made the target reconstructible (NoShow ~ 1 - completed -
    discontinued) and caused massive leakage (ROC-AUC = 1.0 for every model,
    including families as different as a random forest and an MLP); it is kept
    in the default exclusion set as a safeguard, in case a preprocessed Parquet
    from an earlier run (notebook section 3.2bis) still contains it.

    `dts` (actual service start) and `dta` (waiting time before the
    appointment) are excluded for a related reason, real-world availability
    rather than pure leakage: they are realised measurements, known only once
    the visit is under way or finished, hence unavailable when the prediction
    must actually be made (before the appointment, to decide on overbooking).
    `dtc` is already dropped in `compute_service_time`, but `dts`/`dta`
    survived in the DataFrame until this point.

    Args:
        train_df: training partition (modified in place for the encoded and
            standardised columns).
        test_df: test partition (idem).
        inference_df: optional ILP holdout partition (idem).
        categorical_cols: categorical columns to encode (integer labels).
        excluded_cols: target columns to exclude from the features.
        random_state: seed (unused here, kept for API consistency with the rest
            of the module).
        onehot_max_cardinality: categorical columns with at most this many
            levels are one-hot encoded; the others keep an integer code. A tree
            splits an integer code space freely and recovers the groups, but a
            neural network reads it as an ordered quantity, zip code 437 close
            to 438 and far from 900, which is meaningless. One-hot fixes that,
            at the cost of one column per level, so it is worth it only while
            the cardinality stays small: `zip` alone would add 919 columns.

    Returns:
        (X_train, X_test, X_inference, feature_cols, scaler, cat_encodings)
    """
    if categorical_cols is None:
        categorical_cols = ["stopcode", "sta3n", "non_va", "state", "zip"]
    if excluded_cols is None:
        excluded_cols = {"service_time", "NoShow", "group_label", "cluster",
                          "completed", "discontinued", "dts", "dta"}

    cat_cols = [c for c in categorical_cols if c in train_df.columns]
    dfs = [train_df, test_df] + ([inference_df] if inference_df is not None else [])

    for col in ("zip", "state"):
        if col in cat_cols:
            for d in dfs:
                d[col] = d[col].fillna("MISSING").astype(str)
    for col in cat_cols:
        if train_df[col].dtype != object:
            for d in dfs:
                d[col] = d[col].astype(str)

    features = [c for c in train_df.columns if c not in excluded_cols]
    numeric_cols = [c for c in features if c not in cat_cols]

    cat_encodings = {}
    for col in cat_cols:
        cats = pd.Categorical(train_df[col], ordered=False).categories
        mapping = {c: i for i, c in enumerate(cats)}
        cat_encodings[col] = mapping
        for d in dfs:
            d[col] = d[col].map(mapping).fillna(-1).astype(int)

    onehot_cols = [c for c in cat_cols if len(cat_encodings[c]) <= onehot_max_cardinality]
    ordinal_cols = [c for c in cat_cols if c not in onehot_cols]
    n_dummies = sum(len(cat_encodings[c]) for c in onehot_cols)
    print(f"Features: {len(features)} | Numeric: {len(numeric_cols)} | "
          f"Categorical one-hot: {len(onehot_cols)} ({n_dummies} columns) | "
          f"Categorical ordinal: {len(ordinal_cols)}")
    for c in onehot_cols:
        print(f"   one-hot  {c:<10} {len(cat_encodings[c]):>4} levels")
    for c in ordinal_cols:
        print(f"   ordinal  {c:<10} {len(cat_encodings[c]):>4} levels (too many to expand)")

    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    for d in [test_df] + ([inference_df] if inference_df is not None else []):
        d[numeric_cols] = scaler.transform(d[numeric_cols])

    def _design_matrix(df):
        """Numeric + ordinal columns, plus one dummy per level of the low
        cardinality categoricals. float32 halves the memory of a matrix that is
        mostly zeros once expanded."""
        if df is None:
            return None
        base = df[numeric_cols + ordinal_cols].astype(np.float32)
        if not onehot_cols:
            return base
        parts = [base]
        for col in onehot_cols:
            codes = df[col].to_numpy()
            for level, idx in cat_encodings[col].items():
                parts.append(pd.Series((codes == idx).astype(np.float32),
                                       index=df.index, name=f"{col}={level}"))
        return pd.concat(parts, axis=1)

    X_train = _design_matrix(train_df)
    X_test = _design_matrix(test_df)
    X_inf = _design_matrix(inference_df)
    feature_cols = list(X_train.columns)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}", end="")
    if X_inf is not None:
        print(f" | X_inference: {X_inf.shape}", end="")
    print()
    return X_train, X_test, X_inf, feature_cols, scaler, cat_encodings


# ============================================================================
# 2. Data Understanding (EDA)
# ============================================================================

def display_missing_values(
    data: pd.DataFrame,
    sentinel_col: Optional[str] = None,
    sentinel_value: Optional[str] = None,
) -> pd.Series:
    """Percentage of missing values per column, optionally also counting a
    sentinel value (e.g. the '*Missing*' string used for `state`).

    Args:
        data: DataFrame to analyse.
        sentinel_col: column in which a sentinel value is also counted.
        sentinel_value: sentinel value added to the `sentinel_col` count.

    Returns:
        Series of missing-value percentages per column.
    """
    missing = data.isnull().sum()
    if sentinel_col is not None and sentinel_col in data.columns:
        missing[sentinel_col] += (data[sentinel_col] == sentinel_value).sum()
    pct = (missing / len(data)) * 100
    print("Percentage of missing values per column:")
    return pct


def detect_outliers_iqr(data: pd.DataFrame, feature: str) -> pd.Index:
    """Indices of a column's outliers under the IQR rule (1.5 x IQR).

    Args:
        data: source DataFrame.
        feature: numeric column to analyse.

    Returns:
        Index of the rows flagged as outliers.
    """
    q1, q3 = np.percentile(data[feature], [25, 75])
    iqr = q3 - q1
    step = 1.5 * iqr
    return data[(data[feature] < q1 - step) | (data[feature] > q3 + step)].index


def visualize_outliers(data: pd.DataFrame, prefix: str = "") -> None:
    """Diagnostic of IQR outliers per numeric column (visualisation only, no row
    is removed).

    Args:
        data: DataFrame to analyse.
        prefix: file-name prefix for the saved figure.
    """
    counts = {c: len(detect_outliers_iqr(data, c)) for c in data.select_dtypes(include=[np.number]).columns}
    print("Outlier count (IQR) per numeric column:", counts)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette=PLOT_COLORS[:len(counts)], ax=ax)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of outliers")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig("outliers_distribution", prefix)
    plt.show()


def plot_target_balance(data: pd.DataFrame, target_col: str, labels: Optional[dict] = None, prefix: str = "") -> None:
    """Pie chart of the class distribution of a categorical/binary target.

    Args:
        data: source DataFrame.
        target_col: target column to plot.
        labels: optional mapping used to rename categories for display.
        prefix: file-name prefix.
    """
    counts = data[target_col].value_counts()
    if labels:
        counts.index = [labels.get(i, i) for i in counts.index]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=PLOT_COLORS[:len(counts)], textprops={"fontsize": FS_ANNOT})
    ax.axis("equal")
    _savefig(f"balance_{target_col}", prefix)
    plt.show()


def plot_numeric_histograms(
    data: pd.DataFrame, columns: Sequence[str], bins: Sequence[int], xlims: Sequence[tuple], prefix: str = ""
) -> None:
    """Stacked histograms for a list of numeric variables.

    Args:
        data: source DataFrame.
        columns: columns to plot.
        bins: number of bins per column (same length as columns).
        xlims: (xmin, xmax) bounds per column (same length as columns).
        prefix: file-name prefix.
    """
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(10, 4 * len(columns)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, col, b, xlim in zip(axes, columns, bins, xlims):
        sns.histplot(data=data, x=col, bins=b, color=PLOT_COLORS[0], edgecolor="black", linewidth=0.5, ax=ax)
        ax.set_xlabel(f"{col} (minutes)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
    _savefig("histograms_numeric", prefix)
    plt.show()


def analyze_temporal_patterns(data: pd.DataFrame, datetime_col: str = "activitydatetime", prefix: str = "") -> pd.DataFrame:
    """Patient volume by weekday and by hour (computed on a copy; the input
    DataFrame is left unchanged).

    Args:
        data: source DataFrame.
        datetime_col: datetime column to decompose.
        prefix: file-name prefix.

    Returns:
        Copy of `data` with the temporary `hour`/`weekday` columns (useful for
        inspection, not kept in the final pipeline).
    """
    tmp = data.copy()
    tmp[datetime_col] = pd.to_datetime(tmp[datetime_col])
    tmp["hour"] = tmp[datetime_col].dt.hour
    tmp["weekday"] = tmp[datetime_col].dt.weekday

    weekday_counts = tmp["weekday"].value_counts().sort_index()
    hour_counts = tmp["hour"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values, color=PLOT_COLORS[0], ax=ax)
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of patients")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.grid(True, axis="y", alpha=0.3)
    _savefig("working_patterns_weekday", prefix)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.barplot(x=hour_counts.index, y=hour_counts.values, color=PLOT_COLORS[0], ax=ax)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of patients")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig("working_patterns_hour", prefix)
    plt.show()
    return tmp


# ============================================================================
# 3. Data Preparation
# ============================================================================

def decompose_disp_column(data: pd.DataFrame, disp_column: str = "disp") -> pd.DataFrame:
    """Decompose the `disp` column into the binary indicators NoShow/discontinued.

    The `completed` indicator (COMPLETE/UPDATE) is deliberately NOT created:
    combined with `discontinued` it reconstructs the target (NoShow ~ 1 -
    completed - discontinued) and biased no-show prediction whenever it
    survived into the feature matrix. Never producing it removes the risk at
    the source rather than relying on a downstream exclusion. No information is
    lost: `NoShow` already encodes CANCELLED and `discontinued` the third
    state, so COMPLETE/UPDATE follows from the two.

    `discontinued` (appointment cancelled by the clinic) is kept: it is an
    operational event distinct from patient absence, and on its own it does not
    recover `NoShow` (DISCONTINUED=0 covers CANCELLED, COMPLETE/UPDATE and EU).
    It is nevertheless excluded from the features (see `prepare_features`),
    being derived from the same source column.

    Args:
        data: source DataFrame (the `disp` column is dropped on output).
        disp_column: name of the column to decompose.

    Returns:
        Enriched DataFrame.
    """
    data["NoShow"] = data[disp_column].map(
        {"DISCONTINUED": 0, "CANCELLED": 1, "COMPLETE/UPDATE": 0, "EU": 0}).astype("int8")
    data["discontinued"] = data[disp_column].map(
        {"DISCONTINUED": 1, "CANCELLED": 0, "COMPLETE/UPDATE": 0, "EU": 0}).astype("int8")
    data.drop(disp_column, axis=1, inplace=True)
    return data


def filter_dates_after(data: pd.DataFrame, datetime_col: str, cutoff_date: str) -> pd.DataFrame:
    """Keep the rows posterior to a cutoff date.

    Args:
        data: source DataFrame.
        datetime_col: datetime column to filter on.
        cutoff_date: cutoff date ('YYYY-MM-DD HH:MM:SS').

    Returns:
        Filtered DataFrame.
    """
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        data[datetime_col] = pd.to_datetime(data[datetime_col])
    return data[data[datetime_col] > pd.to_datetime(cutoff_date)]


def add_temporal_features(data: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Add calendar features (weekday, day of month, month, hour).

    Args:
        data: source DataFrame.
        datetime_column: source datetime column.

    Returns:
        Enriched DataFrame.
    """
    data[datetime_column] = pd.to_datetime(data[datetime_column])
    data["day_of_week"] = data[datetime_column].dt.dayofweek.astype("uint8")
    data["day_of_month"] = data[datetime_column].dt.day.astype("uint8")
    data["month"] = data[datetime_column].dt.month.astype("uint8")
    data["hour"] = data[datetime_column].dt.hour.astype("uint8")
    return data


def drop_high_missing_columns(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Drop columns too sparsely populated to be imputed reliably.

    Args:
        data: source DataFrame.
        columns: columns to drop when present.

    Returns:
        DataFrame without those columns.
    """
    return data.drop(columns=[c for c in columns if c in data.columns])


def drop_missing_rows(data: pd.DataFrame, columns: Sequence[str], missing_sentinels: Optional[dict] = None) -> pd.DataFrame:
    """Listwise deletion of rows whose critical column is missing (true NaN or a
    sentinel value such as '*Missing*').

    Justified here by the small share of affected rows relative to the size of
    the dataset, which makes imputation unnecessary.

    Args:
        data: source DataFrame.
        columns: columns to check.
        missing_sentinels: optional dict {column: sentinel_value} in addition to NaN.

    Returns:
        Filtered DataFrame.
    """
    missing_sentinels = missing_sentinels or {}
    idx_to_drop = pd.Index([])
    for col in columns:
        idx_to_drop = idx_to_drop.union(data[data[col].isnull()].index)
        if col in missing_sentinels:
            idx_to_drop = idx_to_drop.union(data[data[col] == missing_sentinels[col]].index)
    return data.drop(index=idx_to_drop)


def filter_dts_dtc_consistency(data: pd.DataFrame, dts_col: str = "dts", dtc_col: str = "dtc", noshow_col: str = "NoShow") -> pd.DataFrame:
    """Remove records whose service end time (`dtc`) precedes its start time
    (`dts`), a data-entry inconsistency that would invalidate the service-time
    computation.

    For the rare NoShow=1 rows with dts>dtc, the service end time is meaningless
    in principle (the patient was never seen): they are treated like the other
    inconsistent rows rather than kept, so that `service_time` stays
    interpretable only for consultations that actually took place.

    Args:
        data: source DataFrame.
        dts_col: service start time column.
        dtc_col: service end time column.
        noshow_col: no-show target column (not used for filtering, kept for a
            signature consistent with the original logic).

    Returns:
        DataFrame containing only rows with dts <= dtc.
    """
    return data[data[dts_col] <= data[dtc_col]]


def compute_service_time(data: pd.DataFrame, dts_col: str = "dts", dtc_col: str = "dtc") -> pd.DataFrame:
    """Derive `service_time` = dtc - dts (justified by the strong collinearity
    observed between dts and dtc), then drop dtc.

    Args:
        data: source DataFrame.
        dts_col: start time column.
        dtc_col: end time column (dropped on output).

    Returns:
        DataFrame with the added `service_time` column.
    """
    data["service_time"] = data[dtc_col] - data[dts_col]
    data.drop(columns=[dtc_col], inplace=True)
    return data


def separate_stopcodes_by_no_show(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Split the data according to whether a stopcode has ever recorded a no-show.

    Args:
        data: source DataFrame (must contain 'NoShow' and 'stopcode').

    Returns:
        (data_with_no_show, data_without_no_show, stopcodes_without_no_show)
    """
    no_shows = data[data["NoShow"] == 1]
    codes_with = no_shows["stopcode"].unique()
    with_ns = data[data["stopcode"].isin(codes_with)]
    without_ns = data[~data["stopcode"].isin(codes_with)]
    return with_ns, without_ns, without_ns["stopcode"].unique()


def add_patient_history_features(
    data: pd.DataFrame,
    patient_col: str = "patientsid",
    stopcode_col: str = "stopcode",
    datetime_col: str = "activitydatetime",
    noshow_col: str = "NoShow",
    service_time_col: str = "service_time",
) -> pd.DataFrame:
    """Patient history features: days since the previous visit (per
    patient/stopcode), prior no-show rate and prior average service time (per
    patient), all lagged by one rank (`shift(1)`) to exclude any information
    leaking from the current visit.

    Args:
        data: source DataFrame; it need not be sorted beforehand, sorting is
            done internally.
        patient_col: patient identifier.
        stopcode_col: specialty/clinic code.
        datetime_col: visit datetime column.
        noshow_col: binary no-show target.
        service_time_col: service time.

    Returns:
        DataFrame with the added `days_between_visits`, `prior_noShow_rate` and
        `prior_service_time_avg` columns.
    """
    data = data.sort_values([patient_col, stopcode_col, datetime_col])
    data["days_between_visits"] = (
        data.groupby([patient_col, stopcode_col])[datetime_col].diff().dt.days
        .fillna(0).astype("uint16")
    )

    data = data.sort_values([patient_col, datetime_col])
    grp = data.groupby(patient_col)[noshow_col]
    cum_sum = grp.cumsum()
    cum_count = grp.cumcount() + 1
    data["prior_noShow_rate"] = (cum_sum / cum_count).groupby(data[patient_col]).shift(1).fillna(0)

    grp_st = data.groupby(patient_col)[service_time_col]
    cum_sum_st = grp_st.cumsum()
    data["prior_service_time_avg"] = (cum_sum_st / cum_count).groupby(data[patient_col]).shift(1).fillna(0)

    return data


# ============================================================================
# 4. Generic binary modelling, shared by the no-show AND A/B group pipelines
# ============================================================================
#
# Class encoding convention:
#   y_bin = 1  -> class of interest (rare event: no-show, or "long" group A)
#   y_bin = 0  -> reference class (majority)
#
# ----------------------------------------------------------------------------
# 4.0 Exposed settings
# ----------------------------------------------------------------------------
# Everything that drives training is declared here in the open, parameter
# grids, fixed hyperparameters, search strategy, subsample sizes, class
# rebalancing, decision threshold, and is overridden from the notebook through
# `TrainingConfig`, without editing this module:
#
#     cfg = TrainingConfig(models=["XGBoost"], search="random", n_iter=30)
#     cfg = cfg.replace(param_grids={"XGBoost": {"max_depth": [3, 6, 10]}})
#     models, y_tr, y_te, X_fit, y_fit = train_binary_classifiers(..., config=cfg)
#     display(models.search_report)   # what the search actually selected
# ----------------------------------------------------------------------------

SUPPORTED_MODELS: tuple[str, ...] = ("RandomForest", "XGBoost", "AdaBoost", "ANN")

#: Grids explored by the hyperparameter search. Overridable per model through
#: `TrainingConfig(param_grids={...})`, an override replaces the grid of that
#: model only, the others keep their default.
DEFAULT_PARAM_GRIDS: dict[str, dict] = {
    # Depth is capped and a minimum leaf size imposed: left unconstrained, the
    # forest reached a perfect training score (AUC 1.00 train against 0.73 test)
    # by isolating individual rows. min_samples_leaf is what forbids that.
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [20, 50, 200],
        "max_features": ["sqrt", 0.3],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.6, 0.8],
        "min_child_weight": [5, 20, 50],
        "reg_lambda": [1.0, 10.0],
    },
    "AdaBoost": {
        "n_estimators": [100, 300],
        "learning_rate": [0.5, 1.0],
    },
    # The ANN is searched over a wider space than the other families: it is the
    # reference model of the study, and the narrow (64,32)/(128,64) grid used
    # previously left it far behind the tree ensembles.
    "ANN": {
        "hidden_layer_sizes": [(128, 64), (256, 128)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-3, 3e-4],
    },
}

#: FIXED hyperparameters (not searched) of each family. Overridable through
#: `TrainingConfig(base_params={...})`; a key given there wins over the values
#: set by the module (including `class_weight` / `scale_pos_weight`), which
#: makes it possible to switch a specific behaviour off.
DEFAULT_BASE_PARAMS: dict[str, dict] = {
    "RandomForest": {},
    "XGBoost": {"eval_metric": "logloss", "verbosity": 0},
    "AdaBoost": {},
    "ANN": {"solver": "adam", "max_iter": 300, "early_stopping": True,
            "validation_fraction": 0.1, "n_iter_no_change": 25,
            "batch_size": 512, "learning_rate": "adaptive"},
}

#: Criteria available for choosing the decision threshold on P(positive class).
THRESHOLD_METRICS: tuple[str, ...] = (
    "f1_macro",           # F1 averaged over both classes (historical default)
    "f1_positive",        # F1 of the class of interest only (favours no-show recall)
    "balanced_accuracy",  # mean of the per-class recalls
    "youden",             # J = TPR - FPR (optimal point of the ROC curve)
    "fixed",              # user-imposed threshold (`fixed_threshold`)
)

SEARCH_STRATEGIES: tuple[str, ...] = ("grid", "random", "none")

#: Families rebalanced by resampling inside the fit rather than by class
#: weights, because neither accepts a weight: the MLP has no such parameter and
#: AdaBoost breaks when its base learner carries one.
SMOTE_BALANCED_MODELS: tuple[str, ...] = ("ANN", "AdaBoost")
PIPE_STEP = "clf"   # name of the classifier step inside the resampling pipeline

# Reported metric set, fixed by the paper's conventions: AUC, accuracy,
# precision, recall, F1-score. Precision, recall and F1 are MACRO-averaged
# (unweighted mean over both classes), so the rare class weighs as much as the
# majority one; accuracy and AUC are computed over both classes as usual.
AUC_COL = "AUC"
ACC_COL = "Accuracy"
PREC_COL = "Precision"
REC_COL = "Recall"
F1_COL = "F1-score"
THR_COL = "Threshold"
REPORTED_METRICS = [AUC_COL, ACC_COL, PREC_COL, REC_COL, F1_COL]

#: Decimals used for every reported metric, in tables and in figures. Two is
#: what the paper reports, and it stops bar labels from overlapping.
METRIC_DECIMALS = 2


@dataclass
class TrainingConfig:
    """Control panel for binary classifier training.

    Each field is one tuning lever; the default values reproduce the historical
    behaviour of the pipeline exactly.

    Attributes:
        models: families to train, among `SUPPORTED_MODELS`. Shortening the
            list (e.g. `["XGBoost"]`) speeds up tuning iterations considerably.
        search: hyperparameter search strategy, "grid" (GridSearchCV,
            exhaustive), "random" (RandomizedSearchCV, `n_iter` draws: allows
            much wider grids at constant budget), or "none" (no search, direct
            fit with `base_params`).
        n_iter: number of combinations drawn when `search="random"`.
        cv: number of cross-validation folds used by the search.
        scoring: score optimised by the search (any scikit-learn name:
            "roc_auc", "average_precision", "f1_macro", "recall", ...).
            "average_precision" is often more discriminating than "roc_auc" on
            a target as imbalanced as no-show (~3%).
        search_sample: size of the tuning subsample (drawn from X_fit);
            `None` = all of X_fit.
        fit_sample: size of the final fit subsample (drawn from X_train);
            `None` = all of X_train.
        fit_sample_by_model: per-model override of `fit_sample`, e.g.
            `{"AdaBoost": 3_000_000}`. RandomForest and XGBoost are parallel and
            take the whole training set; AdaBoost and the scikit-learn MLP are
            sequential, so their cost grows linearly and they need a ceiling.
        use_smote: apply SMOTE to the tuning subset (never to the test set).
        smote_params: arguments passed to `SMOTE` (e.g.
            `{"sampling_strategy": 0.3, "k_neighbors": 3}`).
        balance_classes: enable `class_weight="balanced"` (RF/ANN-compatible)
            and `scale_pos_weight` (XGBoost). Set to False to rely on SMOTE
            and/or the optimal threshold only.
        scale_pos_weight: value imposed for XGBoost; `None` = computed
            automatically (n_negative / n_positive). Lowering it reduces
            over-flagging of the rare class.
        param_grids: per-model overrides of `DEFAULT_PARAM_GRIDS`.
        base_params: per-model overrides of `DEFAULT_BASE_PARAMS`.
        threshold_metric: criterion for choosing the decision threshold, among
            `THRESHOLD_METRICS`.
        threshold_range: (start, stop, step) of the threshold sweep.
        fixed_threshold: threshold used when `threshold_metric="fixed"`.
        n_jobs: parallelism (search, and models that support it).
        verbose: 0 = silent, 1 = standard training log.
        random_state: reproducibility seed.
    """

    models: Sequence[str] = SUPPORTED_MODELS
    search: str = "grid"
    n_iter: int = 20
    cv: int = 3
    scoring: str = "roc_auc"
    search_sample: Optional[int] = 80_000
    fit_sample: Optional[int] = 800_000
    fit_sample_by_model: dict = field(default_factory=dict)
    use_smote: bool = True
    smote_params: dict = field(default_factory=dict)
    balance_classes: bool = True
    scale_pos_weight: Optional[float] = None
    param_grids: dict = field(default_factory=dict)
    base_params: dict = field(default_factory=dict)
    threshold_metric: str = "f1_macro"
    threshold_range: tuple = (0.05, 0.96, 0.01)
    fixed_threshold: float = 0.5
    n_jobs: int = -1
    verbose: int = 1
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        unknown = [m for m in self.models if m not in SUPPORTED_MODELS]
        if unknown:
            raise ValueError(f"Unknown models {unknown}, expected among {list(SUPPORTED_MODELS)}")
        if self.search not in SEARCH_STRATEGIES:
            raise ValueError(f"search='{self.search}' is invalid, expected among {list(SEARCH_STRATEGIES)}")
        if self.threshold_metric not in THRESHOLD_METRICS:
            raise ValueError(f"threshold_metric='{self.threshold_metric}' is invalid, "
                             f"expected among {list(THRESHOLD_METRICS)}")
        self.models = tuple(self.models)

    def grid_for(self, name: str) -> dict:
        """Grid actually explored for this model (default + override)."""
        grid = copy.deepcopy(DEFAULT_PARAM_GRIDS.get(name, {}))
        grid.update(copy.deepcopy(self.param_grids.get(name, {})))
        return grid

    def base_for(self, name: str) -> dict:
        """Effective fixed hyperparameters for this model (default + override)."""
        base = copy.deepcopy(DEFAULT_BASE_PARAMS.get(name, {}))
        base.update(copy.deepcopy(self.base_params.get(name, {})))
        return base

    def replace(self, **overrides) -> "TrainingConfig":
        """Copy of the configuration with the given fields replaced.

        Args:
            **overrides: `TrainingConfig` fields to change.

        Returns:
            A new `TrainingConfig` (the original is left unchanged).
        """
        known = {f.name for f in dataclasses.fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return dataclasses.replace(copy.deepcopy(self), **copy.deepcopy(overrides))

    def summary(self) -> pd.DataFrame:
        """Readable table of every active setting, grids included.

        Returns:
            Two-column DataFrame (Setting, Value), ready for `display`.
        """
        rows = [
            ("Models trained", ", ".join(self.models)),
            ("Hyperparameter search", self.search + (f" (n_iter={self.n_iter})" if self.search == "random" else "")),
            ("Cross-validation folds (cv)", self.cv),
            ("Optimised score", self.scoring),
            ("Tuning subsample", "all of X_fit" if self.search_sample is None else f"{self.search_sample:,}"),
            ("Fit subsample", "all of X_train" if self.fit_sample is None else f"{self.fit_sample:,}"),
            ("SMOTE (tuning)", f"yes {self.smote_params}" if self.use_smote else "no"),
            ("Class weighting", "yes" if self.balance_classes else "no"),
            ("scale_pos_weight (XGB)", "auto" if self.scale_pos_weight is None else self.scale_pos_weight),
            ("Threshold criterion", self.threshold_metric + (f" ({self.fixed_threshold})" if self.threshold_metric == "fixed" else "")),
            ("Threshold sweep", f"{self.threshold_range[0]} -> {self.threshold_range[1]} step {self.threshold_range[2]}"),
            ("n_jobs / seed", f"{self.n_jobs} / {self.random_state}"),
        ]
        rows += [(f"Grid: {m}", str(self.grid_for(m))) for m in self.models]
        rows += [(f"Fixed: {m}", str(self.base_for(m))) for m in self.models if self.base_for(m)]
        return pd.DataFrame(rows, columns=["Setting", "Value"])


class ModelSet(dict):
    """dict {name: trained model} enriched with training traceability.

    Behaves exactly like the plain dict returned previously (the evaluation
    functions need no change), but additionally carries:
      - `search_report`: DataFrame (grid, best parameters, CV score, timings)
      - `config`       : the `TrainingConfig` that was used
      - `best_params`  : dict {name: selected hyperparameters}
    """

    search_report: pd.DataFrame
    config: Optional[TrainingConfig]
    best_params: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_report = pd.DataFrame()
        self.config = None
        self.best_params = {}


def _n_candidates(grid: dict) -> int:
    """Number of combinations in a hyperparameter grid."""
    n = 1
    for values in grid.values():
        n *= max(len(values), 1)
    return n


def _make_estimator(name: str, cfg: TrainingConfig, spw: float, params: Optional[dict] = None,
                    n_jobs: Optional[int] = None):
    """Instantiate a classifier of family `name` according to the configuration.

    Hyperparameters follow an explicit priority order: values set by the module
    < `cfg.base_params` < `params` (selected by the search or imposed by the
    caller).

    Args:
        name: model family (`SUPPORTED_MODELS`).
        cfg: training configuration.
        spw: effective scale_pos_weight (XGBoost).
        params: explicit hyperparameters.
        n_jobs: model-level parallelism; `None` = `cfg.n_jobs`. The
            hyperparameter search forces 1 to avoid core oversubscription
            (nested search x model parallelism), which badly degrades tuning
            time and can stall the joblib workers.

    Returns:
        Untrained scikit-learn estimator.
    """
    kwargs: dict = {"random_state": cfg.random_state}
    model_n_jobs = cfg.n_jobs if n_jobs is None else n_jobs

    if name == "RandomForest":
        kwargs["n_jobs"] = model_n_jobs
        if cfg.balance_classes:
            kwargs["class_weight"] = "balanced"
        estimator_cls = RandomForestClassifier
    elif name == "XGBoost":
        kwargs["n_jobs"] = model_n_jobs
        if cfg.balance_classes:
            kwargs["scale_pos_weight"] = spw
        estimator_cls = XGBClassifier
    elif name == "AdaBoost":
        # SAMME is the only algorithm left in scikit-learn >= 1.6; older versions need it
        # stated explicitly. Checked by signature, not by catching an exception.
        if "algorithm" in inspect.signature(AdaBoostClassifier.__init__).parameters:
            kwargs["algorithm"] = "SAMME"
        estimator_cls = AdaBoostClassifier
    elif name == "ANN":
        estimator_cls = MLPClassifier
    else:  # pragma: no cover - guarded by TrainingConfig.__post_init__
        raise ValueError(f"Unknown model: {name}")

    kwargs.update(cfg.base_for(name))

    if name in SMOTE_BALANCED_MODELS and cfg.balance_classes:
        # These two families cannot be rebalanced by weighting. MLPClassifier
        # accepts neither class_weight nor sample_weight. AdaBoostClassifier
        # exposes no class_weight either, and weighting its base learner breaks
        # the algorithm outright: the stump then optimises a different criterion
        # than the boosting distribution, its weighted error exceeds 50% and
        # sklearn aborts with "BaseClassifier ... is worse than random".
        # Resampling inside the fit is the way that works for both. An
        # imbalanced-learn pipeline resamples on fit only, never on predict, so
        # the reported probabilities stay honest.
        estimator = ImbPipeline([
            ("smote", SMOTE(random_state=cfg.random_state, **cfg.smote_params)),
            (PIPE_STEP, estimator_cls(**kwargs)),
        ])
    else:
        estimator = estimator_cls(**kwargs)

    if params:
        # Grid keys addressing a pipeline step need the step prefix; accept them
        # written either way so callers need not know about the wrapper.
        if isinstance(estimator, ImbPipeline):
            params = {(k if "__" in k else f"{PIPE_STEP}__{k}"): v for k, v in params.items()}
        estimator.set_params(**params)
    return estimator


def _search_hyperparams(name: str, cfg: TrainingConfig, spw: float,
                        Xs: pd.DataFrame, ys: np.ndarray) -> tuple[dict, float, int]:
    """Hyperparameter search for one model family.

    Args:
        name: model family.
        cfg: configuration (strategy, cv, scoring, n_iter...).
        spw: effective scale_pos_weight.
        Xs: features of the tuning subset (possibly SMOTE-resampled).
        ys: matching binary target.

    Returns:
        (best_params, cv_score, n_candidates_evaluated), ({}, nan, 0) when no
        search is performed.
    """
    grid = cfg.grid_for(name)
    if cfg.search == "none" or not grid:
        return {}, float("nan"), 0

    # n_jobs=1 on the model: the search is what parallelises (over folds and
    # candidates); doing both oversubscribes the cores.
    base = _make_estimator(name, cfg, spw, n_jobs=1)
    if isinstance(base, ImbPipeline):
        grid = {(k if "__" in k else f"{PIPE_STEP}__{k}"): v for k, v in grid.items()}
    if cfg.search == "random":
        n_cand = min(cfg.n_iter, _n_candidates(grid))
        search = RandomizedSearchCV(base, grid, n_iter=n_cand, cv=cfg.cv, scoring=cfg.scoring,
                                    n_jobs=cfg.n_jobs, random_state=cfg.random_state, refit=False)
    else:
        n_cand = _n_candidates(grid)
        search = GridSearchCV(base, grid, cv=cfg.cv, scoring=cfg.scoring,
                              n_jobs=cfg.n_jobs, refit=False)
    search.fit(Xs, ys)
    return dict(search.best_params_), float(search.best_score_), n_cand


def train_binary_classifiers(
    X_train: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    feature_cols: list[str],
    positive_label,
    search_sample: Optional[int] = None,
    fit_sample: Optional[int] = None,
    random_state: Optional[int] = None,
    config: Optional[TrainingConfig] = None,
    **overrides,
) -> tuple[ModelSet, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Train the requested binary classifiers (RandomForest, XGBoost, AdaBoost,
    ANN by default) with a threefold imbalance strategy: SMOTE (tuning subset)
    + class_weight/scale_pos_weight + optimal decision threshold (computed
    separately, see `_best_threshold`).

    Every setting is driven by `config` (`TrainingConfig`): families trained,
    grids, search strategy and budget, optimised score, subsample sizes, SMOTE,
    rebalancing, threshold.

    X_train may hold tens of millions of rows (70/20/10 split of the full
    dataset, see `make_master_split`): AdaBoost and the scikit-learn MLP
    (sequential, non-distributed implementations) would be impractical at that
    scale. The final fit therefore runs on a bounded subsample `fit_sample`
    drawn from X_train, and the tuning subsample `search_sample` is itself
    drawn from that subsample.

    Args:
        X_train: training features (already encoded/standardised).
        y_train_raw: raw training target (any values, e.g. 0/1 or 'A'/'B').
        X_test: test features.
        y_test_raw: raw test target.
        feature_cols: feature column names (used for variable importance).
        positive_label: value of `y_*_raw` denoting the (minority) class of interest.
        search_sample: shortcut overriding `config.search_sample` when given.
        fit_sample: shortcut overriding `config.fit_sample` when given.
        random_state: shortcut overriding `config.random_state` when given.
        config: training configuration; `TrainingConfig()` by default.
        **overrides: one-off overrides of `config` fields
            (e.g. `models=["XGBoost"], search="random", n_iter=40`).

    Returns:
        (models, y_train_bin, y_test_bin, X_fit, y_fit), `models` is a
        `ModelSet` (dict {name: model}) carrying `search_report`, `best_params`
        and `config`; `X_fit`/`y_fit` are the bounded subsample actually used
        for the fit, to be reused when comparing train vs test performance (see
        `plot_train_test_comparison`).
    """
    cfg = config or TrainingConfig()
    direct = {k: v for k, v in
              {"search_sample": search_sample, "fit_sample": fit_sample, "random_state": random_state}.items()
              if v is not None}
    if direct or overrides:
        cfg = cfg.replace(**{**direct, **overrides})

    log = print if cfg.verbose else (lambda *a, **k: None)

    y_tr = (y_train_raw == positive_label).astype(int).to_numpy()
    y_te = (y_test_raw == positive_label).astype(int).to_numpy()

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    spw = cfg.scale_pos_weight if cfg.scale_pos_weight is not None else round(n_neg / max(n_pos, 1), 2)
    log(f"Positive class (of interest) '{positive_label}': {n_pos:,}  |  reference: {n_neg:,}")
    log(f"scale_pos_weight (XGB): {spw}" + ("" if cfg.balance_classes else "  [class weighting disabled]"))

    f_n = len(X_train) if cfg.fit_sample is None else min(cfg.fit_sample, len(X_train))
    fit_idx = X_train.sample(n=f_n, random_state=cfg.random_state).index
    fit_pos = X_train.index.get_indexer(fit_idx)
    X_fit, y_fit = X_train.loc[fit_idx], y_tr[fit_pos]
    log(f"Fit subsample: {f_n:,} / {len(X_train):,} rows")

    s_n = len(X_fit) if cfg.search_sample is None else min(cfg.search_sample, len(X_fit))
    idx = X_fit.sample(n=s_n, random_state=cfg.random_state).index
    pos = X_fit.index.get_indexer(idx)
    Xs, ys = X_fit.loc[idx], y_fit[pos]
    log(f"Tuning subsample: {s_n:,} rows  |  search: {cfg.search} (cv={cfg.cv}, scoring={cfg.scoring})")

    # No resampling here: SMOTE applied to the whole tuning subset before the
    # cross-validation put interpolated copies of a minority point in the
    # validation fold while its neighbours stayed in the training fold. The CV
    # score then measured memorisation of synthetic data (0.99 ROC-AUC against
    # 0.75 on the test set) and the search selected whichever model memorised
    # best, a RandomForest with unlimited depth. Rebalancing now happens
    # strictly inside each fold: the ANN carries SMOTE in its own pipeline, the
    # other families use class weights, which need no resampling at all.
    log(f"Class rebalancing: {'class weights (ANN: SMOTE inside its pipeline)' if cfg.balance_classes else 'disabled'}")

    models = ModelSet()
    rows = []

    for name in cfg.models:
        log(f"\n--- {name} ---")
        t0 = time.time()
        best_params, cv_score, n_cand = _search_hyperparams(name, cfg, spw, Xs, ys)
        t_search = time.time() - t0
        if n_cand:
            log(f"{name} best params: {best_params}  | CV {cfg.scoring}: {cv_score:.4f}  "
                f"({n_cand} candidates, {t_search:.1f}s)")
        else:
            log(f"{name}: no search (search='{cfg.search}'), direct fit with the fixed parameters")

        # Per-model fit size: the parallel families take the whole training
        # set, the sequential ones (AdaBoost, MLP) need a ceiling or they never
        # finish. Falls back to the shared X_fit when no override is given.
        if name in cfg.fit_sample_by_model:
            cap = cfg.fit_sample_by_model[name]
            m_n = len(X_train) if cap is None else min(cap, len(X_train))
            m_idx = X_train.sample(n=m_n, random_state=cfg.random_state).index
            X_m, y_m = X_train.loc[m_idx], y_tr[X_train.index.get_indexer(m_idx)]
        else:
            X_m, y_m = X_fit, y_fit

        t0 = time.time()
        model = _make_estimator(name, cfg, spw, params=best_params)
        model.fit(X_m, y_m)
        t_fit = time.time() - t0
        log(f"{name} trained on {len(X_m):,} rows in {t_fit:.1f}s")

        models[name] = model
        rows.append({
            "Model": name,
            "Search": cfg.search if n_cand else "none",
            "Candidates": n_cand,
            f"CV {cfg.scoring}": round(cv_score, 4) if n_cand else np.nan,
            "Best params": str(best_params) if best_params else str(cfg.base_for(name)),
            "Fit rows": len(X_m),
            "Search time (s)": round(t_search, 1),
            "Fit time (s)": round(t_fit, 1),
        })

    models.search_report = pd.DataFrame(rows).set_index("Model")
    models.best_params = {r["Model"]: r["Best params"] for r in rows}
    models.config = cfg
    return models, y_tr, y_te, X_fit, y_fit


def train_single_model(
    name: str,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    params: Optional[dict] = None,
    config: Optional[TrainingConfig] = None,
    models_dict: Optional[dict] = None,
    alias: Optional[str] = None,
    **overrides,
):
    """Retrain ONE model family with imposed hyperparameters, without rerunning
    the search or the other families.

    Intended for fine-tuning iterations: once `train_binary_classifiers` has
    run, a variant can be tested in minutes instead of retraining everything,
    and added to the model dict to be compared with the others during
    evaluation (section 5).

    Args:
        name: model family (`SUPPORTED_MODELS`).
        X_fit: training features (typically the `X_fit` returned by
            `train_binary_classifiers`).
        y_fit: matching binary target (1 = class of interest).
        params: imposed hyperparameters (e.g. `{"max_depth": 6, "n_estimators": 400}`).
        config: reference configuration; `TrainingConfig()` by default.
        models_dict: if given, the trained model is inserted into it.
        alias: insertion key in `models_dict` (default: `name`). Use an alias
            (e.g. "XGBoost_v2") to compare a variant with the initial model.
        **overrides: one-off overrides of `config`.

    Returns:
        The trained model.
    """
    cfg = config or TrainingConfig()
    if overrides:
        cfg = cfg.replace(**overrides)
    n_pos = int((np.asarray(y_fit) == 1).sum())
    n_neg = int((np.asarray(y_fit) == 0).sum())
    spw = cfg.scale_pos_weight if cfg.scale_pos_weight is not None else round(n_neg / max(n_pos, 1), 2)

    model = _make_estimator(name, cfg, spw, params=params)
    if cfg.verbose:
        print(f"Training {alias or name} on {len(X_fit):,} rows, parameters: {params or cfg.base_for(name)}")
    t0 = time.time()
    model.fit(X_fit, y_fit)
    if cfg.verbose:
        print(f"Done in {time.time() - t0:.1f}s")
    if models_dict is not None:
        models_dict[alias or name] = model
    return model


#: History of training runs recorded through `record_training_run`.
TRAINING_RUNS: list[dict] = []


def record_training_run(label: str, results_df: pd.DataFrame,
                        config: Optional[TrainingConfig] = None, notes: str = "") -> pd.DataFrame:
    """Record the results of one training run, to compare successive settings.

    Args:
        label: run name (e.g. "no-show: baseline", "no-show: random search AP").
        results_df: output of `evaluate_binary_classification_models`.
        config: configuration used (its key settings are stored alongside).
        notes: free-form comment.

    Returns:
        The full history (see `training_history_table`).
    """
    meta = {}
    if config is not None:
        meta = {"search": config.search, "scoring": config.scoring, "cv": config.cv,
                "fit_sample": config.fit_sample, "smote": config.use_smote,
                "balance": config.balance_classes, "threshold": config.threshold_metric}
    for _, row in results_df.iterrows():
        TRAINING_RUNS.append({"Run": label, "Model": row["Model"], **meta,
                              **{k: v for k, v in row.items() if k != "Model"},
                              "Notes": notes})
    return training_history_table()


def training_history_table(sort_by: str = AUC_COL) -> pd.DataFrame:
    """Comparison table of every recorded run.

    Args:
        sort_by: column to sort by, descending (when present).

    Returns:
        DataFrame (run x model) of the metrics of each run.
    """
    if not TRAINING_RUNS:
        print("No run recorded: call record_training_run(...) after each evaluation.")
        return pd.DataFrame()
    hist = pd.DataFrame(TRAINING_RUNS)
    if sort_by in hist.columns:
        hist = hist.sort_values(sort_by, ascending=False)
    return hist.reset_index(drop=True)


def clear_training_history() -> None:
    """Clear the training run history."""
    TRAINING_RUNS.clear()


def _threshold_score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score of a set of binary predictions under the chosen threshold criterion.

    Args:
        metric: criterion among `THRESHOLD_METRICS` (excluding "fixed").
        y_true: true binary target.
        y_pred: binary predictions at the threshold being tested.

    Returns:
        Criterion value (to maximise).
    """
    if metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")
    if metric == "f1_positive":
        return f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "youden":
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tpr = cm[1, 1] / max(cm[1].sum(), 1)
        fpr = cm[0, 1] / max(cm[0].sum(), 1)
        return tpr - fpr
    raise ValueError(f"Unknown threshold criterion: {metric}")


def _best_threshold(model, X, y_true: np.ndarray, metric: str = "f1_macro",
                    threshold_range: tuple = (0.05, 0.96, 0.01),
                    fixed_threshold: float = 0.5,
                    proba: Optional[np.ndarray] = None) -> tuple[float, float]:
    """Decision threshold on P(positive class) maximising the chosen criterion.

    Args:
        model: classifier exposing `predict_proba`.
        X: features.
        y_true: binary target (1 = class of interest).
        metric: criterion among `THRESHOLD_METRICS`. "fixed" returns
            `fixed_threshold` without sweeping.
        threshold_range: (start, stop, step) of the sweep.
        fixed_threshold: threshold imposed when `metric="fixed"`.
        proba: pre-computed probabilities (avoids a redundant `predict_proba`).

    Returns:
        (threshold, criterion value at that threshold)
    """
    proba_pos = model.predict_proba(X)[:, 1] if proba is None else proba
    if metric == "fixed":
        yp = np.where(proba_pos >= fixed_threshold, 1, 0)
        return round(float(fixed_threshold), 2), round(float(_threshold_score("f1_macro", y_true, yp)), 4)

    best_t, best_s = 0.5, -np.inf
    for t in np.arange(*threshold_range):
        yp = np.where(proba_pos >= t, 1, 0)
        s = _threshold_score(metric, y_true, yp)
        if s > best_s:
            best_s, best_t = s, t
    return round(float(best_t), 2), round(float(best_s), 4)


def compute_classification_metrics(
    models_dict: dict, X_test: pd.DataFrame, y_te: np.ndarray, positive_name: str, reference_name: str,
    config: Optional["TrainingConfig"] = None, threshold_metric: Optional[str] = None,
    threshold_range: Optional[tuple] = None, fixed_threshold: Optional[float] = None,
) -> tuple[pd.DataFrame, dict, list[str], np.ndarray]:
    """Compute the reported classification metrics of every model at the
    threshold optimised for the chosen criterion: AUC, accuracy, and
    macro-averaged precision, recall and F1-score.

    The threshold criterion is a tuning lever in its own right: on a strongly
    imbalanced target, switching from "f1_macro" to "f1_positive" or "youden"
    changes the recall/precision trade-off radically without retraining
    anything.

    Args:
        models_dict: dict {model_name: trained model}.
        X_test: test features.
        y_te: binary test target (1 = class of interest).
        positive_name: display label of the positive class.
        reference_name: display label of the reference class.
        config: `TrainingConfig` supplying the threshold settings (a `ModelSet`
            returned by `train_binary_classifiers` carries its own in `.config`).
        threshold_metric: threshold criterion, overrides `config` (see `THRESHOLD_METRICS`).
        threshold_range: (start, stop, step) of the sweep, overrides `config`.
        fixed_threshold: threshold imposed when `threshold_metric="fixed"`.

    Returns:
        (results_df, cache, model_names, palette)
    """
    if config is None:
        config = getattr(models_dict, "config", None)
    t_metric = threshold_metric or (config.threshold_metric if config else "f1_macro")
    t_range = threshold_range or (config.threshold_range if config else (0.05, 0.96, 0.01))
    t_fixed = fixed_threshold if fixed_threshold is not None else (config.fixed_threshold if config else 0.5)

    model_names = list(models_dict.keys())
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))
    rows, cache = [], {}

    for name, model in models_dict.items():
        p_pos = model.predict_proba(X_test)[:, 1]
        y_def = model.predict(X_test)
        bt, _ = _best_threshold(model, X_test, y_te, metric=t_metric,
                                threshold_range=t_range, fixed_threshold=t_fixed, proba=p_pos)
        y_opt = np.where(p_pos >= bt, 1, 0)

        auc = roc_auc_score(y_te, p_pos)
        acc = accuracy_score(y_te, y_opt)
        prec = precision_score(y_te, y_opt, average="macro", zero_division=0)
        rec = recall_score(y_te, y_opt, average="macro", zero_division=0)
        f1 = f1_score(y_te, y_opt, average="macro", zero_division=0)
        cm_d = confusion_matrix(y_te, y_def, labels=[0, 1])
        cm_t = confusion_matrix(y_te, y_opt, labels=[0, 1])

        cache[name] = dict(p_pos=p_pos, y_opt=y_opt, cm_def=cm_d, cm_opt=cm_t,
                           bt=bt, auc=auc, f1_t=f1, acc=acc, prec=prec, rec=rec)
        rows.append({
            "Model": name,
            AUC_COL: round(auc, METRIC_DECIMALS),
            ACC_COL: round(acc, METRIC_DECIMALS),
            PREC_COL: round(prec, METRIC_DECIMALS),
            REC_COL: round(rec, METRIC_DECIMALS),
            F1_COL: round(f1, METRIC_DECIMALS),
            THR_COL: bt,
        })

    results_df = pd.DataFrame(rows)
    print(f"\n{'='*70}\n  RESULTS, {positive_name} vs {reference_name}"
          f"   (threshold optimised on: {t_metric})\n{'='*70}")
    display(results_df.set_index("Model"))
    return results_df, cache, model_names, palette


def plot_metrics_comparison(results_df: pd.DataFrame, model_names: list[str], palette: np.ndarray, prefix: str = "") -> None:
    """Grouped bars comparing the five reported metrics across models.

    Args:
        results_df: output of `compute_classification_metrics`.
        model_names: model names, in display order.
        palette: per-model colours.
        prefix: file-name prefix.
    """
    metrics_keys = REPORTED_METRICS
    metrics_labels = REPORTED_METRICS
    x = np.arange(len(metrics_keys))
    width = 0.72 / len(model_names)

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for i, (name, color) in enumerate(zip(model_names, palette)):
        row = results_df[results_df["Model"] == name].iloc[0]
        vals = [row[m] for m in metrics_keys]
        offs = x + (i - (len(model_names) - 1) / 2) * width
        bars = ax.bar(offs, vals, width, label=name, color=color, alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{v:.{METRIC_DECIMALS}f}", ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    _savefig("metrics_comparison", prefix)
    plt.show()


def plot_confusion_matrices(cache: dict, model_names: list[str], class_names: list[str], palette: np.ndarray, prefix: str = "") -> None:
    """Normalised confusion matrices (at the optimal threshold) for every model.

    Args:
        cache: dict returned by `compute_classification_metrics`.
        model_names: model names.
        class_names: [reference, positive] labels for the axes.
        palette: per-model colours.
        prefix: file-name prefix.
    """
    n = len(model_names)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for ax, (name, color) in zip(axes_flat, zip(model_names, palette)):
        cm_raw = cache[name]["cm_opt"]
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, cmap="Blues", ax=ax, xticklabels=class_names, yticklabels=class_names,
                    linewidths=1.2, linecolor="white", vmin=0, vmax=1, cbar=True)
        for i in range(2):
            for j in range(2):
                pct, n_ij = cm_norm[i, j], cm_raw[i, j]
                txt_color = "white" if pct > 0.55 else "black"
                ax.text(j + 0.5, i + 0.38, f"{pct:.1%}", ha="center", va="center",
                        fontsize=FS_ANNOT + 2, fontweight="bold", color=txt_color)
                ax.text(j + 0.5, i + 0.65, f"({n_ij:,})", ha="center", va="center", fontsize=FS_ANNOT_SMALL, color=txt_color)
        ax.set_title(f"{name}  |  threshold={cache[name]['bt']}  |  F1={cache[name]['f1_t']:.{METRIC_DECIMALS}f}", color=color)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    _savefig("confusion_matrices", prefix)
    plt.show()


def plot_roc_curves(cache: dict, y_te: np.ndarray, model_names: list[str], palette: np.ndarray, prefix: str = "") -> None:
    """Overlaid ROC curves for every model.

    Args:
        cache: dict returned by `compute_classification_metrics`.
        y_te: binary test target.
        model_names: model names.
        palette: per-model colours.
        prefix: file-name prefix.
    """
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for name, color in zip(model_names, palette):
        fpr, tpr, _ = roc_curve(y_te, cache[name]["p_pos"])
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f"{name}  AUC = {cache[name]['auc']:.{METRIC_DECIMALS}f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate (TPR)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    _savefig("roc_curves", prefix)
    plt.show()


def plot_precision_recall(cache: dict, y_te: np.ndarray, model_names: list[str], palette: np.ndarray, positive_name: str, prefix: str = "") -> None:
    """Precision-recall curves for the (minority) class of interest.

    Args:
        cache: dict returned by `compute_classification_metrics`.
        y_te: binary test target.
        model_names: model names.
        palette: per-model colours.
        positive_name: label of the positive class.
        prefix: file-name prefix.
    """
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for name, color in zip(model_names, palette):
        pr, rc, _ = precision_recall_curve(y_te, cache[name]["p_pos"])
        ap = average_precision_score(y_te, cache[name]["p_pos"])
        ax.plot(rc, pr, lw=2.2, color=color, label=f"{name}  AP = {ap:.{METRIC_DECIMALS}f}")
    baseline = float(y_te.mean())
    ax.axhline(baseline, linestyle="--", color="gray", lw=1.2, alpha=0.7, label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel(f"Recall, {positive_name}")
    ax.set_ylabel(f"Precision, {positive_name}")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    _savefig("precision_recall_curves", prefix)
    plt.show()


def plot_feature_importances(
    models_dict: dict, feature_cols: list[str], model_names: list[str], palette: np.ndarray,
    X_eval: Optional[pd.DataFrame] = None, y_eval: Optional[np.ndarray] = None,
    perm_sample: int = 20_000, random_state: int = RANDOM_STATE, prefix: str = "",
) -> None:
    """Variable importance: native Gini for tree-based models
    (`.feature_importances_`); permutation importance (ROC-AUC drop) for the
    others (e.g. the MLP, which exposes no native importance attribute).

    Args:
        models_dict: dict {name: model}.
        feature_cols: feature column names.
        model_names: model names.
        palette: per-model colours.
        X_eval: features used for the permutation importance of models without
            `.feature_importances_`. Required when such a model is present.
        y_eval: matching binary target.
        perm_sample: bounded subsample of X_eval (permutation importance
            re-evaluates the model n_repeats x n_features times).
        random_state: seed.
        prefix: file-name prefix.
    """
    importances: dict[str, tuple[str, pd.Series]] = {}
    for name, model in models_dict.items():
        if hasattr(model, "feature_importances_"):
            importances[name] = ("Gini", pd.Series(model.feature_importances_, index=feature_cols))
        elif X_eval is not None and y_eval is not None:
            n = min(perm_sample, len(X_eval))
            idx = X_eval.sample(n=n, random_state=random_state).index
            pos = X_eval.index.get_indexer(idx)
            result = permutation_importance(
                model, X_eval.loc[idx], y_eval[pos],
                scoring="roc_auc", n_repeats=5, random_state=random_state, n_jobs=-1)
            importances[name] = ("Permutation (ROC-AUC drop)", pd.Series(result.importances_mean, index=feature_cols))
        else:
            print(f"{name}: no '.feature_importances_' and no X_eval/y_eval provided, skipped.")

    if not importances:
        print("No variable importance available.")
        return

    ncols = min(len(importances), 2)
    nrows = (len(importances) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()
    for ax, (name, (kind, fi)) in zip(axes, importances.items()):
        top15 = fi.sort_values().tail(15)
        color = palette[model_names.index(name)]
        bars = ax.barh(top15.index, top15.values, color=color, alpha=0.85)
        for bar, v in zip(bars, top15.values):
            ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=FS_ANNOT_SMALL)
        ax.set_title(name)
        ax.set_xlabel(kind)
        ax.grid(axis="x", alpha=0.25)
        sns.despine(ax=ax)
    for ax in axes[len(importances):]:
        ax.set_visible(False)
    _savefig("feature_importances", prefix)
    plt.show()


def plot_train_test_comparison(
    models_dict: dict, X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
    model_names: list[str], palette: np.ndarray, prefix: str = "",
) -> pd.DataFrame:
    """Compare AUC and F1-score (threshold 0.5) between the data seen during
    training (X_train/y_train, the `X_fit`/`y_fit` subsample actually used by
    `train_binary_classifiers`, NOT the whole `train_df`) and the unseen test
    set, to detect overfitting or data leakage.

    A small train/test gap with scores close to 1.0 everywhere is NOT a sign of
    robustness: combined with models from very different families (trees,
    boosting, neural network) all converging to near-perfect scores, it is the
    signature of data leakage (a feature encodes the target almost directly)
    rather than of genuine learning, true overfitting would instead show a
    large train/test gap.

    Args:
        models_dict: dict {name: trained model}.
        X_train: features seen during training (X_fit).
        y_train: matching binary target (y_fit).
        X_test: test features.
        y_test: binary test target.
        model_names: model names, in display order.
        palette: per-model colours.
        prefix: file-name prefix.

    Returns:
        DataFrame with train and test AUC/F1-score, and the train-test gap.
    """
    rows = []
    for name in model_names:
        model = models_dict[name]
        p_tr = model.predict_proba(X_train)[:, 1]
        p_te = model.predict_proba(X_test)[:, 1]
        auc_tr, auc_te = roc_auc_score(y_train, p_tr), roc_auc_score(y_test, p_te)
        f1_tr = f1_score(y_train, model.predict(X_train), average="macro", zero_division=0)
        f1_te = f1_score(y_test, model.predict(X_test), average="macro", zero_division=0)
        rows.append({
            "Model": name,
            f"{AUC_COL} (train)": round(auc_tr, METRIC_DECIMALS), f"{AUC_COL} (test)": round(auc_te, METRIC_DECIMALS),
            f"{AUC_COL} gap (train-test)": round(auc_tr - auc_te, METRIC_DECIMALS),
            f"{F1_COL} (train)": round(f1_tr, METRIC_DECIMALS), f"{F1_COL} (test)": round(f1_te, METRIC_DECIMALS),
            f"{F1_COL} gap (train-test)": round(f1_tr - f1_te, METRIC_DECIMALS),
        })
    comparison_df = pd.DataFrame(rows)
    print(f"\n{'='*70}\n  TRAIN vs TEST, overfitting / leakage check\n{'='*70}")
    display(comparison_df.set_index("Model"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(model_names))
    width = 0.35
    for ax, metric in zip(axes, [AUC_COL, F1_COL]):
        tr_vals = comparison_df[f"{metric} (train)"].to_numpy()
        te_vals = comparison_df[f"{metric} (test)"].to_numpy()
        ax.bar(x - width / 2, tr_vals, width, label="Train", color=PLOT_COLORS[0], alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, te_vals, width, label="Test", color=PLOT_COLORS[1], alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        sns.despine(ax=ax)
    _savefig("train_test_comparison", prefix)
    plt.show()
    return comparison_df


ALL_EVAL_PLOTS: tuple[str, ...] = ("metrics", "train_test", "confusion", "roc", "pr", "importances")


def evaluate_binary_classification_models(
    models_dict: dict, X_test: pd.DataFrame, y_te: np.ndarray,
    positive_name: str, reference_name: str, feature_cols: list[str],
    X_train: Optional[pd.DataFrame] = None, y_tr: Optional[np.ndarray] = None,
    prefix: str = "", config: Optional["TrainingConfig"] = None,
    threshold_metric: Optional[str] = None, threshold_range: Optional[tuple] = None,
    fixed_threshold: Optional[float] = None,
    plots: Sequence[str] = ALL_EVAL_PLOTS, perm_sample: int = 20_000,
    eval_sample: Optional[int] = None,
) -> pd.DataFrame:
    """Chain the metric computation and the standard visualisations.

    Args:
        models_dict: dict {name: trained model}.
        X_test: test features.
        y_te: binary test target.
        positive_name: label of the class of interest.
        reference_name: label of the reference class.
        feature_cols: feature columns.
        X_train: features seen during training (X_fit), optional, when given
            together with y_tr, adds a train vs test comparison (overfitting /
            leakage check, see `plot_train_test_comparison`).
        y_tr: matching binary training target (y_fit).
        prefix: file-name prefix (e.g. 'noshow', 'servicetime') so the figures
            of the other pipeline are not overwritten.
        config: `TrainingConfig` supplying the threshold settings (taken from
            `models_dict.config` when omitted).
        threshold_metric: threshold criterion, overrides `config`.
        threshold_range: (start, stop, step) of the threshold sweep.
        fixed_threshold: threshold imposed when `threshold_metric="fixed"`.
        plots: figures to produce, among `ALL_EVAL_PLOTS`. Shortening the list
            (e.g. `plots=["metrics"]`, or `plots=[]`) speeds up tuning
            iterations markedly, the ANN permutation importance and the
            train/test comparison are the most expensive items.
        perm_sample: subsample size for permutation importance.
        eval_sample: when given, evaluate on a random subsample of that size
            instead of the whole test set (fast iterations; set back to None
            for the final figures).

    Returns:
        results_df (see `compute_classification_metrics`).
    """
    if eval_sample is not None and eval_sample < len(X_test):
        idx = X_test.sample(n=eval_sample, random_state=RANDOM_STATE).index
        pos = X_test.index.get_indexer(idx)
        X_test, y_te = X_test.loc[idx], y_te[pos]
        print(f"Evaluating on a subsample of {eval_sample:,} test rows")

    plots = tuple(plots)
    unknown = [p for p in plots if p not in ALL_EVAL_PLOTS]
    if unknown:
        raise ValueError(f"Unknown figures {unknown}, expected among {list(ALL_EVAL_PLOTS)}")

    results_df, cache, model_names, palette = compute_classification_metrics(
        models_dict, X_test, y_te, positive_name, reference_name,
        config=config, threshold_metric=threshold_metric,
        threshold_range=threshold_range, fixed_threshold=fixed_threshold)
    if "metrics" in plots:
        plot_metrics_comparison(results_df, model_names, palette, prefix)
    if "train_test" in plots and X_train is not None and y_tr is not None:
        plot_train_test_comparison(models_dict, X_train, y_tr, X_test, y_te, model_names, palette, prefix)
    if "confusion" in plots:
        plot_confusion_matrices(cache, model_names, [reference_name, positive_name], palette, prefix)
    if "roc" in plots:
        plot_roc_curves(cache, y_te, model_names, palette, prefix)
    if "pr" in plots:
        plot_precision_recall(cache, y_te, model_names, palette, positive_name, prefix)
    if "importances" in plots:
        plot_feature_importances(models_dict, feature_cols, model_names, palette,
                                 X_eval=X_test, y_eval=y_te, perm_sample=perm_sample, prefix=prefix)
    return results_df


# ============================================================================
# 5. Probability calibration (required for the ILP q_i coefficients)
# ============================================================================

def plot_calibration_curve_(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, n_bins: int = 10, prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Calibration curve (reliability diagram).

    Args:
        y_true: true binary target.
        y_prob: predicted probabilities of the positive class.
        model_name: model name (title / file name).
        n_bins: number of calibration bins.
        prefix: file-name prefix.

    Returns:
        (fraction_of_positives, mean_predicted_value)
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot(mean_pred, frac_pos, "o-", color=PLOT_COLORS[0], label=model_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    _savefig(f"calibration_curve_{model_name}", prefix)
    plt.show()
    return frac_pos, mean_pred


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (lower = better calibrated AND more discriminating).

    Args:
        y_true: true binary target.
        y_prob: predicted probabilities of the positive class.

    Returns:
        Brier score.
    """
    return float(brier_score_loss(y_true, y_prob))


def calibrate_and_select(
    model, X_test: pd.DataFrame, y_test: np.ndarray,
    methods: tuple[str, ...] = ("sigmoid", "isotonic"),
    calib_fraction: float = 0.5, random_state: int = RANDOM_STATE,
) -> tuple[object, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Recalibrate an already trained model with Platt scaling (sigmoid) and
    isotonic regression, and keep whichever minimises the Brier score, needed
    because the ILP requires calibrated q_i, not merely discriminating ones.

    The test set is split into a disjoint calibration subset (where the
    correction is fitted) and evaluation subset: the base model (already
    trained on the training set) is frozen with `FrozenEstimator`, so only the
    calibration correction is learned, on data used neither to train the base
    model nor to report the final evaluation. Without that separation,
    calibrating and evaluating on the same subset would overstate the
    calibration quality achieved.

    Args:
        model: already trained, uncalibrated binary classifier.
        X_test: full test features.
        y_test: full binary test target.
        methods: calibration methods to compare.
        calib_fraction: fraction of X_test allocated to fitting the calibration.
        random_state: seed.

    Returns:
        (best_calibrated_or_original_model, comparison_df, X_eval, y_eval),
        X_eval/y_eval are the evaluation subset (disjoint from the calibration
        fit), to be reused when plotting the calibration curves.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    X_calib, X_eval, y_calib, y_eval = train_test_split(
        X_test, y_test, test_size=(1 - calib_fraction), random_state=random_state, stratify=y_test)

    raw_prob_eval = model.predict_proba(X_eval)[:, 1]
    rows = [{"Method": "none (raw)", "Brier": compute_brier_score(y_eval, raw_prob_eval)}]
    candidates = {"none (raw)": model}

    for method in methods:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrated.fit(X_calib, y_calib)
        prob = calibrated.predict_proba(X_eval)[:, 1]
        brier = compute_brier_score(y_eval, prob)
        rows.append({"Method": method, "Brier": brier})
        candidates[method] = calibrated

    comparison_df = pd.DataFrame(rows).sort_values("Brier").reset_index(drop=True)
    display(comparison_df)
    best_method = comparison_df.iloc[0]["Method"]
    print(f"Calibration method selected: {best_method}")
    return candidates[best_method], comparison_df, X_eval, y_eval


# ============================================================================
# 6. Service-time clustering (K-means) and classification
# ============================================================================

def evaluate_clustering_ks(
    df: pd.DataFrame, service_time_col: str = "service_time", candidate_ks: Optional[list[int]] = None,
    sample_size: int = 200_000, random_state: int = RANDOM_STATE,
) -> tuple[dict, dict, np.ndarray, StandardScaler]:
    """Evaluate several values of K (SAE elbow method + silhouette coefficient).

    Args:
        df: DataFrame containing `service_time_col`.
        service_time_col: column to cluster.
        candidate_ks: values of K to test.
        sample_size: sample size used to compute the silhouette.
        random_state: seed.

    Returns:
        (sae_per_k, silhouette_per_k, full_standardised_service_time, scaler)
    """
    if candidate_ks is None:
        candidate_ks = [2, 3, 4, 5]

    values = df[[service_time_col]].astype(float)
    scaler = StandardScaler()
    service_time_std = scaler.fit_transform(values)

    sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    sample_std = scaler.transform(sample[[service_time_col]].astype(float))

    sae_results, sil_results = {}, {}
    print("KMeans clustering")
    for k in candidate_ks:
        clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(sample_std)
        centers = clusterer.cluster_centers_
        sae = float(np.sum(np.abs(sample_std - centers[labels])))
        sil = float(silhouette_score(sample_std, labels))
        sae_results[k], sil_results[k] = sae, sil
        print(f"  K={k} | SAE={sae:.2f} | Silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(list(sae_results), list(sae_results.values()), "o-", color=PLOT_COLORS[0])
    axes[0].set_xlabel("K"); axes[0].set_ylabel("SAE")
    axes[0].grid(alpha=0.4); sns.despine(ax=axes[0])
    axes[1].plot(list(sil_results), list(sil_results.values()), "o-", color=PLOT_COLORS[2])
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.4); sns.despine(ax=axes[1])
    _savefig("clustering_k_evaluation")
    plt.show()
    return sae_results, sil_results, service_time_std, scaler


def bootstrap_silhouette(data: np.ndarray, labels: np.ndarray, n_bootstrap: int = 10, sample_size: int = 10_000) -> tuple[float, float]:
    """Bootstrap the silhouette coefficient to validate the stability of the
    selected K.

    Args:
        data: standardised data.
        labels: cluster labels.
        n_bootstrap: number of resamples.
        sample_size: size of each resample.

    Returns:
        (mean, standard deviation) of the bootstrapped silhouette.
    """
    sils = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(data), size=min(sample_size, len(data)), replace=True)
        sils.append(silhouette_score(data[idx], labels[idx]))
    return float(np.mean(sils)), float(np.std(sils))


def fit_clustering_and_label(
    df: pd.DataFrame, best_k: int, service_time_std: np.ndarray, service_time_col: str = "service_time", random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """Final clustering and A/B/C labelling by decreasing service time.

    Args:
        df: DataFrame to enrich (modified in place).
        best_k: selected number of clusters.
        service_time_std: standardised service time (same order as df).
        service_time_col: source column.
        random_state: seed.

    Returns:
        (labelled_df, per_cluster_profile, label_map, label_names)
    """
    clusterer = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = clusterer.fit_predict(service_time_std)
    df["cluster"] = labels
    profile = (df.groupby("cluster")[service_time_col].agg(["count", "mean", "median"])
               .sort_values("mean", ascending=False))
    print("\nGroup profile:")
    print(profile)

    order = profile.index.tolist()
    label_names = ["A", "B", "C", "D", "E"][:len(order)]
    label_map = {cluster: label_names[i] for i, cluster in enumerate(order)}
    df["group_label"] = df["cluster"].map(label_map)
    return df, profile, label_map, label_names


def learn_decision_tree_cutoffs(
    df: pd.DataFrame, service_time_col: str = "service_time", group_col: str = "group_label", label_names: Optional[list[str]] = None,
) -> tuple[DecisionTreeClassifier, float]:
    """Learn the service_time -> group cutoff with a shallow decision tree and
    EXTRACT the numeric threshold from its root (instead of leaving it implicit
    in the figure).

    Args:
        df: DataFrame with `service_time_col` and `group_col`.
        service_time_col: single explanatory variable.
        group_col: categorical target (groups A/B/...).
        label_names: class names for the tree plot.

    Returns:
        (fitted_tree, root_cutoff_in_minutes)
    """
    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    tree.fit(df[[service_time_col]], df[group_col])

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    plot_tree(tree, feature_names=[service_time_col], class_names=label_names, filled=True,
              rounded=True, fontsize=FS_TREE, ax=ax, proportion=True)
    _savefig("decision_tree_cutoffs")
    plt.show()

    primary_cutoff = float(tree.tree_.threshold[0])
    print(f"\nPrimary cutoff (tree root): {primary_cutoff:.2f} min")
    print("\nFinal profile per group:")
    print(df.groupby(group_col)[service_time_col].agg(["count", "mean", "median"]))
    return tree, primary_cutoff


def get_group_labels_for_split(df_sample: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Align the clustering group labels with X_train/X_test through the shared index.

    Args:
        df_sample: labelled source DataFrame (`group_label`).
        X_train: training features (same index as df_sample).
        X_test: test features (same index as df_sample).

    Returns:
        (y_train_groups, y_test_groups)
    """
    y_tr = df_sample.loc[X_train.index, "group_label"]
    y_te = df_sample.loc[X_test.index, "group_label"]
    print("Train distribution:", y_tr.value_counts().to_dict())
    print("Test distribution :", y_te.value_counts().to_dict())
    return y_tr, y_te


# ============================================================================
# 7. Deployment, unified inference and serialisation
# ============================================================================

def predict_patient(
    features_df: pd.DataFrame, noshow_model, group_classifier,
    positive_group_label: str = "A", reference_group_label: str = "B",
) -> pd.DataFrame:
    """Single inference entry point: (features) -> (q_i, group).

    Uses the group classifier trained on the FEATURES (`group_classifier`,
    section 4b), not the service_time -> group tree from
    `learn_decision_tree_cutoffs` (which requires the realised service time to
    be known already, unavailable before a new patient's consultation). The
    tree remains useful for the interpretable cutoff analysis, but not for
    production inference.

    Args:
        features_df: already encoded/standardised features (same columns as in
            training), for both calibrated models.
        noshow_model: calibrated no-show model (predict_proba returns P(NoShow)
            in column 1).
        group_classifier: group classifier trained by `train_binary_classifiers`
            (predict_proba returns P(group == positive_group_label) in column 1).
        positive_group_label: label of the group encoded as 1 by `train_binary_classifiers`.
        reference_group_label: label of the group encoded as 0.

    Returns:
        DataFrame indexed like `features_df`, with columns `q_i` (show-up
        probability = 1 - P(NoShow)) and `group` (predicted label).
    """
    p_noshow = noshow_model.predict_proba(features_df)[:, 1]
    p_group_positive = group_classifier.predict_proba(features_df)[:, 1]
    group = np.where(p_group_positive >= 0.5, positive_group_label, reference_group_label)
    return pd.DataFrame({"q_i": 1.0 - p_noshow, "group": group}, index=features_df.index)


def save_artifacts(
    noshow_model, cluster_tree, group_classifier, scaler: StandardScaler,
    cat_encodings: dict, feature_cols: list[str], name_prefix: str = "final",
) -> None:
    """Serialise the three inference artifacts and the preprocessing objects.

    Args:
        noshow_model: calibrated no-show model.
        cluster_tree: service_time -> group tree.
        group_classifier: group assignment classifier for new patients.
        scaler: StandardScaler fitted on the training set.
        cat_encodings: categorical encodings fitted on the training set.
        feature_cols: ordered list of feature columns.
        name_prefix: prefix of the output files in models/.
    """
    ensure_output_dirs()
    joblib.dump(noshow_model, f"{MODELS_DIR}/{name_prefix}_noshow_model.joblib")
    joblib.dump(cluster_tree, f"{MODELS_DIR}/{name_prefix}_cluster_tree.joblib")
    joblib.dump(group_classifier, f"{MODELS_DIR}/{name_prefix}_group_classifier.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/{name_prefix}_scaler.joblib")
    joblib.dump({"cat_encodings": cat_encodings, "feature_cols": feature_cols},
                f"{MODELS_DIR}/{name_prefix}_preprocessing.joblib")
    print(f"Artifacts saved to {MODELS_DIR}/ with prefix '{name_prefix}_'.")


def load_artifacts(name_prefix: str = "final") -> dict:
    """Reload the artifacts serialised by `save_artifacts`.

    Args:
        name_prefix: prefix used when saving.

    Returns:
        dict {noshow_model, cluster_tree, group_classifier, scaler, cat_encodings, feature_cols}
    """
    prep = joblib.load(f"{MODELS_DIR}/{name_prefix}_preprocessing.joblib")
    return {
        "noshow_model": joblib.load(f"{MODELS_DIR}/{name_prefix}_noshow_model.joblib"),
        "cluster_tree": joblib.load(f"{MODELS_DIR}/{name_prefix}_cluster_tree.joblib"),
        "group_classifier": joblib.load(f"{MODELS_DIR}/{name_prefix}_group_classifier.joblib"),
        "scaler": joblib.load(f"{MODELS_DIR}/{name_prefix}_scaler.joblib"),
        "cat_encodings": prep["cat_encodings"],
        "feature_cols": prep["feature_cols"],
    }


# ============================================================================
# 8. ILP core, shared with notebooks/gen_nb.py (01_ilp_validation.ipynb)
# ============================================================================
#
# Both the numerical-validation notebook generator and this pipeline import
# Instance/Config/Solution/solve_ilp from here, which guarantees they solve
# EXACTLY the same model.
# ============================================================================

# The MIP solver is HiGHS, driven through Pyomo's appsi interface. That interface
# exposes what the legacy SolverFactory wrapper did not: the termination status,
# the incumbent objective and the best bound. The previous code read a
# non-existent `res.solver.mip_gap`, caught the AttributeError and reported a gap
# of 0 for every instance, optimal or not (audit B1). There is no fallback solver:
# a run without HiGHS must fail, not switch silently to another algorithm.
from pyomo.contrib.appsi.solvers import Highs as _AppsiHighs
from pyomo.contrib.appsi.base import TerminationCondition as _TC
from pyomo.common.collections import ComponentMap

SOLVER_NAME = "appsi_highs"
if not _AppsiHighs().available():  # pragma: no cover - environment check
    raise ImportError("HiGHS is not available through pyomo.contrib.appsi (pip install highspy)")

#: Termination conditions after which the incumbent may be loaded: a feasible
#: solution exists, but only `optimal` carries a proof of optimality.
_FEASIBLE_STOPS = {_TC.optimal, _TC.maxTimeLimit, _TC.maxIterations,
                   _TC.objectiveLimit, _TC.interrupted}


def init_solver() -> str:
    """Name of the MIP solver used throughout (kept for backward compatibility)."""
    return SOLVER_NAME


@dataclass
class SolveReport:
    """Outcome of one MIP solve, as read from the solver (never inferred)."""
    status: str            # appsi termination condition name, e.g. "optimal", "maxTimeLimit"
    objective: float       # incumbent objective value (solver's objective, minimised)
    best_bound: float      # best proven lower bound on that objective
    gap: float             # |objective - best_bound| / max(|objective|, 1e-9)
    solve_time: float      # wall-clock seconds of the call: Pyomo-to-HiGHS translation + solve
    options: dict          # solver options actually applied, plus HiGHS' own measurements

    @property
    def proven_optimal(self) -> bool:
        return self.status == "optimal"


def relative_gap(objective: float, bound: float) -> float:
    """Relative MIP gap |objective - bound| / max(|objective|, 1e-9).

    Same convention as HiGHS' mip_rel_gap. Infinite when no bound is known.
    """
    if bound is None or not np.isfinite(bound) or objective is None or not np.isfinite(objective):
        return float("inf")
    return abs(objective - bound) / max(abs(objective), 1e-9)


def solve_model(m, cfg: "Config", warm_start: Optional[dict] = None) -> SolveReport:
    """Solve a Pyomo MIP with HiGHS and load the solution only once it is known to exist.

    Args:
        m: Pyomo model with a single minimised objective.
        cfg: solver options (`time_limit`, `mip_gap`, `threads`).
        warm_start: optional ComponentMap {variable: value}; values are set on the
            model and HiGHS is asked to start from them.

    Returns:
        SolveReport read from the solver.

    Raises:
        RuntimeError: no feasible solution (infeasible, unbounded, error, or the
            time limit was reached before any incumbent was found).
    """
    opt = _AppsiHighs()
    opt.config.load_solution = False
    opt.config.time_limit = float(cfg.time_limit)
    opt.config.mip_gap = float(cfg.mip_gap)
    opt.highs_options = {"threads": int(cfg.threads)}
    if warm_start:
        for var, val in warm_start.items():
            var.set_value(val)
        opt.config.warmstart = True
    options = {"time_limit": cfg.time_limit, "mip_rel_gap": cfg.mip_gap, "threads": cfg.threads,
               "warm_start": bool(warm_start)}

    t0 = time.perf_counter()
    res = opt.solve(m)
    elapsed = time.perf_counter() - t0

    tc = res.termination_condition
    incumbent = res.best_feasible_objective
    if tc not in _FEASIBLE_STOPS or incumbent is None or not np.isfinite(incumbent):
        raise RuntimeError(f"HiGHS returned no feasible solution (termination: {tc.name}, "
                           f"incumbent: {incumbent}, options: {options})")
    res.solution_loader.load_vars()
    bound = res.best_objective_bound
    # HiGHS' own figures, read from the underlying highspy model. The wall-clock
    # time above also counts the translation of the Pyomo model, which dominates on
    # large instances (about 8 s for 40 patients), so the time limit does not bound
    # it. Recording both keeps the reported solve times honest.
    highs = getattr(opt, "_solver_model", None)
    options["highs_internals"] = highs is not None
    options["highs_run_time"] = float(highs.getRunTime()) if highs is not None else float("nan")
    options["highs_mip_gap"] = float(highs.getInfo().mip_gap) if highs is not None else float("nan")
    return SolveReport(status=tc.name, objective=float(incumbent),
                       best_bound=float(bound) if bound is not None else float("nan"),
                       gap=relative_gap(incumbent, bound), solve_time=elapsed, options=options)


#: Fields of `plne_parameters.json`, the only channel from the prediction pipeline
#: to the ILP campaign (audit 2.3). Schema 2 replaces the hand-extracted file of
#: run_20260926_2002, which carried no method, no provenance and K-means centroids.
PLNE_PARAMETER_FIELDS = {
    "schema_version": int, "duration_method": str, "duration_stat": str,
    "cutoff_minutes": float, "s_A": float, "s_B": float,
    "pi_A_actual": float, "pi_A_predicted": float, "noshow_rate": float,
    "group_error_rate": float, "n_train": int, "n_holdout": int,
    "created": str, "git_commit": str, "source_run": str, "smoke": bool,
}
PLNE_SCHEMA_VERSION = 2


def check_plne_parameters(P: dict, slot_lengths: Sequence[float]) -> dict:
    """Validate the ILP parameters exported by the pipeline (audit B4 and 2.3).

    Raises:
        ValueError: a field is missing or has the wrong type, or a value lies
            outside its admissible range: 0.05 < pi_A < 0.5, s_A > s_B > 0, and
            s_A <= max(slot_lengths) (a long patient must fit in the longest slot,
            otherwise the campaign leaves the domain of the model, audit B4).
    """
    missing = [k for k in PLNE_PARAMETER_FIELDS if k not in P]
    if missing:
        raise ValueError(f"plne_parameters: missing fields {missing} (schema {PLNE_SCHEMA_VERSION} "
                         f"expected; rerun the export stage of data_driven_plan.ipynb)")
    for key, typ in PLNE_PARAMETER_FIELDS.items():
        ok = isinstance(P[key], (int, float)) and not isinstance(P[key], bool) if typ is float \
            else isinstance(P[key], typ)
        if not ok:
            raise ValueError(f"plne_parameters: {key}={P[key]!r} is not of type {typ.__name__}")
    if P["schema_version"] != PLNE_SCHEMA_VERSION:
        raise ValueError(f"plne_parameters: schema {P['schema_version']}, expected {PLNE_SCHEMA_VERSION}")
    if not 0.05 < P["pi_A_actual"] < 0.5:
        raise ValueError(f"pi_A = {P['pi_A_actual']:.4f} outside (0.05, 0.5)")
    if not P["s_A"] > P["s_B"] > 0:
        raise ValueError(f"expected s_A > s_B > 0, got s_A={P['s_A']}, s_B={P['s_B']}")
    if P["s_A"] > max(slot_lengths):
        raise ValueError(f"s_A = {P['s_A']:.2f} min exceeds the longest slot {max(slot_lengths)} min "
                         f"(audit B4): change SLOT_LENGTHS or the duration method, do not run")
    for key in ("noshow_rate", "group_error_rate", "pi_A_predicted"):
        if not 0.0 <= P[key] <= 1.0:
            raise ValueError(f"plne_parameters: {key}={P[key]} is not a proportion")
    return P


def load_plne_parameters(path: str, slot_lengths: Sequence[float]) -> dict:
    """Read and validate `plne_parameters.json` (see `check_plne_parameters`)."""
    with open(path, encoding="utf-8") as fh:
        return check_plne_parameters(json.load(fh), slot_lengths)


@dataclass
class Instance:
    """Instance of the outpatient scheduling problem with overbooking (k<=2).

    The three unit costs have no default: every instance states the objective it
    is solved under. c_O = 0 gives the base model of the manuscript, equations (1)
    and (8); c_O > 0 adds the expected overtime term of its section 8.5. The former
    `include_overtime` flag was read nowhere while overtime was always charged
    (audit B5); it is removed.
    """
    n: int
    D: int
    T: list
    L: list
    s: np.ndarray
    q: np.ndarray
    c_I: float
    c_W: float
    c_O: float
    Q_bar: Optional[float] = None

    def __post_init__(self):
        self.s = np.asarray(self.s, dtype=float)
        self.q = np.asarray(self.q, dtype=float)
        if len(self.s) != self.n or len(self.q) != self.n:
            raise ValueError(f"s and q must have n={self.n} entries (got {len(self.s)}, {len(self.q)})")
        if not np.all((self.q > 0) & (self.q < 1)):
            raise ValueError("q must lie in the open interval (0, 1)")
        if min(self.c_I, self.c_W, self.c_O) < 0:
            raise ValueError(f"unit costs must be non-negative (c_I={self.c_I}, c_W={self.c_W}, c_O={self.c_O})")
        if len(self.T) != self.D or any(len(self.L[d]) != self.T[d] for d in range(self.D)):
            raise ValueError("L must list T[d] slot lengths for each of the D days")


def objective_latex(c_I: float, c_W: float, c_O: float) -> str:
    """LaTeX of the objective actually minimised by `solve_ilp` (reduced form).

    The constant c_I * sum_t L_t is dropped, as in equation (9) of the manuscript.
    With c_O = 0 the expression is exactly equation (9); with c_O > 0 the overtime
    terms o_it and the correction -delta_ijt appear.
    """
    def num(v):
        return f"{v:g}".replace(".", "{,}")
    if c_O == 0:
        return (rf"\min_{{x,z}}\; -{num(c_I)} \sum_{{i,t}} a_{{it}}\, x_{{it}} + \sum_{{(i,j),t}}"
                rf"\big({num(c_W)}\, w_{{ij}} - {num(c_I)}\, \delta_{{ijt}}\big)\, z_{{ijt}}")
    return (rf"\min_{{x,z}}\; \sum_{{i,t}} \big({num(c_O)}\, o_{{it}} - {num(c_I)}\, a_{{it}}\big)\, x_{{it}}"
            rf" + \sum_{{(i,j),t}} \big({num(c_W)}\, w_{{ij}} - {num(c_I + c_O)}\, \delta_{{ijt}}\big)\, z_{{ijt}}")


@dataclass
class Config:
    """Solver options for the ILP. Every field is logged in `Solution.options`."""
    z_binary: bool = False
    pruning: bool = False
    pruning_thr: float = 1e-4
    time_limit: float = 120.0
    mip_gap: float = 1e-6
    threads: int = 1               # one HiGHS thread; parallelism is across instances
    require_optimal: bool = False  # raise instead of returning an unproven incumbent
    symmetry_breaking: bool = False  # occupancy non-increasing over identical slots of a day


@dataclass
class Solution:
    """Solution returned by `solve_ilp`."""
    planning: dict
    E_idle: float
    E_wait: float
    E_OT: float
    Z_reduced: float
    Z_complete: float
    solve_time: float
    gap: float              # relative gap on the solver objective (reduced cost)
    n_served: int
    n_double: int
    x_val: dict
    z_val: dict
    coeffs: dict
    status: str = ""        # termination condition read from HiGHS
    best_bound: float = float("nan")    # best bound on the reduced objective
    gap_complete: float = float("nan")  # gap relative to the complete cost Z_complete
    options: dict = field(default_factory=dict)

    @property
    def proven_optimal(self) -> bool:
        return self.status == "optimal"


def compute_coefficients(inst: Instance) -> dict:
    """Pre-compute the a/delta/w/o/gamma coefficients of the ILP, with sign
    assertions guaranteeing the validity of the continuous relaxation of z
    (Proposition 1 of the manuscript).

    Args:
        inst: problem instance.

    Returns:
        dict of coefficients plus the `slots`/`pairs` lists and `total_L`.
    """
    n, D = inst.n, inst.D
    I = range(n)
    slots = [(d, t) for d in range(D) for t in range(inst.T[d])]
    pairs = [(i, j) for i in I for j in I if i < j]

    a, delta, w, o, gamma = {}, {}, {}, {}, {}
    for i in I:
        for (d, t) in slots:
            L = inst.L[d][t]
            a[i, d, t] = inst.q[i] * min(inst.s[i], L)
            o[i, d, t] = inst.q[i] * inst.s[i] - a[i, d, t]

    for (i, j) in pairs:
        si, sj, qi, qj = inst.s[i], inst.s[j], inst.q[i], inst.q[j]
        w[i, j] = qi * qj * min(si, sj)
        for (d, t) in slots:
            L = inst.L[d][t]
            d_ij = qi * qj * (min(si + sj, L) - min(si, L) - min(sj, L))
            assert d_ij <= 1e-9, f"delta[{i},{j},{d},{t}]={d_ij:.4e} > 0"
            delta[i, j, d, t] = d_ij
            gamma[i, j, d, t] = -d_ij

    for (i, j) in pairs:
        for (d, t) in slots:
            cz = inst.c_W * w[i, j] - (inst.c_I + inst.c_O) * delta[i, j, d, t]
            assert cz >= -1e-9, f"coeff z[{i},{j}] = {cz:.4e} < 0"

    total_L = sum(inst.L[d][t] for (d, t) in slots)
    return dict(a=a, delta=delta, w=w, o=o, gamma=gamma, slots=slots, pairs=pairs, total_L=total_L)


def solve_ilp(inst: Instance, cfg: Optional[Config] = None,
              warm_start: Optional[dict] = None) -> Solution:
    """Build and solve the overbooking ILP (k<=2) with Pyomo/HiGHS.

    Args:
        inst: problem instance.
        cfg: solver options (default: Config()).
        warm_start: optional feasible schedule {(d, t): [patients]} (e.g. the
            greedy B3) given to HiGHS as its first incumbent; x and z are set
            consistently (z = 1 for the pair of a double slot).

    Returns:
        Full solution (SPT schedule, E[Idle]/E[Wait]/E[OT] breakdown, etc.).
    """
    if cfg is None:
        cfg = Config()
    coeffs = compute_coefficients(inst)
    a, delta, w, o, gamma = (coeffs[k] for k in ("a", "delta", "w", "o", "gamma"))
    slots, pairs = coeffs["slots"], coeffs["pairs"]

    I = list(range(inst.n))
    active_pairs = [(i, j) for (i, j) in pairs if not cfg.pruning or inst.q[i] * inst.q[j] > cfg.pruning_thr]
    x_idx = [(i, d, t) for i in I for (d, t) in slots]
    z_idx = [(i, j, d, t) for (i, j) in active_pairs for (d, t) in slots]

    m = pyo.ConcreteModel()
    m.x = pyo.Var(x_idx, domain=pyo.Binary, initialize=0)
    m.z = pyo.Var(z_idx, domain=pyo.Binary, initialize=0) if cfg.z_binary else pyo.Var(z_idx, bounds=(0, 1), initialize=0)

    def _obj(m):
        xp = sum((inst.c_O * o[i, d, t] - inst.c_I * a[i, d, t]) * m.x[i, d, t] for (i, d, t) in x_idx)
        zp = sum((inst.c_W * w[i, j] - (inst.c_I + inst.c_O) * delta[i, j, d, t]) * m.z[i, j, d, t] for (i, j, d, t) in z_idx)
        return xp + zp
    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    m.C1 = pyo.ConstraintList()
    for i in I:
        m.C1.add(sum(m.x[i, d, t] for (d, t) in slots) <= 1)

    m.C2 = pyo.ConstraintList()
    for (d, t) in slots:
        m.C2.add(sum(m.x[i, d, t] for i in I) <= 2)

    m.C3 = pyo.ConstraintList()
    for (i, j, d, t) in z_idx:
        m.C3.add(m.z[i, j, d, t] >= m.x[i, d, t] + m.x[j, d, t] - 1)

    if inst.Q_bar is not None:
        m.C5 = pyo.ConstraintList()
        for (d, t) in slots:
            m.C5.add(sum(inst.q[i] * m.x[i, d, t] for i in I) <= inst.Q_bar)

    # Optional symmetry breaking (audit 2.4). Within a day, two slots of the same
    # length L have identical coefficients (a, o, delta, gamma depend on L only) and
    # the same capacity, so their contents can be swapped without changing the cost.
    # Ordering their occupancies (number of patients) non-increasingly along the day
    # keeps at least one optimal solution and removes the permuted copies. It is
    # valid only because the cost of a slot does not depend on its position t.
    m.SB = pyo.ConstraintList()
    if cfg.symmetry_breaking:
        for d in range(inst.D):
            for length in sorted(set(inst.L[d])):
                same = [t for t in range(inst.T[d]) if inst.L[d][t] == length]
                for t, t_next in zip(same, same[1:]):
                    m.SB.add(sum(m.x[i, d, t] for i in I) >= sum(m.x[i, d, t_next] for i in I))

    start = None
    if warm_start is not None:
        start = ComponentMap((m.x[k], 0) for k in x_idx)   # Pyomo variables are not hashable
        start.update((m.z[k], 0) for k in z_idx)
        for (d, t), pats in warm_start.items():
            if len(pats) > 2:
                raise ValueError(f"warm start: slot {(d, t)} holds {len(pats)} patients")
            for i in pats:
                start[m.x[i, d, t]] = 1
            if len(pats) == 2 and (min(pats), max(pats), d, t) in m.z:
                start[m.z[min(pats), max(pats), d, t]] = 1

    report = solve_model(m, cfg, warm_start=start)
    if cfg.require_optimal and not report.proven_optimal:
        raise RuntimeError(f"Optimality not proven: status {report.status}, gap {report.gap:.3e}")

    x_val = {(i, d, t): pyo.value(m.x[i, d, t]) for (i, d, t) in x_idx}
    z_val = {(i, j, d, t): pyo.value(m.z[i, j, d, t]) for (i, j, d, t) in z_idx}

    # Proposition 1 guarantees integral z at the optimum only. A proven optimum
    # with a fractional z is a contradiction and stops the run; an unproven
    # incumbent may carry fractional z, which is why the cost below is recomputed
    # from x alone (z = x_i * x_j).
    n_fractional = sum(1 for v in z_val.values() if v is not None and abs(v - round(v)) > 1e-6)
    if report.proven_optimal and n_fractional:
        raise ValueError(f"{n_fractional} fractional z at a proven optimum: Proposition 1 violated")

    planning = {}
    for (d, t) in slots:
        pats = [i for i in I if (x_val.get((i, d, t), 0) or 0) > 0.5]
        if pats:
            pats.sort(key=lambda i: inst.s[i])
            planning[d, t] = pats

    E_idle = E_wait = E_OT = 0.0
    for (d, t), pats in planning.items():
        L = inst.L[d][t]
        if len(pats) == 1:
            i = pats[0]
            E_idle += inst.c_I * (L - a[i, d, t])
            E_OT += inst.c_O * o[i, d, t]
        elif len(pats) == 2:
            i, j = pats
            if i > j:
                i, j = j, i
            E_idle += inst.c_I * (L - a[i, d, t] - a[j, d, t] - delta[i, j, d, t])
            E_wait += inst.c_W * w[i, j]
            E_OT += inst.c_O * (o[i, d, t] + o[j, d, t] + gamma[i, j, d, t])
    for (d, t) in slots:
        if (d, t) not in planning:
            E_idle += inst.c_I * inst.L[d][t]

    # Exact expected cost of the returned schedule, and its reduced form (the
    # constant c_I * sum(L) removed), which is what the solver minimises.
    const = inst.c_I * coeffs["total_L"]
    Z_complete = E_idle + E_wait + E_OT
    Z_reduced = Z_complete - const
    tol = 1e-6 * max(1.0, abs(report.objective))
    if report.proven_optimal and abs(Z_reduced - report.objective) > tol:
        raise ValueError(f"Recomputed objective {Z_reduced:.6f} differs from the proven optimum "
                         f"{report.objective:.6f}: model and evaluation disagree")
    if Z_reduced > report.objective + tol:
        raise ValueError(f"Recomputed objective {Z_reduced:.6f} exceeds the solver incumbent "
                         f"{report.objective:.6f}, impossible with non-negative z coefficients")

    gap = relative_gap(Z_reduced, report.best_bound)
    lower_complete = report.best_bound + const
    gap_complete = (Z_complete - lower_complete) / max(abs(Z_complete), 1e-9)

    n_served = sum(len(v) for v in planning.values())
    n_double = sum(1 for v in planning.values() if len(v) == 2)

    options = {**report.options, "z_binary": cfg.z_binary, "pruning": cfg.pruning,
               "symmetry_breaking": cfg.symmetry_breaking, "n_symmetry_cuts": len(m.SB),
               "fractional_z": n_fractional}
    return Solution(planning=planning, E_idle=E_idle, E_wait=E_wait, E_OT=E_OT,
                    Z_reduced=Z_reduced, Z_complete=Z_complete, solve_time=report.solve_time, gap=gap,
                    n_served=n_served, n_double=n_double, x_val=x_val, z_val=z_val, coeffs=coeffs,
                    status=report.status, best_bound=report.best_bound, gap_complete=gap_complete,
                    options=options)


def eval_slot_exact(pats: Sequence[int], q: np.ndarray, s: np.ndarray, L: float,
                    spt: bool = True) -> tuple:
    """Expected (idle, wait, overtime) of one slot, in minutes, by enumerating the
    2^k show/no-show scenarios of its k patients, served in SPT order (or in the
    order given when spt=False, used to test the SPT dominance).

    This is the slot-separable evaluation the ILP coefficients encode: every slot
    starts on time and nothing carries over to the next one. Unweighted by costs.
    """
    pats = list(pats)
    k = len(pats)
    e_idle = e_wait = e_ot = 0.0
    for mask in range(1 << k):
        prob = 1.0
        present = []
        for b, i in enumerate(pats):
            if mask >> b & 1:
                prob *= q[i]
                present.append(i)
            else:
                prob *= 1.0 - q[i]
        clock = wait = 0.0
        for i in (sorted(present, key=lambda i: s[i]) if spt else present):
            wait += clock
            clock += s[i]
        e_idle += prob * max(0.0, L - clock)
        e_wait += prob * wait
        e_ot += prob * max(0.0, clock - L)
    return e_idle, e_wait, e_ot


#: Largest number of patients of one day enumerated by `eval_day_exact` (2^n scenarios).
MAX_EXACT_DAY_PATIENTS = 22


def eval_day_exact(slot_patients: Sequence[Sequence[int]], q: np.ndarray, s: np.ndarray,
                   L_slots: Sequence[float]) -> tuple:
    """Expected (idle, wait, overtime) of one day, in minutes, under the dynamics of
    the Lindley simulator, computed exactly by enumerating the 2^n show patterns.

    Patients arrive at the nominal start of their slot and are served in SPT order
    within it; a late slot delays the next ones. Idle time is counted before each
    slot start and after the last patient up to the end of the day (terminal idle,
    audit P1). Durations are deterministic, so the enumeration is exact.

    Raises:
        ValueError: if the day has more than MAX_EXACT_DAY_PATIENTS patients.
    """
    slots = [sorted(p, key=lambda i: s[i]) for p in slot_patients]
    pats = [i for p in slots for i in p]
    if len(pats) > MAX_EXACT_DAY_PATIENTS:
        raise ValueError(f"{len(pats)} patients: 2^n enumeration capped at {MAX_EXACT_DAY_PATIENTS}")
    starts = np.concatenate([[0.0], np.cumsum(L_slots[:-1])])
    day_end = float(sum(L_slots))
    e_idle = e_wait = e_ot = 0.0
    for mask in range(1 << len(pats)):
        show = {i: bool(mask >> b & 1) for b, i in enumerate(pats)}
        prob = float(np.prod([q[i] if show[i] else 1.0 - q[i] for i in pats]))
        prev_end = idle = wait = 0.0
        for t, slot in enumerate(slots):
            idle += max(0.0, starts[t] - prev_end)
            clock = max(prev_end, starts[t])
            for i in slot:
                if show[i]:
                    wait += clock - starts[t]
                    clock += s[i]
            prev_end = clock
        idle += max(0.0, day_end - prev_end)
        e_idle += prob * idle
        e_wait += prob * wait
        e_ot += prob * max(0.0, prev_end - day_end)
    return e_idle, e_wait, e_ot


def _lognormal_params(mean: float, cv: float) -> tuple:
    """(mu, sigma) of the lognormal law with the given mean and coefficient of variation."""
    sigma2 = np.log(1.0 + cv ** 2)
    return np.log(mean) - sigma2 / 2.0, np.sqrt(sigma2)


def simulate_day(slot_patients: Sequence[Sequence[int]], q: np.ndarray, s: np.ndarray,
                 L_slots: Sequence[float], M: int, rng: np.random.Generator,
                 cv: float = 0.0, shows: Optional[np.ndarray] = None,
                 durations: Optional[np.ndarray] = None) -> tuple:
    """Vectorised Lindley simulation of one day over M scenarios.

    Same dynamics as `eval_day_exact`, terminal idle included (audit P1). For
    common random numbers across schedules (audit P3), pass `shows` (bool, shape
    (M, n)) and `durations` (shape (M, n)) drawn once per instance; they are then
    indexed by patient, so two schedules of the same patients face the same draws.

    Returns:
        (idle, wait, ot), three arrays of shape (M,), in minutes.
    """
    starts = np.concatenate([[0.0], np.cumsum(L_slots[:-1])])
    day_end = float(sum(L_slots))
    idle = np.zeros(M)
    wait = np.zeros(M)
    prev_end = np.zeros(M)
    for t, slot in enumerate(slot_patients):
        idle += np.maximum(0.0, starts[t] - prev_end)
        clock = np.maximum(prev_end, starts[t])
        for i in sorted(slot, key=lambda i: s[i]):
            show = shows[:, i] if shows is not None else rng.random(M) < q[i]
            if durations is not None:
                dur = durations[:, i]
            elif cv > 0:
                mu, sig = _lognormal_params(float(s[i]), cv)
                dur = rng.lognormal(mu, sig, M)
            else:
                dur = np.full(M, float(s[i]))
            wait += show * (clock - starts[t])
            clock = clock + show * dur
        prev_end = clock
    idle += np.maximum(0.0, day_end - prev_end)
    ot = np.maximum(0.0, prev_end - day_end)
    return idle, wait, ot


def draw_scenarios(inst: "Instance", M: int, rng: np.random.Generator, cv: float = 0.0) -> tuple:
    """Show indicators and durations of every patient for M scenarios (common random numbers)."""
    shows = rng.random((M, inst.n)) < inst.q[None, :]
    if cv > 0:
        mu, sig = zip(*(_lognormal_params(float(v), cv) for v in inst.s))
        durations = rng.lognormal(np.array(mu)[None, :], np.array(sig)[None, :], (M, inst.n))
    else:
        durations = np.broadcast_to(inst.s[None, :].astype(float), (M, inst.n))
    return shows, durations


def simulate_instance(inst: "Instance", planning: dict, M: int, rng: np.random.Generator,
                      cv: float = 0.0, scenarios: Optional[tuple] = None) -> np.ndarray:
    """Weighted cost c_I*idle + c_W*wait + c_O*ot summed over the days, per scenario.

    Returns the (M,) array of scenario costs, so that callers can report a mean,
    a confidence interval or a paired test. With `scenarios` (from `draw_scenarios`)
    every schedule of the same instance is evaluated on the same draws.
    """
    shows, durations = scenarios if scenarios is not None else (None, None)
    cost = np.zeros(M)
    for d in range(inst.D):
        slot_patients = [planning.get((d, t), []) for t in range(inst.T[d])]
        idle, wait, ot = simulate_day(slot_patients, inst.q, inst.s, inst.L[d], M, rng, cv,
                                      shows=shows, durations=durations)
        cost += inst.c_I * idle + inst.c_W * wait + inst.c_O * ot
    return cost


def evaluate_planning_exact(inst: "Instance", planning: dict, day_dynamics: bool = True) -> dict:
    """Exact expected cost of a schedule.

    day_dynamics=False: slot-separable evaluation (what the ILP optimises).
    day_dynamics=True: Lindley day with carry-over and terminal idle (what the
    simulator estimates), exact by 2^n enumeration per day.
    """
    E_idle = E_wait = E_OT = 0.0
    for d in range(inst.D):
        if day_dynamics:
            slot_patients = [planning.get((d, t), []) for t in range(inst.T[d])]
            ei, ew, eo = eval_day_exact(slot_patients, inst.q, inst.s, inst.L[d])
        else:
            ei = ew = eo = 0.0
            for t in range(inst.T[d]):
                si, sw, so = eval_slot_exact(planning.get((d, t), []), inst.q, inst.s, inst.L[d][t])
                ei, ew, eo = ei + si, ew + sw, eo + so
        E_idle += ei
        E_wait += ew
        E_OT += eo
    return {"E_idle": E_idle, "E_wait": E_wait, "E_OT": E_OT,
            "Z": inst.c_I * E_idle + inst.c_W * E_wait + inst.c_O * E_OT}


def build_instance_from_patients(
    df_scored: pd.DataFrame, T: list[int], L: list[list[float]],
    s_A: float, s_B: float, c_I: float, c_W: float, c_O: float,
    group_col: str = "group", q_col: str = "q_i", positive_group: str = "A",
) -> Instance:
    """Build an ILP Instance from ACTUALLY predicted patients (never seen during
    training), bridging from `predict_patient`.

    Args:
        df_scored: DataFrame returned by `predict_patient` (columns `q_col`,
            `group_col`), one patient per row.
        T: number of slots per day, e.g. [4] for a single 4-slot day.
        L: duration (minutes) of each slot, e.g. [[30.,30.,60.,60.]].
        s_A: representative service duration of group A (long).
        s_B: representative service duration of group B (short).
        c_I, c_W, c_O: unit idle/wait/overtime costs.
        group_col: column of `df_scored` holding the predicted group.
        q_col: column of `df_scored` holding the predicted show-up probability.
        positive_group: value of `group_col` denoting the "long" group (s_A).

    Returns:
        Instance ready for `solve_ilp`.
    """
    n = len(df_scored)
    s = np.where(df_scored[group_col].to_numpy() == positive_group, s_A, s_B)
    q = np.clip(df_scored[q_col].to_numpy(dtype=float), 0.01, 0.99)
    return Instance(n=n, D=len(T), T=T, L=L, s=s, q=q, c_I=c_I, c_W=c_W, c_O=c_O)
