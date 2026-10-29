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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
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
    accuracy_score, precision_score, recall_score, log_loss,
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


def available_cpus(cgroup_root: str = "/sys/fs/cgroup") -> int:
    """CPUs this process may actually use: the affinity mask, capped by the cgroup CPU
    quota of a container (RunPod: 16 vCPU while the host reports 24 cores / 48 threads).

    os.cpu_count() and psutil report the host; using them in a container
    oversubscribes the allotted CPUs. A missing cgroup file means no quota.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    j = os.path.join
    quotas = [(j(cgroup_root, "cpu.max"), None),                       # cgroup v2: "<quota> <period>"
              (j(cgroup_root, "cpu", "cpu.cfs_quota_us"), j(cgroup_root, "cpu", "cpu.cfs_period_us"))]  # v1
    for path, period_path in quotas:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            parts = fh.read().split()
        if period_path is None:
            quota, period = parts[0], parts[1]
        else:
            with open(period_path) as fh:
                quota, period = parts[0], fh.read().strip()
        if quota not in ("max", "-1"):
            n = min(n, max(1, int(int(quota) // int(period))))
    return int(n)


def available_memory(cgroup_root: str = "/sys/fs/cgroup") -> int:
    """Bytes this process may still allocate: the free memory of the machine, capped
    by the cgroup memory limit of a container minus what the container already uses
    (psutil reads /proc/meminfo, which shows the host inside a container)."""
    import psutil
    avail = int(psutil.virtual_memory().available)
    j = os.path.join
    for limit_path, usage_path in ((j(cgroup_root, "memory.max"), j(cgroup_root, "memory.current")),
                                   (j(cgroup_root, "memory", "memory.limit_in_bytes"),
                                    j(cgroup_root, "memory", "memory.usage_in_bytes"))):
        if not (os.path.exists(limit_path) and os.path.exists(usage_path)):
            continue
        with open(limit_path) as fh:
            limit = fh.read().strip()
        with open(usage_path) as fh:
            usage = int(fh.read().strip())
        if limit != "max" and int(limit) < 2 ** 60:
            avail = min(avail, int(limit) - usage)
    return max(avail, 0)


def check_free_disk(path: str, need_gb: float) -> float:
    """Stop before a stage writes artifacts the disk cannot hold (the RunPod container
    disk is 5 GB; the artifacts of a full run take about 70 GB). Returns the free GB."""
    import shutil
    os.makedirs(path, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 2 ** 30
    if free_gb < need_gb:
        raise RuntimeError(f"{free_gb:.1f} GB free under {os.path.abspath(path)}, {need_gb:.0f} GB needed: "
                           f"run from a volume with enough space (e.g. /workspace on RunPod)")
    print(f"Disk: {free_gb:.1f} GB free under {os.path.abspath(path)} ({need_gb:.0f} GB needed)")
    return free_gb


def resolve_n_jobs(n_jobs: int) -> int:
    """n_jobs as an explicit count: -1 means every available CPU (see `available_cpus`)."""
    if n_jobs == -1:
        return available_cpus()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be -1 or positive, got {n_jobs}")
    return min(int(n_jobs), available_cpus())


def latex_table(df: pd.DataFrame, path: str, caption: str, label: str,
                decimals: int = 3, index: bool = False) -> str:
    """Write a booktabs LaTeX table with escaped headers and cells (audit rec. 14).

    Column names such as "E[Idle] (min)", "q_i" or "Écart (%)" are escaped
    (underscores, percent signs, ampersands), floats rounded to `decimals`,
    negative zeros printed as zeros. The caption is written as given (it may
    hold math).
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(decimals) + 0.0
    tex = out.to_latex(index=index, escape=True, float_format=f"%.{decimals}f",
                       caption=caption, label=label, position="htbp")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    return tex


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


def load_raw_data(path: str, patient_fraction: Optional[float] = None, key: str = "smoke",
                  patient_col: str = "patientsid") -> pd.DataFrame:
    """Read the raw Parquet data, optionally for a fraction of the patients only.

    patient_fraction (smoke test): patients are kept when a 64-bit hash of
    their identifier falls below the fraction, so every visit of a kept
    patient is kept and the selection does not depend on the row order.
    """
    import pyarrow.dataset as pads
    dset = pads.dataset(path)
    if patient_fraction is None:
        return dset.to_table().to_pandas()
    if not 0 < patient_fraction < 1:
        raise ValueError(f"patient_fraction={patient_fraction} outside (0, 1)")
    ids = pd.Series(dset.to_table(columns=[patient_col]).column(0).to_numpy()).drop_duplicates()
    keep = ids[_uniform_from_id(ids, key) < patient_fraction].to_numpy()
    table = dset.to_table(filter=pads.field(patient_col).isin(keep))
    print(f"Patient sample: {len(keep):,} / {len(ids):,} patients ({patient_fraction:.1%}), "
          f"{table.num_rows:,} rows")
    return table.to_pandas()


#: Stages of the main notebook (audit 1.7), in execution order.
STAGES: tuple[str, ...] = ("prep", "durations", "tune", "fit", "calibrate", "evaluate", "export")


class StageRunner:
    """Resumable execution of the main notebook (audit 1.7).

    Each stage writes its artifacts to models/run_<id>/<stage>/ with a `.done`
    marker holding the hash of its configuration (chained with the hashes of
    the stages before it). A stage whose marker matches is loaded instead of
    recomputed; stages after `target` are neither computed nor loaded.
    runs/<id>/STATUS.md records start, end, duration, peak memory and artifact
    of every stage.
    """

    def __init__(self, run_id: str, target: str = "all", root: str = MODELS_DIR, status_dir: str = "runs"):
        if target != "all" and target not in STAGES:
            raise ValueError(f"STAGE={target!r}, expected 'all' or one of {STAGES}")
        self.run_id, self.target = run_id, target
        self.dir = os.path.join(root, f"run_{run_id}")
        self.status_path = os.path.join(status_dir, run_id, "STATUS.md")
        self.cfg, self.hash, self._t0 = {}, {}, {}
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
        self._rows = self._read_status()

    def wanted(self, stage: str) -> bool:
        last = len(STAGES) - 1 if self.target == "all" else STAGES.index(self.target)
        return STAGES.index(stage) <= last

    def configure(self, stage: str, **cfg) -> str:
        k = STAGES.index(stage)
        upstream = self.hash.get(STAGES[k - 1], "") if k else ""
        self.cfg[stage] = {"stage": stage, "upstream": upstream, **cfg}
        blob = json.dumps(self.cfg[stage], sort_keys=True, default=str)
        self.hash[stage] = __import__("hashlib").sha1(blob.encode()).hexdigest()[:16]
        return self.hash[stage]

    def _marker(self, stage: str) -> str:
        return os.path.join(self.dir, stage, ".done")

    def done(self, stage: str) -> bool:
        path = self._marker(stage)
        if stage not in self.hash or not os.path.exists(path):
            return False
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("hash") == self.hash[stage]

    def compute(self, stage: str) -> bool:
        """True when the stage must run now (wanted, configured, not done)."""
        if stage not in self.hash:
            raise RuntimeError(f"stage {stage!r} used before RUN.configure")
        todo = self.wanted(stage) and not self.done(stage)
        if todo and stage not in self._t0:
            self._t0[stage] = time.time()
            self._status(stage, "running")
        elif self.wanted(stage) and not todo:
            print(f"[{stage}] artifacts found for this configuration: loaded, not recomputed")
        return todo

    def finish(self, stage: str, **artifacts) -> None:
        d = os.path.join(self.dir, stage)
        os.makedirs(d, exist_ok=True)
        big = sum(v.nbytes for v in artifacts.values() if isinstance(v, np.ndarray) and v.ndim == 2)
        if big:
            check_free_disk(d, 1.1 * big / 2 ** 30 + 1)
        small = {}
        for k, v in artifacts.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                np.save(os.path.join(d, f"{k}.npy"), np.ascontiguousarray(v))
            else:
                small[k] = v
        joblib.dump(small, os.path.join(d, "artifacts.joblib"))
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(self.cfg[stage], fh, indent=2, default=str)
        with open(self._marker(stage), "w", encoding="utf-8") as fh:
            json.dump({"hash": self.hash[stage], "finished": pd.Timestamp.now().isoformat(timespec="seconds")}, fh)
        self._status(stage, "done", artifact=d)

    def load(self, stage: str) -> dict:
        d = os.path.join(self.dir, stage)
        if not self.done(stage):
            raise RuntimeError(f"stage {stage!r} has no artifacts for this configuration")
        out = joblib.load(os.path.join(d, "artifacts.joblib"))
        for f in os.listdir(d):
            if f.endswith(".npy"):
                out[f[:-4]] = np.load(os.path.join(d, f), mmap_mode="r")
        return out

    def path(self, stage: str, name: str) -> str:
        d = os.path.join(self.dir, stage)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)

    def _read_status(self) -> dict:
        return {}

    def _status(self, stage: str, state: str, artifact: str = "") -> None:
        import resource
        now = time.time()
        row = self._rows.get(stage, {"start": now})
        row.update(state=state, artifact=artifact or row.get("artifact", ""),
                   peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        if state == "done":
            row["end"] = now
            row["start"] = self._t0.get(stage, row["start"])
        self._rows[stage] = row
        fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
        lines = [f"# Run {self.run_id}", "", f"Target stage: {self.target}", "",
                 "| Stage | State | Start | End | Duration (s) | Peak RSS (GB) | Artifact |",
                 "|---|---|---|---|---|---|---|"]
        for st in STAGES:
            if st in self._rows:
                r = self._rows[st]
                end = r.get("end")
                lines.append(f"| {st} | {r['state']} | {fmt(r['start'])} | {fmt(end) if end else ''} | "
                             f"{(end - r['start']):.0f} | {r['peak_gb']:.2f} | {r['artifact']} |"
                             if end else f"| {st} | {r['state']} | {fmt(r['start'])} |  |  | {r['peak_gb']:.2f} |  |")
        with open(self.status_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


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


@dataclass
class MasterSplit:
    """Disjoint partitions of the working data (audit rec. 13).

    `train` (70 %) is the union of `fit` (tuning and fitting), `valid` (model
    selection inside a family: hyperparameters, early stopping, boosting
    iterations) and `calib` (probability calibration, choice of the calibration
    method and of the model, decision thresholds). `test` (20 %) is used for the
    final metrics only, `inference` (10 %) is the ILP holdout.
    """
    data: pd.DataFrame
    train: pd.DataFrame
    fit: pd.DataFrame
    valid: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame
    inference: pd.DataFrame
    mode: str

    def sizes(self) -> pd.DataFrame:
        parts = {"fit": self.fit, "valid": self.valid, "calib": self.calib,
                 "test": self.test, "inference": self.inference}
        total = sum(len(d) for d in parts.values())
        return pd.DataFrame([{"Partition": k, "Rows": len(d), "Share": len(d) / total,
                              "NoShow rate": d["NoShow"].mean()} for k, d in parts.items()])


def _uniform_from_id(ids: pd.Series, key: str) -> np.ndarray:
    """Deterministic U(0, 1) per identifier: 64-bit hash of (id, key) / 2^64."""
    h = pd.util.hash_pandas_object(ids.astype("int64"), index=False, hash_key=key.ljust(16, "0")[:16])
    return h.to_numpy(dtype=np.uint64) / float(2 ** 64)


def make_master_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    test_size: float = 0.20,
    inference_size: float = 0.10,
    inner_sizes: tuple = (0.80, 0.10, 0.10),
    split_mode: str = "stratified",
    patient_col: str = "patientsid",
    noshow_col: str = "NoShow",
    service_time_col: str = "service_time",
    n_quantile_bins: int = 10,
    random_state: int = RANDOM_STATE,
) -> MasterSplit:
    """Single leak-proof partition, shared by both ML pipelines.

    The whole DataFrame is partitioned (no row discarded) into train / test /
    inference holdout (70 / 20 / 10 % by default), then train into fit / valid
    / calib (80 / 10 / 10 % of train by default, audit rec. 13: the decision
    threshold and the calibration were previously fitted on the test set).

    split_mode="stratified" (default, as in run_20260926_2002): row-level split
    stratified jointly on the no-show target and on service-time quantile bins,
    with the same seeds as before for train / test / inference.
    split_mode="patient_group" (ablation): every visit of a patient falls in the
    same partition, assigned from a hash of the patient identifier; proportions
    then hold in expectation only and stratification is not enforced.

    Returns:
        MasterSplit; the indices of the partitions are subsets of `data.index`.
    """
    if abs((train_size + test_size + inference_size) - 1.0) > 1e-9:
        raise ValueError(f"train_size + test_size + inference_size must equal 1.0 (got "
                         f"{train_size + test_size + inference_size})")
    if abs(sum(inner_sizes) - 1.0) > 1e-9 or len(inner_sizes) != 3:
        raise ValueError(f"inner_sizes (fit, valid, calib) must sum to 1.0, got {inner_sizes}")
    if split_mode not in ("stratified", "patient_group"):
        raise ValueError(f"split_mode={split_mode!r}")

    df_s = df.reset_index(drop=True)
    print(f"Working data: {len(df_s):,} rows, split mode {split_mode}")
    st_bin = pd.qcut(df_s[service_time_col], q=n_quantile_bins, duplicates="drop").astype(str)
    strat_key = df_s[noshow_col].astype(str) + "_" + st_bin

    if split_mode == "stratified":
        df_inf, df_mod = train_test_split(
            df_s, test_size=(1.0 - inference_size), random_state=random_state, stratify=strat_key)
        train_df, test_df = train_test_split(
            df_mod, test_size=test_size / (train_size + test_size), random_state=random_state,
            stratify=strat_key.loc[df_mod.index])
        fit_frac, valid_frac, calib_frac = inner_sizes
        fit_df, rest = train_test_split(train_df, train_size=fit_frac, random_state=random_state,
                                        stratify=strat_key.loc[train_df.index])
        valid_df, calib_df = train_test_split(rest, test_size=calib_frac / (valid_frac + calib_frac),
                                              random_state=random_state,
                                              stratify=strat_key.loc[rest.index])
    else:
        if patient_col not in df_s.columns:
            raise ValueError(f"split_mode='patient_group' needs the column {patient_col!r}, absent "
                             f"(a preprocessed file from an earlier run drops it)")
        u = _uniform_from_id(df_s[patient_col], f"split{random_state}")
        v = _uniform_from_id(df_s[patient_col], f"inner{random_state}")
        part = np.where(u < inference_size, "inference",
                        np.where(u < inference_size + test_size, "test", "train"))
        inner = np.where(v < inner_sizes[0], "fit", np.where(v < inner_sizes[0] + inner_sizes[1], "valid", "calib"))
        df_inf, test_df = df_s[part == "inference"], df_s[part == "test"]
        train_df = df_s[part == "train"]
        tr_inner = inner[part == "train"]
        fit_df, valid_df, calib_df = (train_df[tr_inner == k] for k in ("fit", "valid", "calib"))
        shared = set(train_df[patient_col]) & (set(test_df[patient_col]) | set(df_inf[patient_col]))
        if shared:
            raise AssertionError(f"{len(shared)} patients in both train and test/inference")

    # train rows ordered fit | valid | calib: slices of the training matrix are then
    # views (no copy of a matrix of tens of millions of rows)
    train_df = pd.concat([fit_df, valid_df, calib_df])
    split = MasterSplit(data=df_s, train=train_df, fit=fit_df, valid=valid_df, calib=calib_df,
                        test=test_df, inference=df_inf, mode=split_mode)
    print(split.sizes().to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ensure_output_dirs()
    inf_path = f"{INFERENCE_DIR}/inference_set_raw.csv"
    df_inf.to_csv(inf_path, index=False)
    print(f"  Inference holdout saved -> {inf_path}")
    return split


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
                          "completed", "discontinued", "dts", "dta", "patientsid"}

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
#   y = 1  -> class of interest (rare event: no-show, or "long" group A)
#   y = 0  -> reference class (majority)
#
# Partitions (audit rec. 13): hyperparameters, boosting iterations and early
# stopping are chosen on `valid`; the final model is refitted on fit + valid;
# calibration, model choice and decision thresholds use `calib`; `test` gives
# the reported metrics and nothing else.
#
# Imbalance (same mechanism as the first paper, Fall et al., section II.D:
# SMOTE combined with class weighting): every family is an imbalanced-learn
# pipeline that applies SMOTE at fit time only (sampling_strategy <= 0.1, so the
# minority reaches at most 10 % of the majority; skipped when it already does),
# followed by the classifier weighted to balance the RESAMPLED classes:
# class_weight="balanced" (RandomForest), scale_pos_weight = n_neg / n_pos
# (XGBoost), balanced sample_weight (AdaBoost, MLP). The probabilities of such a
# model refer to balanced classes; `prior_correct` maps them back to the true
# base rate analytically.
# ----------------------------------------------------------------------------

SUPPORTED_MODELS: tuple[str, ...] = ("RandomForest", "XGBoost", "AdaBoost", "ANN")

#: Grids of the successive-halving search (audit 1.4: max_samples and
#: min_samples_leaf bound the cost of the forest; batch_size the MLP's).
#: AdaBoost is not searched by halving: one fit per learning rate at the maximal
#: number of estimators, the number of iterations being read on `valid` with
#: staged_predict_proba (see `TrainingConfig.adaboost_*`).
DEFAULT_PARAM_GRIDS: dict[str, dict] = {
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [20, 50, 100, 200],
        "max_samples": [0.1, 0.2, 0.3],
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
    "AdaBoost": {},
    "ANN": {
        "hidden_layer_sizes": [(128, 64), (256, 128)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-3, 3e-4],
        "batch_size": [2048, 4096, 8192],
    },
}

#: Fixed hyperparameters of each family (not searched).
DEFAULT_BASE_PARAMS: dict[str, dict] = {
    "RandomForest": {"bootstrap": True},
    "XGBoost": {"eval_metric": "logloss", "verbosity": 0, "tree_method": "hist"},
    "AdaBoost": {},
    "ANN": {"solver": "adam", "max_iter": 200, "early_stopping": True,
            "validation_fraction": 0.1, "n_iter_no_change": 5},
}

#: Decision-threshold criteria. Youden's J is kept (audit decision D6); "fixed"
#: imposes a threshold. Thresholds are always chosen on the calib partition.
THRESHOLD_METRICS: tuple[str, ...] = ("youden", "fixed")
SEARCH_STRATEGIES: tuple[str, ...] = ("halving", "none")
PIPE_STEP = "clf"   # name of the classifier step inside the resampling pipeline

AUC_COL = "AUC-ROC"
AP_COL = "AUC-PR"
LOGLOSS_COL = "Log-loss"
BRIER_COL = "Brier"
BSS_COL = "BSS"
ACC_COL = "Accuracy"
BASE_ACC_COL = "Accuracy (majority class)"
PREC_COL = "Precision (macro)"
REC_COL = "Recall (macro)"
F1_COL = "F1 (macro)"
PREC_POS_COL = "Precision (positive)"
REC_POS_COL = "Recall (positive)"
F1_POS_COL = "F1 (positive)"
THR_COL = "Threshold"
REPORTED_METRICS = [AUC_COL, AP_COL, ACC_COL, PREC_COL, REC_COL, F1_COL, F1_POS_COL]
METRIC_DECIMALS = 2


class BalancedFit(ClassifierMixin, BaseEstimator):
    """Fit `estimator` with weights balancing the classes of the y it receives.

    Inside the resampling pipeline the y received is the one AFTER SMOTE, so the
    weights balance the resampled classes. mode="sample_weight" passes balanced
    sample weights to fit (AdaBoost, MLP); mode="scale_pos_weight" sets
    XGBoost's scale_pos_weight to n_neg / n_pos.
    """

    def __init__(self, estimator=None, mode: str = "sample_weight"):
        self.estimator = estimator
        self.mode = mode

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.utils.class_weight import compute_sample_weight
        y = np.asarray(y)
        est = clone(self.estimator)
        if self.mode == "sample_weight":
            est.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        elif self.mode == "scale_pos_weight":
            est.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
            est.fit(X, y)
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        self.estimator_ = est
        self.classes_ = est.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)

    def staged_predict_proba(self, X):
        return self.estimator_.staged_predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


class CachedSMOTE(SMOTE):
    """SMOTE whose synthetic rows are memoised on disk (pod readiness).

    SMOTE is deterministic given (X, y, parameters, random_state), and the same
    resampling is requested many times: by the four families refitted on fit +
    valid, by every candidate of a halving iteration (they share the subsample),
    by AdaBoost. Its nearest-neighbour search grows with the square of the
    minority size (about 40 s for 40,000 minority rows in 200 dimensions), so at
    full scale one call takes hours. The cache key is a BLAKE2b hash of the exact
    bytes of X and y and of the parameters; on a hit the stored synthetic rows
    are appended to X exactly as SMOTE appends them, so the output is identical
    bit for bit (tests/test_pod_resources.py). cache_dir=None disables the cache.
    """
    _parameter_constraints = {**SMOTE._parameter_constraints, "cache_dir": [str, None]}

    def __init__(self, *, sampling_strategy="auto", random_state=None, k_neighbors=5, cache_dir=None):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state,
                         k_neighbors=k_neighbors)
        self.cache_dir = cache_dir

    def _key(self, X, y) -> str:
        import hashlib
        h = hashlib.blake2b(digest_size=20)
        X = np.ascontiguousarray(X)
        h.update(str((X.shape, X.dtype.str, self.sampling_strategy, self.random_state,
                      self.k_neighbors)).encode())
        h.update(memoryview(X).cast("B"))
        h.update(np.ascontiguousarray(np.asarray(y)).tobytes())
        return h.hexdigest()

    def fit_resample(self, X, y, **params):
        if self.cache_dir is None:
            return super().fit_resample(X, y, **params)
        X = np.asarray(X)
        y = np.asarray(y)
        path = os.path.join(self.cache_dir, f"smote_{self._key(X, y)}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                X_new, y_new = z["X_new"], z["y_new"]
        else:
            X_res, y_res = super().fit_resample(X, y, **params)
            X_new, y_new = X_res[len(X):], y_res[len(y):]
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = path + f".{os.getpid()}.tmp.npz"
            np.savez(tmp, X_new=X_new, y_new=y_new)
            os.replace(tmp, path)          # atomic: concurrent workers never read a partial file
            return X_res, y_res
        return np.vstack([X, X_new.astype(X.dtype, copy=False)]), np.concatenate([y, y_new.astype(y.dtype, copy=False)])


def prior_correct(p: np.ndarray, prior: float, train_prior: float = 0.5) -> np.ndarray:
    """Bayes correction of probabilities learnt under a class prior `train_prior`
    back to the true prior `prior` (Saerens et al., 2002). Balanced weighting
    after resampling makes the effective training prior 0.5."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    a = p * prior / train_prior
    b = (1 - p) * (1 - prior) / (1 - train_prior)
    return a / (a + b)


def corrected_log_loss(y: np.ndarray, p: np.ndarray, prior: float) -> float:
    return float(log_loss(y, prior_correct(p, prior), labels=[0, 1]))


def _corrected_log_loss_scorer(prior: float):
    """Scorer (higher is better) for the search: minus the log-loss of the
    prior-corrected probabilities (audit decision D4)."""
    def scorer(estimator, X, y):
        return -corrected_log_loss(y, estimator.predict_proba(X)[:, 1], prior)
    return scorer


@dataclass
class TrainingConfig:
    """Control panel of the classifier training; every field appears in `summary()`.

    Attributes:
        models: families to train, among `SUPPORTED_MODELS`.
        search: "halving" (HalvingRandomSearchCV on the predefined fit / valid
            split, resource = training rows) or "none" (base parameters).
        halving_iterations: iterations of successive halving (k).
        last_candidates: candidates compared at the last iteration, on all of fit.
        min_resources: fit rows of the first iteration (audit 1.3: 500,000).
        max_resources: fit rows of the last iteration; None = all of fit.
            The number of candidates and the halving factor follow from these
            four values (`halving_schedule`): every iteration then costs about
            last_candidates full fits, the budget split of successive halving.
        memory_safety: share of the available memory the parallel search may use;
            the number of search processes is capped accordingly (`search_jobs`).
        smote_sampling_strategy: minority / majority ratio reached by SMOTE
            (<= 0.1, audit 1.4); SMOTE is skipped when the ratio already exceeds it.
        smote_k_neighbors: neighbours used by SMOTE.
        adaboost_max_estimators: the single AdaBoost fit per learning rate.
        adaboost_learning_rates: learning rates tried for AdaBoost.
        param_grids / base_params: per-model overrides of the defaults.
        threshold_metric: criterion of the reported decision threshold.
        n_jobs: processes of the search and threads of the parallel models.
    """
    models: Sequence[str] = SUPPORTED_MODELS
    search: str = "halving"
    halving_iterations: int = 3
    last_candidates: int = 2
    min_resources: int = 500_000
    max_resources: Optional[int] = None
    memory_safety: float = 0.6
    smote_sampling_strategy: float = 0.1
    smote_k_neighbors: int = 5
    smote_cache_dir: Optional[str] = None   # memoised SMOTE output (identical results, see CachedSMOTE)
    adaboost_max_estimators: int = 400
    adaboost_learning_rates: tuple = (0.5, 1.0)
    param_grids: dict = field(default_factory=dict)
    base_params: dict = field(default_factory=dict)
    threshold_metric: str = "youden"
    fixed_threshold: float = 0.5
    n_jobs: int = -1
    verbose: int = 1
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        unknown = [m for m in self.models if m not in SUPPORTED_MODELS]
        if unknown:
            raise ValueError(f"Unknown models {unknown}, expected among {list(SUPPORTED_MODELS)}")
        if self.search not in SEARCH_STRATEGIES:
            raise ValueError(f"search={self.search!r}, expected among {list(SEARCH_STRATEGIES)}")
        if self.threshold_metric not in THRESHOLD_METRICS:
            raise ValueError(f"threshold_metric={self.threshold_metric!r}, expected among {list(THRESHOLD_METRICS)}")
        if not 0 < self.smote_sampling_strategy <= 0.1:
            raise ValueError("smote_sampling_strategy must lie in (0, 0.1] (audit 1.4)")
        self.n_jobs = resolve_n_jobs(self.n_jobs)   # explicit count: -1 would mean the host's CPUs
        self.models = tuple(self.models)

    def grid_for(self, name: str) -> dict:
        grid = copy.deepcopy(DEFAULT_PARAM_GRIDS.get(name, {}))
        grid.update(copy.deepcopy(self.param_grids.get(name, {})))
        return grid

    def base_for(self, name: str) -> dict:
        base = copy.deepcopy(DEFAULT_BASE_PARAMS.get(name, {}))
        base.update(copy.deepcopy(self.base_params.get(name, {})))
        return base

    def replace(self, **overrides) -> "TrainingConfig":
        known = {f.name for f in dataclasses.fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return dataclasses.replace(copy.deepcopy(self), **copy.deepcopy(overrides))

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def summary(self) -> pd.DataFrame:
        rows = [
            ("Models trained", ", ".join(self.models)),
            ("Search", f"{self.search}: {self.halving_iterations} iterations, {self.last_candidates} candidates "
                       f"at the last one, fit rows {self.min_resources:,} -> "
                       f"{'all of fit' if self.max_resources is None else f'{self.max_resources:,}'}"),
            ("Search processes", f"capped so that they use at most {self.memory_safety:.0%} of the available memory"),
            ("Search criterion", "log-loss of prior-corrected probabilities on valid"),
            ("Final fit", "fit + valid, best parameters"),
            ("Imbalance", f"SMOTE(sampling_strategy={self.smote_sampling_strategy}, k={self.smote_k_neighbors}) "
                          f"at fit time + balanced weights"),
            ("AdaBoost", f"one fit per learning rate {self.adaboost_learning_rates} at "
                         f"{self.adaboost_max_estimators} estimators, iterations chosen on valid"),
            ("Threshold criterion", f"{self.threshold_metric} on calib"),
            ("n_jobs / seed", f"{self.n_jobs} / {self.random_state}"),
        ]
        rows += [(f"Grid: {m}", str(self.grid_for(m))) for m in self.models]
        rows += [(f"Fixed: {m}", str(self.base_for(m))) for m in self.models if self.base_for(m)]
        return pd.DataFrame(rows, columns=["Setting", "Value"])


class ModelSet(dict):
    """dict {name: model refitted on fit + valid}, with the search report."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_report = pd.DataFrame()
        self.config = None
        self.best_params = {}
        self.prior = float("nan")


def _wrapped(name: str) -> bool:
    return name in ("XGBoost", "AdaBoost", "ANN")


def _prefixed(name: str, params: dict) -> dict:
    """Grid keys as seen from the pipeline (step prefix, plus the wrapper's)."""
    pre = f"{PIPE_STEP}__estimator__" if _wrapped(name) else f"{PIPE_STEP}__"
    return {(k if k.startswith(PIPE_STEP + "__") else pre + k): v for k, v in params.items()}


def make_estimator(name: str, cfg: TrainingConfig, minority_ratio: float,
                   params: Optional[dict] = None, n_jobs: Optional[int] = None):
    """Untrained pipeline [SMOTE] -> classifier for family `name` (see the
    imbalance note at the top of this section).

    Args:
        minority_ratio: n_pos / n_neg of the training target; SMOTE is added
            only when it is below `cfg.smote_sampling_strategy`.
        params: hyperparameters (unprefixed), applied over the base ones.
        n_jobs: threads of the model; None = cfg.n_jobs.
    """
    threads = cfg.n_jobs if n_jobs is None else n_jobs
    rs = cfg.random_state
    base = cfg.base_for(name)
    if name == "RandomForest":
        clf = RandomForestClassifier(random_state=rs, n_jobs=threads, class_weight="balanced", **base)
    elif name == "XGBoost":
        clf = BalancedFit(XGBClassifier(random_state=rs, n_jobs=threads, **base), mode="scale_pos_weight")
    elif name == "AdaBoost":
        kw = {"algorithm": "SAMME"} if "algorithm" in inspect.signature(AdaBoostClassifier.__init__).parameters else {}
        clf = BalancedFit(AdaBoostClassifier(random_state=rs, **kw, **base), mode="sample_weight")
    elif name == "ANN":
        clf = BalancedFit(MLPClassifier(random_state=rs, **base), mode="sample_weight")
    else:  # pragma: no cover - guarded by TrainingConfig
        raise ValueError(f"Unknown model: {name}")
    steps = []
    if minority_ratio < cfg.smote_sampling_strategy:
        steps.append(("smote", CachedSMOTE(sampling_strategy=cfg.smote_sampling_strategy,
                                           k_neighbors=cfg.smote_k_neighbors, random_state=rs,
                                           cache_dir=cfg.smote_cache_dir)))
    steps.append((PIPE_STEP, clf))
    pipe = ImbPipeline(steps)
    if params:
        pipe.set_params(**_prefixed(name, params))
    return pipe


def halving_schedule(n_grid: int, n_rows: int, frac_fit: float, cfg: "TrainingConfig") -> dict:
    """Explicit successive-halving schedule (audit 1.3: from 500,000 rows to all of fit).

    With k iterations and L candidates at the last one, the factor f and the
    number of candidates n follow: rows grow from r_min to r_max = all rows and
    candidates shrink from n to L by the same factor f per iteration, so
    f^(k-1) = r_max / r_min = n / L and every iteration trains about L full fits
    (the equal-budget split of successive halving). When the grid holds fewer
    than L r_max / r_min combinations, n is the grid size and r_min grows
    accordingly. Rows are those of fit + valid (the resource of the search); the
    fit share of every subsample stays frac_fit.

    Returns the parameters for HalvingRandomSearchCV and the schedule it will
    follow, recomputed with scikit-learn's own formulas.
    """
    import math
    k, L = cfg.halving_iterations, cfg.last_candidates
    if k < 2 or L < 1:
        raise ValueError("halving_iterations must be >= 2 and last_candidates >= 1")
    r_max = n_rows if cfg.max_resources is None else min(n_rows, int(cfg.max_resources / frac_fit))
    r_min_target = min(int(cfg.min_resources / frac_fit), r_max)
    # Rows grow by the integer ratio q from first to last iteration. scikit-learn counts
    # iterations with floor(log(r_max // r_min, f)) and caps the last rows at r_max:
    # with r_min = r_max // q and f^(k-1) a hair below q, it runs k iterations and the
    # last one trains on all rows but fewer than q (rounding, < 0.001 % at full scale).
    q = int(min(r_max / r_min_target, n_grid / L))
    if q < 2:
        raise ValueError(f"grid of {n_grid} combinations or rows {r_max:,} too small for "
                         f"{k} halving iterations ending with {L} candidates")
    r_min = r_max // q
    f = q ** (1.0 / (k - 1)) * (1 - 1e-12)

    def cascade(n):
        out = [n]
        for _ in range(k - 1):
            out.append(math.ceil(out[-1] / f))
        return out

    # largest number of candidates whose successive ceil(n / f) ends with L finalists
    n_cand = min(n_grid, math.ceil(L * q))
    while n_cand > L and cascade(n_cand)[-1] > L:
        n_cand -= 1
    n_req = 1 + math.floor(math.log(n_cand, f))
    n_pos = 1 + math.floor(math.log(r_max // r_min, f))
    n_it = min(n_req, n_pos)
    rows = [min(int(f ** i * r_min), r_max) for i in range(n_it)]
    cands = cascade(n_cand)[:n_it]
    if n_it != k or cands[-1] != L or r_max - rows[-1] > q:
        raise RuntimeError(f"halving schedule could not be built: {n_it} iterations, rows {rows}, "
                           f"candidates {cands}")
    return {"n_candidates": n_cand, "factor": f, "min_resources": r_min, "max_resources": r_max,
            "iterations": n_it, "rows": rows, "fit_rows": [round(r * frac_fit) for r in rows],
            "candidates": cands}


def search_jobs(cfg: "TrainingConfig", n_rows: int, n_features: int) -> tuple[int, int]:
    """(processes, threads per model) of the parallel search, from the memory.

    A search process holds, at the last iteration, a copy of the rows it trains
    and scores on plus SMOTE's output and the model's own copy (XGBoost's
    DMatrix, the MLP's batches): about 2.5 times the float32 rows of fit +
    valid. The processes are capped so that together they use at most
    `memory_safety` of the available memory (`available_memory`), and the
    remaining CPUs go to the models' threads. Neither number changes a result.
    """
    per_process = 2.5 * n_rows * n_features * 4
    budget = cfg.memory_safety * available_memory()
    jobs = int(max(1, min(cfg.n_jobs, budget // per_process)))
    return jobs, max(1, cfg.n_jobs // jobs)


def _grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def _halving_search(name, cfg, X_fv, y_fv, n_fit, prior, ratio):
    """HalvingRandomSearchCV on the predefined split (fit rows train, valid rows score),
    following `halving_schedule`; the schedule actually run is checked against it."""
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV, PredefinedSplit
    grid = cfg.grid_for(name)
    fold = np.r_[np.full(n_fit, -1), np.zeros(len(y_fv) - n_fit, dtype=int)]
    frac_fit = n_fit / len(y_fv)
    jobs, threads = search_jobs(cfg, len(y_fv), X_fv.shape[1])
    if _grid_size(grid) <= cfg.last_candidates:
        # nothing to eliminate: every combination is compared on all of fit
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(make_estimator(name, cfg, ratio, n_jobs=threads), _prefixed(name, grid),
                              cv=PredefinedSplit(fold), scoring=_corrected_log_loss_scorer(prior),
                              refit=False, n_jobs=jobs, error_score="raise")
        search.fit(X_fv, y_fv)
        res = pd.DataFrame(search.cv_results_)
        table = res[["params", "mean_test_score"]].rename(columns={"mean_test_score": "-log-loss (valid)"})
        best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        return best, float(-search.best_score_), table
    sched = halving_schedule(_grid_size(grid), len(y_fv), frac_fit, cfg)
    print(f"{name}: {sched['iterations']} iterations, candidates {sched['candidates']}, "
          f"fit rows {sched['fit_rows']}, factor {sched['factor']:.3f}; "
          f"{jobs} search processes x {threads} threads")
    search = HalvingRandomSearchCV(
        make_estimator(name, cfg, ratio, n_jobs=threads), _prefixed(name, grid),
        n_candidates=sched["n_candidates"], factor=sched["factor"], resource="n_samples",
        min_resources=sched["min_resources"], max_resources=sched["max_resources"],
        cv=PredefinedSplit(fold), scoring=_corrected_log_loss_scorer(prior), refit=False,
        random_state=cfg.random_state, n_jobs=jobs, error_score="raise")
    search.fit(X_fv, y_fv)
    if list(search.n_resources_) != sched["rows"] or list(search.n_candidates_) != sched["candidates"]:
        raise RuntimeError(f"{name}: schedule run {search.n_resources_} / {search.n_candidates_} "
                           f"differs from the planned {sched['rows']} / {sched['candidates']}")
    res = pd.DataFrame(search.cv_results_)
    table = res[["iter", "n_resources", "params", "mean_test_score"]].copy()
    table["n_resources"] = (table["n_resources"] * frac_fit).round().astype(int)
    table = table.rename(columns={"n_resources": "fit rows", "mean_test_score": "-log-loss (valid)"})
    table.attrs.update(schedule=sched, search_jobs=jobs, model_threads=threads)
    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    return best, float(-search.best_score_), table


def _adaboost_staged(cfg, X_fit, y_fit, X_valid, y_valid, prior, ratio):
    """One AdaBoost fit per learning rate at the maximal number of estimators;
    the number of iterations minimising the corrected log-loss on valid is read
    from staged_predict_proba (audit 1.4)."""
    rows, best = [], (np.inf, None)
    for lr in cfg.adaboost_learning_rates:
        model = make_estimator("AdaBoost", cfg, ratio, {"learning_rate": lr,
                                                         "n_estimators": cfg.adaboost_max_estimators})
        model.fit(X_fit, y_fit)
        staged = model.named_steps[PIPE_STEP].staged_predict_proba(X_valid)
        for it, p in enumerate(staged, start=1):
            ll = corrected_log_loss(y_valid, p[:, 1], prior)
            rows.append({"learning_rate": lr, "n_estimators": it, "log-loss (valid)": ll})
            if ll < best[0]:
                best = (ll, {"learning_rate": lr, "n_estimators": it})
    return best[1], float(best[0]), pd.DataFrame(rows)


def _tune_adaboost_from_files(xa_path: str, y_path: str, n_fit: int, n_fv: int, cfg: TrainingConfig):
    """AdaBoost tuning in its own process: matrices are memory-mapped, not copied."""
    XA = np.load(xa_path, mmap_mode="r")
    y = np.load(y_path).astype(int)
    prior = float(y[:n_fit].mean())
    t0 = time.time()
    best, crit, table = _adaboost_staged(cfg, XA[:n_fit], y[:n_fit], XA[n_fit:n_fv], y[n_fit:n_fv],
                                         prior, prior / (1 - prior))
    return best, crit, table, time.time() - t0


def tune_binary_classifiers(X_fv: np.ndarray, y_fv: np.ndarray, n_fit: int,
                            config: Optional[TrainingConfig] = None,
                            adaboost_files: Optional[tuple] = None) -> dict:
    """Hyperparameters of every family, chosen on valid (audit 1.3).

    Args:
        adaboost_files: (xa_path, y_path) of the .npy matrices; when given,
            AdaBoost (sequential, the longest) is tuned in a separate process
            in parallel with the other families (audit 1.7).

    Returns:
        {"best": {family: params}, "criterion": {family: log-loss on valid},
         "tables": {family: search table}, "time": {family: seconds}, "prior": base rate of fit}
    """
    cfg = config or TrainingConfig()
    log = print if cfg.verbose else (lambda *a, **k: None)
    y_fv = np.asarray(y_fv).astype(int)
    X_fit, y_fit, X_valid, y_valid = X_fv[:n_fit], y_fv[:n_fit], X_fv[n_fit:], y_fv[n_fit:]
    prior = float(y_fit.mean())
    ratio = prior / (1 - prior)
    log(f"fit {n_fit:,} rows, valid {len(y_valid):,} rows, positive share {prior:.4f} "
        f"(minority/majority {ratio:.4f}; SMOTE {'on' if ratio < cfg.smote_sampling_strategy else 'off'})")
    out = {"best": {}, "criterion": {}, "tables": {}, "time": {}, "prior": prior}
    future = None
    if adaboost_files is not None and "AdaBoost" in cfg.models and cfg.search != "none":
        import concurrent.futures
        import multiprocessing
        pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        future = pool.submit(_tune_adaboost_from_files, *adaboost_files, n_fit, len(y_fv), cfg)
        log("AdaBoost: tuned in a separate process")
    for name in cfg.models:
        if name == "AdaBoost" and future is not None:
            continue
        t0 = time.time()
        if cfg.search == "none":
            best, crit, table = {}, float("nan"), pd.DataFrame()
        elif name == "AdaBoost":
            best, crit, table = _adaboost_staged(cfg, X_fit, y_fit, X_valid, y_valid, prior, ratio)
        else:
            best, crit, table = _halving_search(name, cfg, X_fv, y_fv, n_fit, prior, ratio)
        out["best"][name], out["criterion"][name], out["tables"][name] = best, crit, table
        out["time"][name] = time.time() - t0
        log(f"{name}: best {best}, corrected log-loss on valid {crit:.5f} ({out['time'][name]:.0f} s)")
    if future is not None:
        best, crit, table, dt = future.result()
        pool.shutdown()
        out["best"]["AdaBoost"], out["criterion"]["AdaBoost"] = best, crit
        out["tables"]["AdaBoost"], out["time"]["AdaBoost"] = table, dt
        log(f"AdaBoost: best {best}, corrected log-loss on valid {crit:.5f} ({dt:.0f} s)")
    return out


def fit_binary_classifiers(X_fv: np.ndarray, y_fv: np.ndarray, tuned: dict,
                           config: Optional[TrainingConfig] = None) -> ModelSet:
    """Refit every family on fit + valid with the parameters chosen on valid."""
    cfg = config or TrainingConfig()
    log = print if cfg.verbose else (lambda *a, **k: None)
    y_fv = np.asarray(y_fv).astype(int)
    prior = tuned["prior"]
    models, rows = ModelSet(), []
    for name in cfg.models:
        t0 = time.time()
        model = make_estimator(name, cfg, prior / (1 - prior), tuned["best"][name])
        model.fit(X_fv, y_fv)
        t_fit = time.time() - t0
        log(f"{name}: refitted on fit + valid ({len(y_fv):,} rows) in {t_fit:.0f} s")
        models[name] = model
        rows.append({"Model": name, "Best params": str(tuned["best"][name]),
                     "Log-loss valid (corrected)": tuned["criterion"][name],
                     "Search time (s)": round(tuned["time"][name], 1), "Refit time (s)": round(t_fit, 1),
                     "Refit rows": len(y_fv)})
    models.search_report = pd.DataFrame(rows).set_index("Model")
    models.search_tables = tuned["tables"]
    models.best_params = dict(tuned["best"])
    models.prior, models.config = prior, cfg
    return models


def train_binary_classifiers(X_fv: np.ndarray, y_fv: np.ndarray, n_fit: int,
                             config: Optional[TrainingConfig] = None) -> ModelSet:
    """Tune every family on fit / valid, then refit it on fit + valid.

    Args:
        X_fv: features of fit followed by valid (contiguous rows, so that
            X_fv[:n_fit] and X_fv[n_fit:] are views, not copies).
        y_fv: binary target (1 = class of interest), same order.
        n_fit: number of fit rows at the top of X_fv.
    """
    tuned = tune_binary_classifiers(X_fv, y_fv, n_fit, config)
    return fit_binary_classifiers(X_fv, y_fv, tuned, config)


#: History of evaluated runs, see `record_training_run`.
TRAINING_RUNS: list[dict] = []


def record_training_run(label: str, results_df: pd.DataFrame, notes: str = "") -> pd.DataFrame:
    """Archive the test metrics of one run, to compare successive settings."""
    for _, row in results_df.iterrows():
        TRAINING_RUNS.append({"Run": label, **row.to_dict(), "Notes": notes})
    return training_history_table()


def training_history_table(sort_by: str = BRIER_COL) -> pd.DataFrame:
    if not TRAINING_RUNS:
        return pd.DataFrame()
    hist = pd.DataFrame(TRAINING_RUNS)
    if sort_by in hist.columns:
        hist = hist.sort_values(sort_by)
    return hist.reset_index(drop=True)


def clear_training_history() -> None:
    TRAINING_RUNS.clear()


# ----------------------------------------------------------------------------
# 4.1 Calibration, model choice and thresholds on the calib partition
# ----------------------------------------------------------------------------
class PriorCorrected(ClassifierMixin, BaseEstimator):
    """A fitted model whose probabilities are mapped back to the true prior."""

    def __init__(self, model=None, prior: float = 0.5):
        self.model = model
        self.prior = prior

    def fit(self, X, y):  # the wrapped model is already fitted
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = prior_correct(self.model.predict_proba(X)[:, 1], self.prior)
        return np.c_[1 - p, p]


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising Youden's J = TPR - FPR (ROC-optimal point)."""
    fpr, tpr, thr = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
    return float(min(thr[k], 1.0))


def share_matching_threshold(p: np.ndarray, target_share: float) -> float:
    """Threshold t such that the share of p >= t equals target_share (rec. 12:
    the predicted share of group A then matches the observed one)."""
    if not 0 < target_share < 1:
        raise ValueError(f"target share {target_share} outside (0, 1)")
    return float(np.quantile(np.asarray(p, dtype=float), 1.0 - target_share))


@dataclass
class CalibrationResult:
    table: pd.DataFrame          # Brier per (model, method) on calib half B
    models: dict                 # {family: calibrated model (best method)}
    methods: dict                # {family: method kept}
    best_name: str               # family with the lowest calibrated Brier
    thresholds: dict             # {family: youden threshold on calib half B}
    p_calib_eval: dict           # {family: calibrated probabilities on calib half B}
    y_calib_eval: np.ndarray


def calibrate_and_select(models: ModelSet, X_calib: np.ndarray, y_calib: np.ndarray,
                         methods: Sequence[str] = ("sigmoid", "isotonic"),
                         random_state: int = RANDOM_STATE) -> CalibrationResult:
    """Calibrate every family on calib, keep its best method, choose the model.

    calib is split in two stratified halves: the calibration maps (Platt
    scaling, isotonic regression, or the analytic prior correction alone) are
    fitted on half A; on half B the calibrated Brier score chooses the method
    of each family and then the model (audit 1.6), and Youden's threshold is
    read. The test partition is not touched.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    y_calib = np.asarray(y_calib).astype(int)
    idx_a, idx_b = train_test_split(np.arange(len(y_calib)), test_size=0.5,
                                    random_state=random_state, stratify=y_calib)
    Xa, ya, Xb, yb = X_calib[idx_a], y_calib[idx_a], X_calib[idx_b], y_calib[idx_b]
    rows, chosen, kept, probs, thresholds = [], {}, {}, {}, {}
    for name, model in models.items():
        cands = {"prior correction": PriorCorrected(model, models.prior).fit(Xa, ya)}
        for m in methods:
            cands[m] = CalibratedClassifierCV(FrozenEstimator(model), method=m).fit(Xa, ya)
        best = (np.inf, None)
        for m, cm in cands.items():
            p = cm.predict_proba(Xb)[:, 1]
            b = float(brier_score_loss(yb, p))
            rows.append({"Model": name, "Method": m, BRIER_COL: b,
                         LOGLOSS_COL: float(log_loss(yb, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]))})
            if b < best[0]:
                best = (b, m)
                probs[name] = p
        chosen[name], kept[name] = cands[best[1]], best[1]
        thresholds[name] = youden_threshold(yb, probs[name])
    table = pd.DataFrame(rows)
    best_rows = table.loc[table.groupby("Model")[BRIER_COL].idxmin()]
    best_name = str(best_rows.sort_values(BRIER_COL).iloc[0]["Model"])
    return CalibrationResult(table=table, models=chosen, methods=kept, best_name=best_name,
                             thresholds=thresholds, p_calib_eval=probs, y_calib_eval=yb)


# ----------------------------------------------------------------------------
# 4.2 Metrics (audit P5)
# ----------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float, base_rate: float) -> dict:
    """Probabilistic and thresholded metrics of one model on one partition.

    Probabilistic: AUC-ROC, AUC-PR, log-loss, Brier, Brier skill score against
    the constant forecast `base_rate` (training prevalence). Thresholded (at a
    threshold chosen on calib): accuracy, and precision / recall / F1 both
    macro-averaged and for the positive class alone; the accuracy of the
    majority-class predictor is given for reference.
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    yp = (p >= threshold).astype(int)
    brier = float(brier_score_loss(y, p))
    brier_ref = float(np.mean((base_rate - y) ** 2))
    return {AUC_COL: float(roc_auc_score(y, p)), AP_COL: float(average_precision_score(y, p)),
            LOGLOSS_COL: float(log_loss(y, p, labels=[0, 1])), BRIER_COL: brier,
            BSS_COL: 1.0 - brier / brier_ref, ACC_COL: float(accuracy_score(y, yp)),
            BASE_ACC_COL: float(max(y.mean(), 1 - y.mean())),
            PREC_COL: float(precision_score(y, yp, average="macro", zero_division=0)),
            REC_COL: float(recall_score(y, yp, average="macro", zero_division=0)),
            F1_COL: float(f1_score(y, yp, average="macro", zero_division=0)),
            PREC_POS_COL: float(precision_score(y, yp, zero_division=0)),
            REC_POS_COL: float(recall_score(y, yp, zero_division=0)),
            F1_POS_COL: float(f1_score(y, yp, zero_division=0)), THR_COL: float(threshold)}


def compute_classification_metrics(calibration: CalibrationResult, X_test: np.ndarray, y_test: np.ndarray,
                                   base_rate: float, threshold_metric: str = "youden",
                                   fixed_threshold: float = 0.5,
                                   cache_dir: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
    """Test metrics of every calibrated model at its threshold chosen on calib.

    Test probabilities are cached per model in `cache_dir` (audit 1.7).
    """
    y_test = np.asarray(y_test).astype(int)
    rows, cache = [], {}
    for name, model in calibration.models.items():
        path = os.path.join(cache_dir, f"test_proba_{name}.npy") if cache_dir else None
        if path and os.path.exists(path):
            p = np.load(path)
        else:
            p = model.predict_proba(X_test)[:, 1]
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(path, p)
        thr = calibration.thresholds[name] if threshold_metric == "youden" else fixed_threshold
        met = classification_metrics(y_test, p, thr, base_rate)
        cm = confusion_matrix(y_test, (p >= thr).astype(int), labels=[0, 1])
        cache[name] = dict(p_pos=p, cm_opt=cm, bt=round(thr, 3), f1_t=met[F1_COL], auc=met[AUC_COL])
        rows.append({"Model": name, "Calibration": calibration.methods[name], **met})
    return pd.DataFrame(rows), cache


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
        inner = model
        # calibrated wrappers -> fitted pipeline -> classifier step -> fitted estimator
        if isinstance(inner, PriorCorrected):
            inner = inner.model
        if hasattr(inner, "calibrated_classifiers_"):
            inner = inner.calibrated_classifiers_[0].estimator
        if hasattr(inner, "estimator") and not isinstance(inner, (ImbPipeline, BalancedFit)):
            inner = inner.estimator           # FrozenEstimator
        inner = inner.named_steps[PIPE_STEP] if isinstance(inner, ImbPipeline) else inner
        if hasattr(inner, "estimator_") and hasattr(inner.estimator_, "feature_importances_"):
            inner = inner.estimator_
        if hasattr(inner, "feature_importances_"):
            importances[name] = ("Impurity / gain importance", pd.Series(inner.feature_importances_, index=feature_cols))
        elif X_eval is not None and y_eval is not None:
            n = min(perm_sample, len(X_eval))
            pos = np.random.default_rng(random_state).choice(len(X_eval), size=n, replace=False)
            result = permutation_importance(
                model, X_eval[pos], np.asarray(y_eval)[pos],
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
    models_dict: dict, X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    model_names: list[str], palette: np.ndarray, prefix: str = "",
) -> pd.DataFrame:
    """AUC-ROC and Brier score on a sample of the training rows and on test,
    to detect overfitting or leakage (a large gap, or near-perfect scores of
    very different families, respectively). Threshold-free."""
    rows = []
    for name in model_names:
        model = models_dict[name]
        p_tr = model.predict_proba(X_train)[:, 1]
        p_te = model.predict_proba(X_test)[:, 1]
        rows.append({"Model": name,
                     f"{AUC_COL} (train)": roc_auc_score(y_train, p_tr), f"{AUC_COL} (test)": roc_auc_score(y_test, p_te),
                     f"{BRIER_COL} (train)": brier_score_loss(y_train, p_tr), f"{BRIER_COL} (test)": brier_score_loss(y_test, p_te)})
    comparison_df = pd.DataFrame(rows)
    comparison_df[f"{AUC_COL} gap (train-test)"] = comparison_df[f"{AUC_COL} (train)"] - comparison_df[f"{AUC_COL} (test)"]
    display(comparison_df.set_index("Model").round(4))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(model_names))
    width = 0.35
    for ax, metric in zip(axes, [AUC_COL, BRIER_COL]):
        ax.bar(x - width / 2, comparison_df[f"{metric} (train)"], width, label="Train", color=PLOT_COLORS[0], alpha=0.85)
        ax.bar(x + width / 2, comparison_df[f"{metric} (test)"], width, label="Test", color=PLOT_COLORS[1], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        sns.despine(ax=ax)
    _savefig("train_test_comparison", prefix)
    plt.show()
    return comparison_df


ALL_EVAL_PLOTS: tuple[str, ...] = ("metrics", "train_test", "confusion", "roc", "pr", "importances")


def evaluate_binary_classification_models(
    calibration: "CalibrationResult", X_test: np.ndarray, y_te: np.ndarray, base_rate: float,
    positive_name: str, reference_name: str, feature_cols: list[str],
    X_train: Optional[np.ndarray] = None, y_tr: Optional[np.ndarray] = None,
    prefix: str = "", threshold_metric: str = "youden", fixed_threshold: float = 0.5,
    plots: Sequence[str] = ALL_EVAL_PLOTS, perm_sample: int = 20_000,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Test metrics (audit P5) and the standard figures of the calibrated models.

    Thresholds come from the calib partition (`calibrate_and_select`), never
    from test. X_train / y_tr: a sample of training rows for the train / test
    comparison.
    """
    plots = tuple(plots)
    unknown = [p for p in plots if p not in ALL_EVAL_PLOTS]
    if unknown:
        raise ValueError(f"Unknown figures {unknown}, expected among {list(ALL_EVAL_PLOTS)}")
    results_df, cache = compute_classification_metrics(
        calibration, X_test, y_te, base_rate, threshold_metric, fixed_threshold, cache_dir)
    model_names = list(calibration.models)
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(model_names)))
    print(f"{positive_name} vs {reference_name}: test metrics, thresholds ({threshold_metric}) chosen on calib")
    display(results_df.set_index("Model").round(4))
    if "metrics" in plots:
        plot_metrics_comparison(results_df, model_names, palette, prefix)
    if "train_test" in plots and X_train is not None and y_tr is not None:
        plot_train_test_comparison(calibration.models, X_train, y_tr, X_test, y_te, model_names, palette, prefix)
    if "confusion" in plots:
        plot_confusion_matrices(cache, model_names, [reference_name, positive_name], palette, prefix)
    if "roc" in plots:
        plot_roc_curves(cache, y_te, model_names, palette, prefix)
    if "pr" in plots:
        plot_precision_recall(cache, y_te, model_names, palette, positive_name, prefix)
    if "importances" in plots:
        plot_feature_importances(calibration.models, feature_cols, model_names, palette,
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


# ============================================================================
# 6. Service-time groups: exact 1-D K-medians (or K-means for the ablation)
# ============================================================================

def _segment_costs(x: np.ndarray, W: np.ndarray, S: np.ndarray, a: np.ndarray, b: int):
    """Weighted L1 cost and weighted median of the segments x[a..b] (a vector, b scalar)."""
    W0 = np.where(a > 0, W[a - 1], 0.0)
    S0 = np.where(a > 0, S[a - 1], 0.0)
    half = W0 + (W[b] - W0) / 2.0
    m = np.minimum(np.searchsorted(W, half, side="left"), b)
    m = np.maximum(m, a)
    xm = x[m]
    cost = xm * (W[m] - W0) - (S[m] - S0) + (S[b] - S[m]) - xm * (W[b] - W[m])
    return cost, xm


def kmedians_1d_exact(values: np.ndarray, k: int) -> dict:
    """Exact K-medians of one-dimensional data (minimum total absolute deviation).

    The optimal clusters of 1-D data are intervals of the sorted values, so a
    dynamic programme over the U distinct values (weighted by their counts)
    finds the global optimum in O(K U^2), with no initialisation and no
    randomness. Service times are integer minutes, so U is small even for tens
    of millions of rows. Centres are the lower weighted medians of the clusters.

    Returns:
        dict with `centers` (ascending), `cutoffs` (midpoints between consecutive
        centres: a value is assigned to the nearest centre), `cost` (sum of
        absolute deviations) and `sizes` (rows per cluster).
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("kmedians_1d_exact: empty input or non-finite values")
    x, w = np.unique(v, return_counts=True)
    U = x.size
    if not 1 <= k <= U:
        raise ValueError(f"k={k} must lie between 1 and the number of distinct values ({U})")
    w = w.astype(float)
    W, S = np.cumsum(w), np.cumsum(w * x)
    D = np.full((k, U), np.inf)
    arg = np.zeros((k, U), dtype=int)
    for b in range(U):
        D[0, b] = _segment_costs(x, W, S, np.array([0]), b)[0][0]
    for j in range(1, k):
        for b in range(j, U):
            a = np.arange(j, b + 1)
            c, _ = _segment_costs(x, W, S, a, b)
            tot = D[j - 1, a - 1] + c
            best = int(np.argmin(tot))
            D[j, b], arg[j, b] = tot[best], a[best]
    bounds, b = [], U - 1
    for j in range(k - 1, -1, -1):
        a = arg[j, b] if j > 0 else 0
        bounds.append((a, b))
        b = a - 1
    bounds.reverse()
    centers = np.array([_segment_costs(x, W, S, np.array([a]), b)[1][0] for a, b in bounds])
    sizes = np.array([W[b] - (W[a - 1] if a > 0 else 0.0) for a, b in bounds])
    return {"centers": centers, "cutoffs": (centers[:-1] + centers[1:]) / 2.0,
            "cost": float(D[k - 1, U - 1]), "sizes": sizes}


@dataclass
class DurationGroups:
    """Service-time groups fitted on the training durations only (audit B3).

    Group "A" is the longest. `labels` applies the training cutoffs to any
    partition, so that test and holdout never influence the groups.
    """
    method: str
    stat: str
    centers: np.ndarray        # ascending
    cutoffs: np.ndarray        # ascending, between consecutive centres
    s_by_group: dict           # duration used by the ILP for each group (statistic `stat`)
    shares: dict               # share of training rows per group
    n_train: int

    @property
    def names(self) -> list:
        return ["ABCDE"[i] for i in range(len(self.centers))][::-1]   # ascending centres -> E..A

    def labels(self, values) -> np.ndarray:
        idx = np.searchsorted(self.cutoffs, np.asarray(values, dtype=float), side="right")
        return np.asarray(self.names)[idx]

    @property
    def cutoff(self) -> float:
        """Boundary between the two longest groups (the A/B cutoff when K = 2)."""
        return float(self.cutoffs[-1])


def fit_duration_groups(values, k: int = 2, method: str = "kmedians", stat: str = "median",
                        random_state: int = RANDOM_STATE) -> DurationGroups:
    """Fit K duration groups on training service times.

    method="kmedians": exact 1-D K-medians (`kmedians_1d_exact`), the method of
    the manuscript; the centres are medians. method="kmeans" (ablation): K-means
    with 10 restarts on the same values.
    """
    v = np.asarray(values, dtype=float)
    if method == "kmedians":
        res = kmedians_1d_exact(v, k)
        centers, cutoffs = res["centers"], res["cutoffs"]
    elif method == "kmeans":
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(v.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
        cutoffs = (centers[:-1] + centers[1:]) / 2.0
    else:
        raise ValueError(f"unknown duration method {method!r}")
    groups = DurationGroups(method=method, stat=stat, centers=centers, cutoffs=cutoffs,
                            s_by_group={}, shares={}, n_train=int(v.size))
    lab = groups.labels(v)
    agg = {"median": np.median, "mean": np.mean}[stat]
    for g in groups.names:
        sel = v[lab == g]
        if sel.size == 0:
            raise ValueError(f"group {g} is empty")
        groups.s_by_group[g] = float(agg(sel))
        groups.shares[g] = float(sel.size / v.size)
    return groups


def evaluate_duration_ks(values, candidate_ks=(2, 3, 4, 5), method: str = "kmedians",
                         sil_sample: int = 20_000, random_state: int = RANDOM_STATE,
                         prefix: str = "") -> pd.DataFrame:
    """Within-cluster cost and silhouette for several K, on training durations.

    The cost is the objective of the method (absolute deviations for K-medians,
    squared deviations for K-means), computed on all values; the silhouette on
    a sample drawn with a local generator (audit rec. 15).
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    sample = rng.choice(v, size=min(sil_sample, v.size), replace=False)
    rows = []
    for k in candidate_ks:
        g = fit_duration_groups(v, k, method, "median" if method == "kmedians" else "mean", random_state)
        lab = g.labels(v)
        centre = dict(zip(g.names, g.centers[::-1]))
        dev = v - np.vectorize(centre.get)(lab)
        cost = float(np.abs(dev).sum()) if method == "kmedians" else float((dev ** 2).sum())
        sil = float(silhouette_score(sample.reshape(-1, 1), g.labels(sample)))
        rows.append({"K": k, "Cost": cost, "Silhouette": sil,
                     "Centres (min)": ", ".join(f"{c:g}" for c in g.centers)})
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(table["K"], table["Cost"], "o-", color=PLOT_COLORS[0])
    axes[0].set_xlabel("Number of groups K")
    axes[0].set_ylabel("Absolute deviation (min)" if method == "kmedians" else "Squared deviation (min²)")
    axes[1].plot(table["K"], table["Silhouette"], "o-", color=PLOT_COLORS[2])
    axes[1].set_xlabel("Number of groups K"); axes[1].set_ylabel("Silhouette coefficient")
    for ax in axes:
        ax.grid(alpha=0.4); sns.despine(ax=ax)
    _savefig("duration_k_evaluation", prefix)
    plt.show()
    return table


def bootstrap_silhouette(values, groups: DurationGroups, n_bootstrap: int = 10,
                         sample_size: int = 10_000, random_state: int = RANDOM_STATE) -> tuple[float, float]:
    """Bootstrap of the silhouette of the fitted groups (stability of K).

    Draws come from a local generator seeded by `random_state` (audit rec. 15:
    the former version used the global NumPy state).
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    sils = []
    for _ in range(n_bootstrap):
        sub = rng.choice(v, size=min(sample_size, v.size), replace=True)
        sils.append(silhouette_score(sub.reshape(-1, 1), groups.labels(sub)))
    return float(np.mean(sils)), float(np.std(sils))


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
    features: np.ndarray, noshow_model, group_model, threshold_A: float,
    s_A: float, s_B: float, index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Single inference entry point: features -> (q_i, p_A, group, s_expected).

    Both models are the CALIBRATED ones of section 5. The group is predicted
    from the features (the service-time tree needs the realised duration,
    unknown before the visit). The threshold on p_A is the one chosen on calib
    so that the predicted share of group A matches the observed share (audit
    rec. 12: at 0.5 the run of 26/09 predicted 42 % of long patients for 15 %).

    Returns:
        DataFrame with q_i = 1 - P(no-show), p_A = P(group A), group (A if
        p_A >= threshold_A) and s_expected = p_A s_A + (1 - p_A) s_B, the
        expected duration under the calibrated group probability.
    """
    p_noshow = noshow_model.predict_proba(features)[:, 1]
    p_A = group_model.predict_proba(features)[:, 1]
    return pd.DataFrame({"q_i": 1.0 - p_noshow, "p_A": p_A,
                         "group": np.where(p_A >= threshold_A, "A", "B"),
                         "s_expected": p_A * s_A + (1.0 - p_A) * s_B}, index=index)


def save_artifacts(
    noshow_model, cluster_tree, group_classifier, scaler: StandardScaler,
    cat_encodings: dict, feature_cols: list[str], name_prefix: str = "final",
    inference: Optional[dict] = None,
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
    if inference is not None:   # threshold on p_A, duration groups, run id
        joblib.dump(inference, f"{MODELS_DIR}/{name_prefix}_inference.joblib")
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
        "inference": joblib.load(f"{MODELS_DIR}/{name_prefix}_inference.joblib"),
    }


def git_commit() -> str:
    """Commit of the code that produced an artifact, "-dirty" if the tree has changes.

    SMART_PLAN_COMMIT overrides it (e.g. a pod running from an exported archive).
    Raises rather than writing an artifact without provenance.
    """
    if os.environ.get("SMART_PLAN_COMMIT"):
        return os.environ["SMART_PLAN_COMMIT"]
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot read the git commit; set SMART_PLAN_COMMIT") from exc
    return head + ("-dirty" if dirty else "")


def export_plne_parameters(paths: Sequence[str], groups: DurationGroups, scores: pd.DataFrame,
                           n_train: int, source_run: str, smoke: bool,
                           slot_lengths: Sequence[float]) -> dict:
    """Write plne_parameters.json (schema 2), the only input of the ILP campaign.

    Args:
        paths: files to write (the run directory, plus the versioned
            data/inference/plne_parameters.json for a production run).
        groups: duration groups fitted on the training durations.
        scores: scored holdout with q_i, group_pred, group_actual, no_show.
        n_train: training rows the groups were fitted on.

    The dictionary is validated with `check_plne_parameters` before writing, so
    a run whose durations leave the model's domain (s_A > longest slot) stops
    here instead of feeding the campaign.
    """
    if len(groups.centers) != 2:
        raise ValueError("the ILP uses two duration groups: K must be 2")
    P = {"schema_version": PLNE_SCHEMA_VERSION, "duration_method": groups.method,
         "duration_stat": groups.stat, "cutoff_minutes": groups.cutoff,
         "s_A": groups.s_by_group["A"], "s_B": groups.s_by_group["B"],
         "pi_A_actual": float((scores["group_actual"] == "A").mean()),
         "pi_A_predicted": float((scores["group_pred"] == "A").mean()),
         "noshow_rate": float(scores["no_show"].mean()),
         "group_error_rate": float((scores["group_pred"] != scores["group_actual"]).mean()),
         "n_train": int(n_train), "n_holdout": int(len(scores)),
         "created": pd.Timestamp.now().isoformat(timespec="seconds"),
         "git_commit": git_commit(), "source_run": source_run, "smoke": bool(smoke)}
    check_plne_parameters(P, slot_lengths)
    for path in paths:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(P, fh, indent=2)
            fh.write("\n")
    return P


def export_holdout_scores(path: str, scores: pd.DataFrame) -> None:
    """Patient-level holdout scores for the ILP campaign (never versioned)."""
    from ilp_campaign import HOLDOUT_COLUMNS, check_holdout_scores
    check_holdout_scores(scores)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    scores[list(HOLDOUT_COLUMNS)].reset_index(drop=True).to_parquet(path, index=False)


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


#: Bounds applied to predicted q_i before they enter an Instance, which needs
#: q in (0, 1): isotonic calibration can return exactly 0 or 1. Same bounds as the
#: campaign generator (ilp_campaign.Q_CLIP).
Q_BOUNDS = (0.01, 0.999)


def build_instance_from_patients(
    df_scored: pd.DataFrame, T: list[int], L: list[list[float]],
    s_A: float, s_B: float, c_I: float, c_W: float, c_O: float,
    duration_mode: str = "class", group_col: str = "group", q_col: str = "q_i",
    positive_group: str = "A",
) -> Instance:
    """Build an ILP Instance from patients scored by `predict_patient`.

    Args:
        df_scored: one patient per row, with `q_col`, `group_col` and, for
            duration_mode="expected", `s_expected`.
        T, L: slots per day and their lengths (minutes).
        s_A, s_B: durations of the two groups (from plne_parameters.json).
        c_I, c_W, c_O: unit costs, all explicit.
        duration_mode: "class" gives each patient the duration of its
            predicted group (the manuscript's model); "expected" uses
            s_expected = p_A s_A + (1 - p_A) s_B (audit rec. 12).

    Returns:
        Instance ready for `solve_ilp`. q is clipped to Q_BOUNDS; the number of
        clipped values is printed.
    """
    if duration_mode == "class":
        s = np.where(df_scored[group_col].to_numpy() == positive_group, s_A, s_B)
    elif duration_mode == "expected":
        s = df_scored["s_expected"].to_numpy(dtype=float)
    else:
        raise ValueError(f"duration_mode={duration_mode!r}, expected 'class' or 'expected'")
    q_raw = df_scored[q_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(q_raw)) or q_raw.min() < 0 or q_raw.max() > 1:
        raise ValueError("predicted q_i must be probabilities")
    q = np.clip(q_raw, *Q_BOUNDS)
    n_clipped = int((q != q_raw).sum())
    if n_clipped:
        print(f"build_instance_from_patients: {n_clipped} q_i clipped to {Q_BOUNDS}")
    return Instance(n=len(df_scored), D=len(T), T=T, L=L, s=s, q=q, c_I=c_I, c_W=c_W, c_O=c_O)
