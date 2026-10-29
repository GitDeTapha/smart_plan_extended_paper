"""Experimental campaign of the overbooking ILP (notebook 01_ilp_validation).

Instance generator and attendance laws. The ILP itself (`Instance`, `Config`,
`solve_ilp`) and the exact evaluators live in data_driven_plan.py, so the two
notebooks solve and evaluate exactly the same model.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from data_driven_plan import Instance

# ----------------------------------------------------------------------------
# Attendance probabilities q_i (audit P2)
# ----------------------------------------------------------------------------
#: alpha + beta of the Beta law of q. 6.111 gives a standard deviation of 0.15 at
#: mean 0.80, the central cell of the former generator, which set the standard
#: deviation to 0.15 at every level and then floored alpha and beta at 0.5. The
#: floor moved the mean: 0.844 instead of 0.90 at p_bar = 0.10, and 0.5 instead
#: of 0.97 at p_bar = 0.03. Fixing the concentration keeps the mean exact.
BETA_CONCENTRATION = 6.111
#: q is kept inside (0, 1), as `Instance` requires; the clip moves the mean by
#: less than 1e-3 at every level used (checked by `check_beta_mean`).
Q_CLIP = (0.01, 0.999)


def beta_parameters(mean_q: float, kappa: float = BETA_CONCENTRATION) -> tuple:
    """(alpha, beta) of the Beta law with mean `mean_q` and concentration `kappa`."""
    if not 0.0 < mean_q < 1.0:
        raise ValueError(f"mean of q must lie in (0, 1), got {mean_q}")
    if kappa <= 0:
        raise ValueError(f"concentration must be positive, got {kappa}")
    return mean_q * kappa, (1.0 - mean_q) * kappa


def draw_q_beta(rng: np.random.Generator, p_bar: float, n: int,
                kappa: float = BETA_CONCENTRATION) -> np.ndarray:
    """n attendance probabilities with mean 1 - p_bar (p_bar: mean no-show rate)."""
    a, b = beta_parameters(1.0 - p_bar, kappa)
    return np.clip(rng.beta(a, b, n), *Q_CLIP)


def draw_q_empirical(rng: np.random.Generator, q_pool: np.ndarray, n: int) -> np.ndarray:
    """n attendance probabilities resampled from calibrated holdout predictions."""
    q_pool = np.asarray(q_pool, dtype=float)
    if q_pool.size == 0 or not np.all((q_pool > 0) & (q_pool < 1)):
        raise ValueError("q_pool must be a non-empty array of probabilities in (0, 1)")
    return np.clip(rng.choice(q_pool, n, replace=True), *Q_CLIP)


def check_beta_mean(p_bar: float, kappa: float = BETA_CONCENTRATION, n: int = 100_000,
                    seed: int = 0, tol: float = 0.01) -> float:
    """Empirical mean of q on n draws; raises if it misses 1 - p_bar by tol or more."""
    q = draw_q_beta(np.random.default_rng(seed), p_bar, n, kappa)
    if abs(q.mean() - (1.0 - p_bar)) >= tol:
        raise AssertionError(f"p_bar={p_bar}: mean of q {q.mean():.4f}, expected {1 - p_bar:.4f}")
    return float(q.mean())


# ----------------------------------------------------------------------------
# Instance generator
# ----------------------------------------------------------------------------
def generate_instance(seed: int, days: int, slots_per_day: int, slot_lengths: Sequence[float],
                      ratio: float, p_bar: float, long_share: float, cW_ratio: float,
                      s_A: float, s_B: float, pi_A: float, c_O: float, c_I: float = 1.0,
                      Q_bar: Optional[float] = None,
                      q_pool: Optional[np.ndarray] = None) -> Instance:
    """Random instance of the campaign.

    Args:
        seed: the seed fixes the slot grid, the groups and q, so that two cells
            differing by one factor share everything else (paired design).
        ratio: n / S, with S = days * slots_per_day.
        p_bar: mean no-show rate (ignored when `q_pool` is given).
        long_share: share of long slots.
        cW_ratio: c_W / c_I.
        pi_A: probability that a patient belongs to the long group A.
        q_pool: calibrated holdout q_i to resample instead of the Beta law (P2 variant).
    """
    rng = np.random.default_rng(seed)
    L_short, L_long = sorted(float(v) for v in slot_lengths)
    S = days * slots_per_day
    n = max(1, round(ratio * S))
    n_long = round(long_share * S)
    grid = rng.permutation(np.array([L_long] * n_long + [L_short] * (S - n_long)))
    L = [row.tolist() for row in grid.reshape(days, slots_per_day)]
    s = np.where(rng.random(n) < pi_A, s_A, s_B)
    q = draw_q_empirical(rng, q_pool, n) if q_pool is not None else draw_q_beta(rng, p_bar, n)
    return Instance(n=n, D=days, T=[slots_per_day] * days, L=L, s=s, q=q,
                    c_I=c_I, c_W=cW_ratio * c_I, c_O=c_O, Q_bar=Q_bar)


# ----------------------------------------------------------------------------
# Holdout predictions (P2 empirical variant, rec. 10 out-of-sample days)
# ----------------------------------------------------------------------------
#: Columns of holdout_scores.parquet, written by the export stage of
#: data_driven_plan.ipynb (patient-level, never versioned).
HOLDOUT_COLUMNS = ("q_i", "group_pred", "group_actual", "no_show")


def check_holdout_scores(df) -> "pd.DataFrame":
    """Validate the scored holdout; raises on a missing column or an aberrant value."""
    missing = [c for c in HOLDOUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"holdout scores: missing columns {missing}")
    if len(df) == 0:
        raise ValueError("holdout scores: empty table")
    if df[list(HOLDOUT_COLUMNS)].isna().any().any():
        raise ValueError("holdout scores: missing values")
    if not ((df["q_i"] > 0) & (df["q_i"] < 1)).all():
        raise ValueError("holdout scores: q_i outside (0, 1)")
    for col in ("group_pred", "group_actual"):
        if not df[col].isin(["A", "B"]).all():
            raise ValueError(f"holdout scores: {col} must be 'A' or 'B'")
    if not df["no_show"].isin([0, 1]).all():
        raise ValueError("holdout scores: no_show must be 0 or 1")
    return df


def load_holdout_scores(path: str) -> "pd.DataFrame":
    import pandas as pd
    return check_holdout_scores(pd.read_parquet(path))
