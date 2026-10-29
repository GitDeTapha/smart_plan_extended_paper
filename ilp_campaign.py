"""Experimental campaign of the overbooking ILP (notebook 01_ilp_validation).

Instance generator and attendance laws. The ILP itself (`Instance`, `Config`,
`solve_ilp`) and the exact evaluators live in data_driven_plan.py, so the two
notebooks solve and evaluate exactly the same model.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import joblib
import numpy as np
import psutil

from data_driven_plan import Config, Instance, evaluate_planning_exact, solve_ilp

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



# ----------------------------------------------------------------------------
# Campaign configuration (rec. 6)
# ----------------------------------------------------------------------------
def physical_cores() -> int:
    """Number of physical cores; raises rather than guessing when it is unknown."""
    n = psutil.cpu_count(logical=False)
    if not n:
        raise RuntimeError("number of physical cores unknown: set n_workers explicitly")
    return int(n)


@dataclass(frozen=True)
class CampaignConfig:
    """Sizes, budgets and grids of the ILP campaign; every field is logged.

    `production()` follows the original plan (S = 40 slots, R = 30 seeds);
    `smoke()` checks that the notebook runs end to end (S = 10, R = 2).
    """
    days: int
    slots_per_day: int
    R: int
    time_limit: float          # seconds of HiGHS per ILP
    mip_gap: float             # relative gap target on the solver objective
    M_sim: int                 # scenarios per simulated schedule
    ratios: tuple              # n / S; refined around the anomaly at 2.0 (rec. 7)
    p_bars: tuple
    long_shares: tuple
    cw_ratios: tuple
    cw_fine: tuple             # c_W / c_I grid of the threshold study (rec. 11)
    holdout_days: int          # out-of-sample days built from holdout patients (rec. 10)
    n_workers: int             # processes; one HiGHS thread each
    seed_base: int = 1000
    central_ratio: float = 1.5
    central_p_bar: float = 0.20
    central_long_share: float = 0.15
    central_cw_ratio: float = 0.50

    @property
    def S(self) -> int:
        return self.days * self.slots_per_day

    @property
    def seeds(self) -> list:
        return [self.seed_base + r for r in range(self.R)]

    def solver_config(self, **over) -> Config:
        return Config(time_limit=self.time_limit, mip_gap=self.mip_gap, threads=1, **over)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def production(cls, n_workers: Optional[int] = None) -> "CampaignConfig":
        return cls(days=5, slots_per_day=8, R=30, time_limit=1800.0, mip_gap=1e-4, M_sim=10_000,
                   ratios=(1.0, 1.25, 1.5, 1.75, 2.0), p_bars=(0.03, 0.10, 0.20, 0.30),
                   long_shares=(0.15, 0.25, 0.50), cw_ratios=(0.1, 0.25, 0.5, 1.0, 2.0),
                   cw_fine=tuple(round(1.0 + 0.05 * k, 2) for k in range(11)),
                   holdout_days=100, n_workers=n_workers or physical_cores())

    @classmethod
    def smoke(cls, n_workers: Optional[int] = None) -> "CampaignConfig":
        return cls(days=2, slots_per_day=5, R=2, time_limit=30.0, mip_gap=1e-4, M_sim=2_000,
                   ratios=(1.0, 1.5, 2.0), p_bars=(0.03, 0.20), long_shares=(0.15, 0.50),
                   cw_ratios=(0.5, 1.0), cw_fine=(1.0, 1.25, 1.5),
                   holdout_days=4, n_workers=n_workers or physical_cores())


# ----------------------------------------------------------------------------
# Greedy policy B3 (also the warm start of the ILP)
# ----------------------------------------------------------------------------
def greedy_b3(inst: Instance) -> dict:
    """B3: patients by decreasing q_i s_i fill the slots one by one, then the
    remaining ones double the slots with the largest residual expected idle."""
    slots = [(d, t) for d in range(inst.D) for t in range(inst.T[d])]
    priority = sorted(range(inst.n), key=lambda i: inst.q[i] * inst.s[i], reverse=True)
    occupancy = {dt: [] for dt in slots}
    for dt, i in zip(slots, priority):
        occupancy[dt].append(i)
    remaining = priority[len(slots):]

    def residual_idle(dt):
        d, t = dt
        return inst.L[d][t] - sum(inst.q[i] * inst.s[i] for i in occupancy[dt])

    for dt, i in zip(sorted(slots, key=residual_idle, reverse=True), remaining):
        occupancy[dt].append(i)
    return {dt: sorted(p, key=lambda i: inst.s[i]) for dt, p in occupancy.items() if p}


def model_cost(inst: Instance, planning: dict) -> float:
    """Expected cost of a schedule under the ILP's own (slot-separable) evaluation."""
    return evaluate_planning_exact(inst, planning, day_dynamics=False)["Z"]


# ----------------------------------------------------------------------------
# Solution cache and parallel execution (rec. 6)
# ----------------------------------------------------------------------------
def spec_key(**parts) -> str:
    """Stable key of a computation: sha1 of its JSON description."""
    def default(o):
        if isinstance(o, np.ndarray):
            return hashlib.sha1(np.ascontiguousarray(o).tobytes()).hexdigest()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return str(o)
    blob = json.dumps(parts, sort_keys=True, default=default)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


class SolutionCache:
    """On-disk cache of solved schedules, one file per key.

    A cell's solution is computed once and reused by the simulation and the
    sensitivity sections; the key covers the instance, the method and the solver
    configuration, so a change of any of them recomputes.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.joblib")

    def get_or_compute(self, key: str, fn: Callable[[], dict]) -> dict:
        path = self._path(key)
        if os.path.exists(path):
            return joblib.load(path)
        value = fn()
        tmp = path + ".tmp"
        joblib.dump(value, tmp)
        os.replace(tmp, path)          # atomic: a killed worker leaves no partial file
        return value


def solve_record(inst: Instance, cfg: Config, warm_start: bool = True) -> dict:
    """Solve the ILP (warm-started from B3) and keep what the campaign reports."""
    start = greedy_b3(inst) if warm_start else None
    sol = solve_ilp(inst, cfg, warm_start=start)
    return {"planning": sol.planning, "status": sol.status, "gap": sol.gap,
            "gap_complete": sol.gap_complete, "best_bound": sol.best_bound,
            "Z_complete": sol.Z_complete, "E_idle": sol.E_idle, "E_wait": sol.E_wait,
            "E_OT": sol.E_OT, "n_served": sol.n_served, "n_double": sol.n_double,
            "solve_time": sol.solve_time, "highs_run_time": sol.options.get("highs_run_time"),
            "warm_start_cost": model_cost(inst, start) if start is not None else float("nan")}


def run_lpt(fn: Callable, tasks: Sequence, cost: Callable, n_workers: int) -> list:
    """Run fn over tasks in n_workers processes, longest (estimated) first.

    Tasks are dispatched one at a time in decreasing order of `cost`, so each
    free worker takes the longest remaining task (LPT list scheduling). BLAS and
    OpenMP pools are limited to one thread per worker; HiGHS gets threads=1 from
    the Config. Results come back in the order of `tasks`.
    """
    order = sorted(range(len(tasks)), key=lambda k: -cost(tasks[k]))
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        out = joblib.Parallel(n_jobs=n_workers, batch_size=1, pre_dispatch="all")(
            joblib.delayed(fn)(tasks[k]) for k in order)
    results = [None] * len(tasks)
    for k, r in zip(order, out):
        results[k] = r
    return results


# ----------------------------------------------------------------------------
# Cells of the design and their cached solutions
# ----------------------------------------------------------------------------
@dataclass
class CampaignContext:
    """Everything a worker needs to rebuild an instance from a cell spec and solve it.

    A spec is a small dict (label, seed, ratio, p_bar, long_share, cW_ratio and
    optional q_source, Q_bar, pi_A); instances are rebuilt from it in the worker,
    so only specs travel between processes.
    """
    cc: CampaignConfig
    s_A: float
    s_B: float
    pi_A: float
    c_O: float
    slot_lengths: tuple
    cache_root: str
    q_pool: Optional[np.ndarray] = None

    def instance(self, spec: dict) -> Instance:
        q_source = spec.get("q_source", "beta")
        if q_source == "holdout" and self.q_pool is None:
            raise ValueError("spec asks for holdout q but the context has no q_pool")
        return generate_instance(
            spec["seed"], days=self.cc.days, slots_per_day=self.cc.slots_per_day,
            slot_lengths=self.slot_lengths, ratio=spec["ratio"], p_bar=spec["p_bar"],
            long_share=spec["long_share"], cW_ratio=spec["cW_ratio"], s_A=self.s_A, s_B=self.s_B,
            pi_A=spec.get("pi_A", self.pi_A), c_O=self.c_O, Q_bar=spec.get("Q_bar"),
            q_pool=self.q_pool if q_source == "holdout" else None)

    def key(self, spec: dict, method: str, **extra) -> str:
        cell = {k: v for k, v in spec.items() if k != "label"}
        return spec_key(cell=cell, method=method, cfg=self.cc.solver_config(),
                        days=self.cc.days, slots_per_day=self.cc.slots_per_day,
                        s=(self.s_A, self.s_B, self.pi_A, self.c_O, self.slot_lengths),
                        q_pool=self.q_pool if spec.get("q_source") == "holdout" else None, **extra)

    def solve(self, spec: dict) -> dict:
        """ILP solution of a cell, computed once (cache) and reused everywhere."""
        return SolutionCache(self.cache_root).get_or_compute(
            self.key(spec, "ilp"), lambda: solve_record(self.instance(spec), self.cc.solver_config()))


def cell_spec(cc: CampaignConfig, label: str, seed: int, **factors) -> dict:
    spec = {"label": label, "seed": int(seed), "ratio": cc.central_ratio, "p_bar": cc.central_p_bar,
            "long_share": cc.central_long_share, "cW_ratio": cc.central_cw_ratio}
    spec.update(factors)
    return spec


def doe_specs(cc: CampaignConfig, with_holdout_q: bool) -> list:
    """One-factor-at-a-time design around the central cell, R seeds per level."""
    specs = []
    for seed in cc.seeds:
        specs.append(cell_spec(cc, "central", seed))
        specs += [cell_spec(cc, f"ratio_{v}", seed, ratio=v) for v in cc.ratios if v != cc.central_ratio]
        specs += [cell_spec(cc, f"pbar_{v}", seed, p_bar=v) for v in cc.p_bars if v != cc.central_p_bar]
        specs += [cell_spec(cc, f"ls_{v}", seed, long_share=v)
                  for v in cc.long_shares if v != cc.central_long_share]
        specs += [cell_spec(cc, f"cW_{v}", seed, cW_ratio=v)
                  for v in cc.cw_ratios if v != cc.central_cw_ratio]
        if with_holdout_q:
            specs.append(cell_spec(cc, "q_holdout", seed, q_source="holdout"))
    return specs


def spec_cost(spec: dict) -> float:
    """Relative solve-time estimate for LPT ordering: grows with the load n / S."""
    return spec["ratio"] * (2.0 if spec.get("Q_bar") is not None else 1.0)


def solve_spec(ctx: CampaignContext, spec: dict) -> dict:
    """Worker entry point: the cached ILP record of a cell, with the cell's factors."""
    rec = ctx.solve(spec)
    out = {k: v for k, v in rec.items() if k != "planning"}
    out.update(spec)
    out["n"] = ctx.instance(spec).n
    return out


def simulate_spec(ctx: CampaignContext, spec: dict, cvs: Sequence[float] = (0.0, 0.3, 0.5)) -> dict:
    """Replays the cached ILP schedule of a cell in the Lindley simulator.

    For each coefficient of variation of the durations (0 = deterministic,
    lognormal otherwise) returns the mean simulated cost and its standard error,
    next to the cost the model predicts for the same schedule.
    """
    from data_driven_plan import draw_scenarios, simulate_instance
    rec = ctx.solve(spec)
    inst = ctx.instance(spec)
    out = {**spec, "Z_model": rec["Z_complete"], "n_double": rec["n_double"],
           "double_rate": rec["n_double"] / ctx.cc.S, "status": rec["status"]}
    for cv in cvs:
        rng = np.random.default_rng(spec["seed"] + 7919)
        scen = draw_scenarios(inst, ctx.cc.M_sim, rng, cv)
        cost = simulate_instance(inst, rec["planning"], ctx.cc.M_sim, rng, cv, scenarios=scen)
        tag = f"cv{cv:g}"
        out[f"Z_sim_{tag}"] = float(cost.mean())
        out[f"se_sim_{tag}"] = float(cost.std(ddof=1) / np.sqrt(cost.size))
    out["gap_pct"] = 100.0 * (out["Z_sim_cv0"] / out["Z_model"] - 1.0)
    return out
