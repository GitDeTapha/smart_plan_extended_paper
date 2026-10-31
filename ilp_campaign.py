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

from data_driven_plan import Config, Instance, available_cpus, evaluate_planning_exact, solve_ilp

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
#: less than 1e-3 at every level used (checked by `check_beta_mean`). Calibrated
#: holdout q_i are stored as predicted, in [0, 1] (isotonic calibration is a step
#: function whose end blocks can be exactly 0 or 1), and clipped here only when
#: they enter an Instance, as `build_instance_from_patients` does (Q_BOUNDS).
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
    """n attendance probabilities resampled from calibrated holdout predictions, clipped to Q_CLIP."""
    q_pool = np.asarray(q_pool, dtype=float)
    if q_pool.size == 0 or not np.all((q_pool >= 0) & (q_pool <= 1)):
        raise ValueError("q_pool must be a non-empty array of probabilities in [0, 1]")
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
    if not ((df["q_i"] >= 0) & (df["q_i"] <= 1)).all():
        raise ValueError("holdout scores: q_i outside [0, 1]")
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
    """Processes of the campaign: one per physical core actually available (rec. 6).

    In a container the allotted CPUs are hardware threads; their number divided by
    the threads per core of the host (psutil, logical / physical) gives the cores
    available. One single-threaded HiGHS per core keeps two solves off the same core,
    so the reported solve times are not inflated by contention between siblings.
    """
    logical, physical = psutil.cpu_count(logical=True), psutil.cpu_count(logical=False)
    if not logical or not physical:
        raise RuntimeError("number of cores unknown: set n_workers explicitly")
    threads_per_core = max(1, logical // physical)
    return max(1, available_cpus() // threads_per_core)


@dataclass(frozen=True)
class CampaignConfig:
    """Sizes, budgets and grids of the ILP campaign; every field is logged.

    `production()`: one clinic day of S = 8 slots, R = 30 seeds, 600 s per ILP;
    `smoke()` checks that the notebook runs end to end (S = 5, R = 2).
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
        # One clinic day of 8 slots (authors' decision of 29/10): the instance a clinic
        # schedules in practice, solved to proven optimality in seconds on a laptop,
        # whereas 5 days x 8 slots (60 patients) kept gaps of 11 to 12 % after 1,800 s
        # (and 65 % at p_bar = 0.03 after 60 s), the incumbent unchanged between 600 and
        # 1,800 s. The 600 s limit is a safeguard; every status and gap is reported.
        return cls(days=1, slots_per_day=8, R=30, time_limit=600.0, mip_gap=1e-6, M_sim=10_000,
                   ratios=(1.0, 1.25, 1.5, 1.75, 2.0), p_bars=(0.03, 0.10, 0.20, 0.30),
                   long_shares=(0.15, 0.25, 0.50), cw_ratios=(0.1, 0.25, 0.5, 1.0, 2.0),
                   holdout_days=100, n_workers=n_workers or physical_cores())

    @classmethod
    def smoke(cls, n_workers: Optional[int] = None) -> "CampaignConfig":
        return cls(days=1, slots_per_day=5, R=2, time_limit=10.0, mip_gap=1e-4, M_sim=2_000,
                   ratios=(1.0, 1.5, 2.0), p_bars=(0.03, 0.20), long_shares=(0.15, 0.50),
                   cw_ratios=(0.5, 1.0),
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
    """Solve the ILP (warm-started from B3) and keep what the campaign reports.

    peak_rss_mb is the peak resident memory of the solving process so far (an
    upper bound of this solve's peak): the evidence that the optimisation runs
    on modest hardware.
    """
    import resource
    start = greedy_b3(inst) if warm_start else None
    sol = solve_ilp(inst, cfg, warm_start=start)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"planning": sol.planning, "status": sol.status, "gap": sol.gap,
            "gap_complete": sol.gap_complete, "best_bound": sol.best_bound,
            "Z_complete": sol.Z_complete, "E_idle": sol.E_idle, "E_wait": sol.E_wait,
            "E_OT": sol.E_OT, "n_served": sol.n_served, "n_double": sol.n_double,
            "solve_time": sol.solve_time, "highs_run_time": sol.options.get("highs_run_time"),
            "warm_start_cost": model_cost(inst, start) if start is not None else float("nan"),
            "peak_rss_mb": peak_mb}


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


def solve_spec(ctx: CampaignContext, spec: dict, allow_infeasible: bool = False) -> dict:
    """Worker entry point: the cached ILP record of a cell, with the cell's factors.

    allow_infeasible: a cell proven infeasible by HiGHS (a hard cap Q_bar may
    leave no way to serve every patient) is returned with status "infeasible"
    and no cost, instead of stopping the campaign. Any other failure raises.
    """
    from data_driven_plan import NoFeasibleSolution
    try:
        rec = ctx.solve(spec)
    except NoFeasibleSolution as exc:
        if not (allow_infeasible and exc.termination == "infeasible"):
            raise
        rec = {"status": "infeasible", "Z_complete": float("nan"), "n_served": 0, "n_double": 0}
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


# ----------------------------------------------------------------------------
# Benchmark policies (rec. 8). Every schedule is evaluated with the same exact
# cost as the ILP (`model_cost`), whatever objective produced it.
# ----------------------------------------------------------------------------
def _x_model(inst: Instance, capacity: int, serve_all: bool = False):
    import pyomo.environ as pyo
    slots = [(d, t) for d in range(inst.D) for t in range(inst.T[d])]
    idx = [(i, d, t) for i in range(inst.n) for (d, t) in slots]
    m = pyo.ConcreteModel()
    m.x = pyo.Var(idx, domain=pyo.Binary, initialize=0)
    m.C1 = pyo.ConstraintList()
    for i in range(inst.n):
        assigned = sum(m.x[i, d, t] for (d, t) in slots)
        m.C1.add(assigned == 1 if serve_all else assigned <= 1)
    m.C2 = pyo.ConstraintList()
    for (d, t) in slots:
        m.C2.add(sum(m.x[i, d, t] for i in range(inst.n)) <= capacity)
    return m, slots, idx


def _solve_x_model(m, inst, slots, cfg) -> dict:
    from data_driven_plan import solve_model
    import pyomo.environ as pyo
    rep = solve_model(m, cfg)
    planning = {}
    for (d, t) in slots:
        pats = [i for i in range(inst.n) if pyo.value(m.x[i, d, t]) > 0.5]
        if pats:
            planning[d, t] = sorted(pats, key=lambda i: inst.s[i])
    return {"planning": planning, "status": rep.status, "gap": rep.gap, "solve_time": rep.solve_time}


def bench_b1_single(inst: Instance, cfg: Config) -> dict:
    """B1, single booking: the ILP's objective with at most one patient per slot.

    One patient per slot cannot serve everyone when n > S: B1 serves at most S
    patients, by definition of single booking, and the others are refused."""
    import pyomo.environ as pyo
    from data_driven_plan import compute_coefficients
    co = compute_coefficients(inst)
    m, slots, idx = _x_model(inst, 1)
    m.obj = pyo.Objective(expr=sum((inst.c_O * co["o"][k] - inst.c_I * co["a"][k]) * m.x[k] for k in idx))
    return _solve_x_model(m, inst, slots, cfg)


def bench_b2_fluid(inst: Instance, cfg: Config) -> dict:
    """B2, fluid (Jensen) approximation: idle of a slot replaced by
    (L - sum q_i s_i x_i)^+, linearised with u >= 0; waiting ignored. Same
    constraints as the ILP, every patient served."""
    import pyomo.environ as pyo
    from data_driven_plan import compute_coefficients
    co = compute_coefficients(inst)
    m, slots, idx = _x_model(inst, 2, serve_all=True)
    m.u = pyo.Var(slots, bounds=(0, None))
    m.Cf = pyo.ConstraintList()
    for (d, t) in slots:
        m.Cf.add(m.u[d, t] >= inst.L[d][t] - sum(inst.q[i] * inst.s[i] * m.x[i, d, t] for i in range(inst.n)))
    m.obj = pyo.Objective(expr=inst.c_I * sum(m.u[dt] for dt in slots)
                          + sum(inst.c_O * co["o"][k] * m.x[k] for k in idx))
    return _solve_x_model(m, inst, slots, cfg)


def bench_b3_greedy(inst: Instance, cfg: Config = None) -> dict:
    return {"planning": greedy_b3(inst), "status": "heuristic", "gap": float("nan"), "solve_time": 0.0}


def bench_b4_valenzuela(inst: Instance, cfg: Config, Q_cap: float = 1.2) -> dict:
    """B4, utility in the spirit of Valenzuela: max sum q_i x - 0.5 * (double slots),
    expected attendance per slot capped at Q_cap, at most two patients per slot.
    The cap can make serving everyone infeasible, so B4 serves as many as it can."""
    import pyomo.environ as pyo
    m, slots, idx = _x_model(inst, 2)
    m.B = pyo.Var(slots, domain=pyo.Binary)
    m.Cp = pyo.ConstraintList()
    for (d, t) in slots:
        m.Cp.add(sum(inst.q[i] * m.x[i, d, t] for i in range(inst.n)) <= Q_cap)
        m.Cp.add(2 * m.B[d, t] >= sum(m.x[i, d, t] for i in range(inst.n)) - 1)
    m.obj = pyo.Objective(expr=-sum(inst.q[i] * m.x[i, d, t] for (i, d, t) in idx)
                          + 0.5 * sum(m.B[dt] for dt in slots))
    return _solve_x_model(m, inst, slots, cfg)


def bench_b5_uniform(inst: Instance, cfg: Config = None) -> dict:
    """B5, uniform overbooking (manuscript section 11.3): patients in their index
    (arrival) order rotate over the slots in chronological order, up to two per
    slot, ignoring q and s."""
    slots = [(d, t) for d in range(inst.D) for t in range(inst.T[d])]
    planning = {}
    for k in range(min(inst.n, 2 * len(slots))):
        planning.setdefault(slots[k % len(slots)], []).append(k)
    planning = {dt: sorted(p, key=lambda i: inst.s[i]) for dt, p in planning.items()}
    return {"planning": planning, "status": "heuristic", "gap": float("nan"), "solve_time": 0.0}


BENCHMARKS = {"B1": ("Single booking", bench_b1_single),
              "B2": ("Fluid (Jensen)", bench_b2_fluid),
              "B3": ("Greedy", bench_b3_greedy),
              "B4": ("Valenzuela utility", bench_b4_valenzuela),
              "B5": ("Uniform overbooking", bench_b5_uniform)}


def bench_spec(ctx: CampaignContext, spec: dict) -> dict:
    """Worker: ILP (from the cache) and the five benchmarks on one cell, all
    evaluated with the exact model cost and, with common random numbers, in the
    Lindley simulator."""
    from data_driven_plan import draw_scenarios, simulate_instance
    inst = ctx.instance(spec)
    cfg = ctx.cc.solver_config()
    cache = SolutionCache(ctx.cache_root)
    plans = {"ILP": ctx.solve(spec)}
    for name, (_, fn) in BENCHMARKS.items():
        plans[name] = cache.get_or_compute(ctx.key(spec, name), lambda fn=fn: fn(inst, cfg))
    rng = np.random.default_rng(spec["seed"] + 104729)
    scen = draw_scenarios(inst, ctx.cc.M_sim, rng)
    row = {**spec}
    for name, rec in plans.items():
        row[f"{name}_Z"] = model_cost(inst, rec["planning"])
        row[f"{name}_Zsim"] = float(simulate_instance(inst, rec["planning"], ctx.cc.M_sim, rng,
                                                      scenarios=scen).mean())
        row[f"{name}_status"] = rec["status"]
        row[f"{name}_double"] = sum(1 for p in rec["planning"].values() if len(p) == 2)
        row[f"{name}_served"] = sum(len(p) for p in rec["planning"].values())
    return row


# ----------------------------------------------------------------------------
# Paired statistics
# ----------------------------------------------------------------------------
def paired_summary(values: Sequence[float], reference: Sequence[float]) -> dict:
    """Paired comparison of a policy with a reference over the same seeds.

    Returns the mean difference (values - reference) with its 95 % Student
    interval, the mean relative difference in %, the number of seeds where the
    policy does strictly better, and the two-sided Wilcoxon signed-rank p-value
    (1.0 when every difference is zero; undefined below two seeds).
    """
    from scipy import stats
    v, r = np.asarray(values, float), np.asarray(reference, float)
    d = v - r
    n = d.size
    mean = float(d.mean())
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    if n < 2:
        p = float("nan")
    elif np.allclose(d, 0.0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    return {"n": n, "mean_diff": mean, "ci_low": mean - half, "ci_high": mean + half,
            "rel_pct": float(np.mean(100.0 * d / r)), "n_better": int((d < -1e-9).sum()),
            "p_wilcoxon": p}


# ----------------------------------------------------------------------------
# Sensitivity (audit P3)
# ----------------------------------------------------------------------------
_PERTURB_CODES = {"q_shift": 1, "q_noise": 2, "duration": 3, "misclass": 4}


def perturbed_instance(inst: Instance, kind: str, level: float, seed: int,
                       s_A: float, s_B: float) -> Instance:
    """Instance seen by the optimiser when its inputs are wrong.

    q_shift: q + level; q_noise: q + N(0, level); duration: s (1 + level);
    misclass: each patient's group (s_A <-> s_B) swapped with probability level.
    The draw depends only on (seed, kind), so every level of a perturbation uses
    the same uniforms or normals (common random numbers across levels).
    """
    if kind not in _PERTURB_CODES:
        raise ValueError(f"unknown perturbation {kind!r}")
    rng = np.random.default_rng([int(seed), _PERTURB_CODES[kind]])
    q, s = inst.q.copy(), inst.s.copy()
    if kind == "q_shift":
        q = np.clip(q + level, *Q_CLIP)
    elif kind == "q_noise":
        q = np.clip(q + level * rng.standard_normal(inst.n), *Q_CLIP)
    elif kind == "duration":
        s = s * (1.0 + level)
    else:
        swap = rng.random(inst.n) < level
        s = np.where(swap, np.where(np.isclose(s, s_A), s_B, s_A), s)
    return dataclasses.replace(inst, q=q, s=s)


def sensitivity_spec(ctx: CampaignContext, spec: dict) -> dict:
    """Worker: regret of optimising on a perturbed instance, measured on the true one.

    The true optimum is the cached ILP of the unperturbed cell (shared with the
    design of experiments). Both schedules are evaluated exactly with the model
    cost of the true instance, and in the simulator with the same scenarios.
    """
    from data_driven_plan import draw_scenarios, simulate_instance
    kind, level = spec["perturb"]
    base = {k: v for k, v in spec.items() if k != "perturb"}
    inst = ctx.instance(base)
    opt = ctx.solve(base)
    hat_inst = perturbed_instance(inst, kind, level, spec["seed"], ctx.s_A, ctx.s_B)
    hat = SolutionCache(ctx.cache_root).get_or_compute(
        ctx.key(base, "ilp_perturbed", perturb=[kind, level]),
        lambda: solve_record(hat_inst, ctx.cc.solver_config()))
    rng = np.random.default_rng(spec["seed"] + 1299709)
    scen = draw_scenarios(inst, ctx.cc.M_sim, rng)
    sim_hat = simulate_instance(inst, hat["planning"], ctx.cc.M_sim, rng, scenarios=scen)
    sim_opt = simulate_instance(inst, opt["planning"], ctx.cc.M_sim, rng, scenarios=scen)
    Z_hat, Z_opt = model_cost(inst, hat["planning"]), model_cost(inst, opt["planning"])
    return {**base, "kind": kind, "level": level, "Z_hat": Z_hat, "Z_opt": Z_opt,
            "deg_pct": 100.0 * (Z_hat / Z_opt - 1.0), "Zsim_hat": float(sim_hat.mean()),
            "Zsim_opt": float(sim_opt.mean()),
            "sim_diff_se": float((sim_hat - sim_opt).std(ddof=1) / np.sqrt(ctx.cc.M_sim)),
            "status_hat": hat["status"], "status_opt": opt["status"]}


def sensitivity_specs(cc: CampaignConfig, group_error_rate: float) -> list:
    """Perturbations of the central cell; misclassification goes up to the
    classifier's measured error rate (audit P3)."""
    rates = sorted({0.05, 0.10, 0.20, round(float(group_error_rate), 4)})
    grid = ([("q_shift", e) for e in (-0.10, -0.05, 0.05, 0.10)] + [("q_noise", 0.05)]
            + [("duration", e) for e in (-0.20, -0.10, 0.10, 0.20)]
            + [("misclass", r) for r in rates])
    return [{**cell_spec(cc, "central", seed), "perturb": [k, lv]} for (k, lv) in grid for seed in cc.seeds]


def sensitivity_table(df) -> "pd.DataFrame":
    """Mean regret per perturbation with 95 % CI and paired tests (exact and simulated)."""
    import pandas as pd
    rows = []
    for (kind, level), g in df.groupby(["kind", "level"], sort=True):
        ex = paired_summary(g["Z_hat"], g["Z_opt"])
        sim = paired_summary(g["Zsim_hat"], g["Zsim_opt"])
        rows.append({"Perturbation": kind, "Level": level, "n": ex["n"],
                     "Exact regret (%)": ex["rel_pct"], "Exact ΔZ": ex["mean_diff"],
                     "95 % CI low": ex["ci_low"], "95 % CI high": ex["ci_high"],
                     "Exact p": ex["p_wilcoxon"], "Simulated ΔZ": sim["mean_diff"],
                     "Simulated p": sim["p_wilcoxon"],
                     "Significant (5 %)": bool(ex["p_wilcoxon"] < 0.05) if ex["n"] > 1 else False})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Out-of-sample validation on holdout patients (rec. 10)
# ----------------------------------------------------------------------------
def holdout_day_specs(cc: CampaignConfig, scores, seed: int) -> list:
    """Clinic days made of holdout patients, sampled without replacement.

    Each day has cc.slots_per_day slots (share of long slots: central cell) and
    round(central_ratio * slots_per_day) patients. The patients' predicted q_i
    and group, their true group and their realised no-show travel in the spec.
    """
    n = round(cc.central_ratio * cc.slots_per_day)
    if n * cc.holdout_days > len(scores):
        raise ValueError(f"holdout too small: {len(scores)} patients for {cc.holdout_days} days of {n}")
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(scores), n * cc.holdout_days, replace=False)
    specs = []
    for k in range(cc.holdout_days):
        day = scores.iloc[rows[k * n:(k + 1) * n]]
        specs.append({"label": "holdout_day", "seed": int(seed) * 100_003 + k, "day": k,
                      "ratio": cc.central_ratio, "long_share": cc.central_long_share,
                      "cW_ratio": cc.central_cw_ratio,
                      "q": [round(float(v), 12) for v in day["q_i"]],
                      "group_pred": day["group_pred"].tolist(),
                      "group_actual": day["group_actual"].tolist(),
                      "no_show": [int(v) for v in day["no_show"]]})
    return specs


def _holdout_instance(ctx: CampaignContext, spec: dict, groups: Sequence[str]) -> Instance:
    cc = ctx.cc
    rng = np.random.default_rng(spec["seed"])
    L_short, L_long = sorted(ctx.slot_lengths)
    n_long = round(spec["long_share"] * cc.slots_per_day)
    L = rng.permutation([L_long] * n_long + [L_short] * (cc.slots_per_day - n_long)).tolist()
    s = np.where(np.asarray(groups) == "A", ctx.s_A, ctx.s_B)
    return Instance(n=len(groups), D=1, T=[cc.slots_per_day], L=[L], s=s,
                    q=np.clip(np.asarray(spec["q"], dtype=float), *Q_CLIP),
                    c_I=1.0, c_W=spec["cW_ratio"], c_O=ctx.c_O)


def holdout_day(ctx: CampaignContext, spec: dict) -> dict:
    """Worker: plan one holdout day on predicted inputs, judge it on the truth.

    Policies: the ILP on predicted groups (the CPS pipeline), the ILP on true
    groups (oracle on durations), single booking B1, greedy B3 and uniform
    overbooking B5, all planned on predicted inputs except the oracle. Each
    schedule is scored on the true instance (true groups, predicted q_i):
    expected cost of the model, exact expected cost of the day (Lindley,
    terminal idle), and realised cost of the day given who actually came.
    """
    from data_driven_plan import eval_day_exact, simulate_day
    pred = _holdout_instance(ctx, spec, spec["group_pred"])
    true = _holdout_instance(ctx, spec, spec["group_actual"])
    cfg = ctx.cc.solver_config()
    cache = SolutionCache(ctx.cache_root)
    plans = {
        "ILP": cache.get_or_compute(ctx.key(spec, "holdout_ilp_pred"), lambda: solve_record(pred, cfg)),
        "ILP_oracle": cache.get_or_compute(ctx.key(spec, "holdout_ilp_true"), lambda: solve_record(true, cfg)),
        "B1": cache.get_or_compute(ctx.key(spec, "holdout_B1"), lambda: bench_b1_single(pred, cfg)),
        "B3": bench_b3_greedy(pred), "B5": bench_b5_uniform(pred)}
    shows = (1 - np.asarray(spec["no_show"]))[None, :].astype(bool)
    durations = true.s[None, :]
    row = {k: spec[k] for k in ("label", "seed", "day")}
    row["n"], row["n_A_pred"], row["n_A_true"] = pred.n, spec["group_pred"].count("A"), spec["group_actual"].count("A")
    row["no_shows"] = int(sum(spec["no_show"]))
    for name, rec in plans.items():
        slot_patients = [rec["planning"].get((0, t), []) for t in range(true.T[0])]
        ei, ew, eo = eval_day_exact(slot_patients, true.q, true.s, true.L[0])
        idle, wait, ot = simulate_day(slot_patients, true.q, true.s, true.L[0], 1, None,
                                      shows=shows, durations=durations)
        row[f"{name}_Zmodel"] = model_cost(true, rec["planning"])
        row[f"{name}_Zday"] = true.c_I * ei + true.c_W * ew + true.c_O * eo
        row[f"{name}_Zreal"] = float(true.c_I * idle[0] + true.c_W * wait[0] + true.c_O * ot[0])
        row[f"{name}_status"] = rec["status"]
        row[f"{name}_served"] = sum(len(p) for p in rec["planning"].values())
    return row
