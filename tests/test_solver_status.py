"""B1: the MIP gap and termination status are read from HiGHS, never assumed."""
import math

import pyomo.environ as pyo
import pytest

from conftest import random_instance
from data_driven_plan import Config, solve_ilp, solve_model


def test_small_instance_is_proven_optimal(illustrative_instance):
    sol = solve_ilp(illustrative_instance, Config(time_limit=60))
    assert sol.status == "optimal"
    assert sol.proven_optimal
    assert math.isfinite(sol.best_bound)
    assert sol.gap <= 1e-6 + 1e-9
    assert sol.gap_complete <= 1e-6
    assert sol.options["threads"] == 1
    # our gap agrees with the one HiGHS reports
    assert abs(sol.options["highs_mip_gap"]) <= 1e-6 + 1e-9


def test_time_limited_instance_reports_a_positive_gap():
    """The case the old code reported as gap = 0: a solve stopped by the time limit.

    The gap that matters for the article is relative to the complete expected cost;
    the gap on the reduced objective (cost minus the constant c_I * sum L) is
    ill-conditioned because that objective can be close to zero.
    """
    inst = random_instance(seed=1000, n=40, days=4, slots_per_day=5)
    sol = solve_ilp(inst, Config(time_limit=2))
    assert sol.status == "maxTimeLimit"
    assert not sol.proven_optimal
    assert 0 < sol.gap_complete < 1
    assert sol.best_bound <= sol.Z_reduced + 1e-9      # a valid lower bound
    assert sol.options["highs_internals"]
    assert sol.options["highs_run_time"] <= sol.solve_time


def test_require_optimal_refuses_an_unproven_incumbent():
    inst = random_instance(seed=1000, n=40, days=4, slots_per_day=5)
    with pytest.raises(RuntimeError, match="Optimality not proven"):
        solve_ilp(inst, Config(time_limit=2, require_optimal=True))


def test_infeasible_model_raises_instead_of_returning_a_value():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.Binary)
    m.c = pyo.Constraint(expr=m.x >= 2)
    m.o = pyo.Objective(expr=m.x)
    with pytest.raises(RuntimeError, match="no feasible solution"):
        solve_model(m, Config(time_limit=5))
