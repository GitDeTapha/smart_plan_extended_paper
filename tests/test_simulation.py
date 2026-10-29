"""Audit P1: exact evaluators and Lindley simulator with terminal idle."""
import numpy as np
import pytest

from conftest import random_instance
from data_driven_plan import (Config, draw_scenarios, eval_day_exact, evaluate_planning_exact,
                              simulate_instance, solve_ilp)

# Schedule [2], [3,4], [6,0], [5,1] of the illustrative instance (audit P1)
P1_PLANNING = {(0, 0): [2], (0, 1): [3, 4], (0, 2): [6, 0], (0, 3): [5, 1]}


def test_p1_reference_values(illustrative_instance):
    inst = illustrative_instance
    slot_model = evaluate_planning_exact(inst, P1_PLANNING, day_dynamics=False)
    day = evaluate_planning_exact(inst, P1_PLANNING, day_dynamics=True)
    assert slot_model["Z"] == pytest.approx(70.82, abs=5e-3)
    assert day["Z"] == pytest.approx(62.76, abs=5e-3)
    # Without terminal idle the day cost was 44.51: the gap is exactly c_I * E[terminal idle]
    assert day["Z"] - 44.51 == pytest.approx(18.25, abs=1e-2)


def test_simulator_matches_exact_day(illustrative_instance):
    inst = illustrative_instance
    cost = simulate_instance(inst, P1_PLANNING, M=400_000, rng=np.random.default_rng(0))
    exact = evaluate_planning_exact(inst, P1_PLANNING)["Z"]
    se = cost.std(ddof=1) / np.sqrt(cost.size)
    assert abs(cost.mean() - exact) < 4 * se, (cost.mean(), exact, se)


def test_common_random_numbers_are_shared(illustrative_instance):
    inst = illustrative_instance
    scen = draw_scenarios(inst, 1000, np.random.default_rng(1))
    a = simulate_instance(inst, P1_PLANNING, 1000, np.random.default_rng(2), scenarios=scen)
    b = simulate_instance(inst, P1_PLANNING, 1000, np.random.default_rng(3), scenarios=scen)
    np.testing.assert_array_equal(a, b)


def test_slot_exact_matches_ilp_coefficients():
    inst = random_instance(4, n=8, days=2, slots_per_day=4, long_share=0.25)
    sol = solve_ilp(inst, Config(require_optimal=True))
    exact = evaluate_planning_exact(inst, sol.planning, day_dynamics=False)
    assert exact["Z"] == pytest.approx(sol.Z_complete, rel=1e-9)


def test_day_enumeration_is_capped():
    q = np.full(30, 0.5)
    s = np.full(30, 10.0)
    with pytest.raises(ValueError):
        eval_day_exact([list(range(30))], q, s, [60.0])
