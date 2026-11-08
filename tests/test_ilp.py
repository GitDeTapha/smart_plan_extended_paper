"""Checks of utils_ilp.py: the ILP finds the true optimum, its closed forms are exact."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils_ilp import (build_milp, greedy, pairs_last, planning_cost, simulate_day, single_booking,  # noqa: E402
                       solve_by_enumeration, solve_ilp, solve_pairs)


def random_instance(rng, n, T):
    q = rng.uniform(0.5, 0.99, n)
    s = rng.choice([15.0, 30.0, 45.0], n)
    L = list(rng.choice([30.0, 45.0], T))
    return q, s, L


@pytest.mark.parametrize("seed", range(50))
def test_ilp_equals_enumeration(seed):
    """On 50 small instances, the ILP and the list of every planning give the same optimum."""
    rng = np.random.default_rng(seed)
    n, T = int(rng.integers(3, 7)), 4
    q, s, L = random_instance(rng, n, T)
    c_I, c_W = 1.0, float(rng.choice([0.25, 0.5, 1.0, 2.0]))
    ilp = solve_ilp(q, s, L, c_I, c_W, time_limit=60, mip_rel_gap=1e-9)
    brute = solve_by_enumeration(q, s, L, c_I, c_W)
    assert ilp["optimal"]
    assert ilp["Z"] == pytest.approx(brute["Z"], abs=1e-7)


def test_closed_forms_equal_enumeration():
    """The ILP objective of any feasible 0/1 point equals the cost listed scenario by scenario."""
    rng = np.random.default_rng(1)
    q, s, L = random_instance(rng, 6, 4)
    c_I, c_W = 1.0, 0.5
    model = build_milp(q, s, L, c_I, c_W)
    planning = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}
    v = np.zeros(len(model["c"]))
    for t, patients in planning.items():
        for i in patients:
            v[model["x_index"][i, t]] = 1
        if len(patients) == 2:
            v[model["z_index"][min(patients), max(patients), t]] = 1
    assert model["c"] @ v + model["constant"] == pytest.approx(planning_cost(planning, q, s, L, c_I, c_W)["Z"])


def test_too_many_patients_raise():
    rng = np.random.default_rng(2)
    q, s, L = random_instance(rng, 9, 4)
    with pytest.raises(ValueError):
        solve_ilp(q, s, L, 1.0, 0.5, time_limit=10, mip_rel_gap=1e-9)


def test_simulation_matches_model_without_carry_over():
    """One patient per slot, durations <= slot lengths: no delay carries over, so the
    simulated mean cost converges to the model cost."""
    rng = np.random.default_rng(3)
    q = rng.uniform(0.6, 0.95, 4)
    s = np.array([15.0, 30.0, 15.0, 30.0])
    L = [30.0, 30.0, 45.0, 30.0]
    planning = single_booking(4, 4)
    shows = rng.random((200_000, 4)) < q
    sim = simulate_day(planning, s, L, shows, 1.0, 0.5, 1.0)
    assert sim["Z"].mean() == pytest.approx(planning_cost(planning, q, s, L, 1.0, 0.5)["Z"], rel=5e-3)


def test_greedy_serves_everyone():
    rng = np.random.default_rng(4)
    q, s, L = random_instance(rng, 7, 4)
    planning = greedy(q, s, L)
    assert sorted(i for p in planning.values() for i in p) == list(range(7))
    assert max(len(p) for p in planning.values()) <= 2


@pytest.mark.parametrize("seed", range(30))
def test_pairs_equal_slot_model_and_enumeration(seed):
    """With identical slots, the pair formulation finds the optimum of the slot model."""
    rng = np.random.default_rng(100 + seed)
    n, T = int(rng.integers(3, 8)), 4
    q = rng.uniform(0.5, 0.99, n)
    s = rng.choice([15.0, 30.0], n)
    c_W = float(rng.choice([0.25, 0.5, 1.0, 2.0]))
    pairs = solve_pairs(q, s, 30.0, T, 1.0, c_W, time_limit=60, mip_rel_gap=1e-9)
    slots = solve_ilp(q, s, [30.0] * T, 1.0, c_W, time_limit=60, mip_rel_gap=1e-9)
    assert pairs["optimal"] and slots["optimal"]
    assert pairs["Z"] == pytest.approx(slots["Z"], abs=1e-7)
    if n <= 6:
        assert pairs["Z"] == pytest.approx(solve_by_enumeration(q, s, [30.0] * T, 1.0, c_W)["Z"], abs=1e-7)


def test_overtime_counted_after_the_end_of_the_day():
    """Two patients of 30 min in the last slot of 30 min: the second one ends 30 min late."""
    sim = simulate_day({0: [0], 1: [1, 2]}, np.array([30.0, 30.0, 30.0]), [30.0, 30.0],
                       np.ones((1, 3), dtype=bool), 1.0, 0.5, 1.0)
    assert sim["overtime"][0] == 30.0 and sim["wait"][0] == 30.0 and sim["idle"][0] == 0.0
    assert sim["Z"][0] == 0.5 * 30.0 + 1.0 * 30.0


def test_pairs_last_keeps_the_cost_of_the_model():
    rng = np.random.default_rng(5)
    q, s, _ = random_instance(rng, 7, 4)
    L = [30.0] * 4
    res = solve_pairs(q, s, 30.0, 4, 1.0, 0.5, 60.0, 1e-9)
    ordered = pairs_last(res["planning"])
    assert planning_cost(ordered, q, s, L, 1.0, 0.5)["Z"] == pytest.approx(res["Z"])
    sizes = [len(ordered[t]) for t in sorted(ordered)]
    assert sizes == sorted(sizes)
