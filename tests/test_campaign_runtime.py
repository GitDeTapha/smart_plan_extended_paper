"""rec. 6: campaign sizes, solution cache, LPT runner, warm start from B3."""
import numpy as np
import pytest

from conftest import random_instance
from data_driven_plan import Config, solve_ilp
from ilp_campaign import (CampaignConfig, SolutionCache, greedy_b3, model_cost, run_lpt,
                          solve_record, spec_key)


def test_production_and_smoke_sizes():
    prod, smoke = CampaignConfig.production(n_workers=1), CampaignConfig.smoke(n_workers=1)
    assert (prod.S, prod.R, prod.days) == (8, 30, 1)       # one clinic day (decision of 29/10)
    assert (smoke.S, smoke.R) == (5, 2)
    assert prod.mip_gap == 1e-6 and prod.time_limit == 600.0
    assert prod.solver_config().threads == 1
    assert max(prod.ratios) <= 2.0                           # every patient fits two per slot


def test_cache_computes_once(tmp_path):
    cache = SolutionCache(str(tmp_path))
    calls = []
    key = spec_key(cell="central", seed=1, q=np.array([0.5, 0.6]))
    for _ in range(3):
        out = cache.get_or_compute(key, lambda: calls.append(1) or {"v": 1})
    assert out == {"v": 1} and len(calls) == 1
    assert spec_key(cell="central", seed=2) != spec_key(cell="central", seed=1)


def _square(x):
    return x * x


def test_lpt_runner_keeps_the_task_order():
    assert run_lpt(_square, [1, 5, 2, 4], cost=lambda x: x, n_workers=2) == [1, 25, 4, 16]


def test_warm_start_from_b3_keeps_the_optimum():
    inst = random_instance(3, n=8, days=2, slots_per_day=4, long_share=0.25)
    greedy = greedy_b3(inst)
    assert max(len(p) for p in greedy.values()) <= 2
    cold = solve_ilp(inst, Config(require_optimal=True))
    warm = solve_ilp(inst, Config(require_optimal=True), warm_start=greedy)
    assert warm.Z_complete == pytest.approx(cold.Z_complete, rel=1e-6)
    assert warm.Z_complete <= model_cost(inst, greedy) + 1e-9
    rec = solve_record(inst, Config(require_optimal=True))
    assert rec["status"] == "optimal" and rec["warm_start_cost"] >= rec["Z_complete"] - 1e-9
