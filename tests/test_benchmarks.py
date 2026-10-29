"""rec. 8: benchmark policies and paired statistics."""
import numpy as np
import pytest

from conftest import random_instance
from data_driven_plan import Config
from ilp_campaign import (BENCHMARKS, bench_b5_uniform, model_cost, paired_summary,
                          solve_record)


def test_uniform_overbooking_rotates_up_to_two_per_slot():
    inst = random_instance(0, n=13, days=1, slots_per_day=5)
    plan = bench_b5_uniform(inst)["planning"]
    assert [sorted(plan[0, t]) for t in range(5)] == [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
    assert sum(len(p) for p in plan.values()) == 10      # 2 x 5 seats, 3 patients left out


def test_no_benchmark_beats_a_proven_optimum():
    inst = random_instance(2, n=8, days=2, slots_per_day=4, long_share=0.25)
    cfg = Config(require_optimal=True)
    opt = solve_record(inst, cfg)
    assert opt["status"] == "optimal"
    for name, (_, fn) in BENCHMARKS.items():
        rec = fn(inst, Config())
        assert max(len(p) for p in rec["planning"].values()) <= (1 if name == "B1" else 2)
        assert model_cost(inst, rec["planning"]) >= opt["Z_complete"] - 1e-6, name


def test_paired_summary():
    ref = np.array([10.0, 12.0, 11.0, 13.0, 9.0, 10.5])
    out = paired_summary(ref + 1.0, ref)
    assert out["mean_diff"] == pytest.approx(1.0) and out["ci_low"] == pytest.approx(1.0)
    assert out["n_better"] == 0 and out["p_wilcoxon"] < 0.05
    assert paired_summary(ref, ref)["p_wilcoxon"] == 1.0
