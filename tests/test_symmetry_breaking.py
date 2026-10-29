"""Audit 2.4: the optional symmetry-breaking cuts keep the optimal objective."""
import pytest

from conftest import random_instance
from data_driven_plan import Config, solve_ilp


@pytest.mark.parametrize("seed", range(20))
def test_symmetry_breaking_keeps_the_optimum(seed):
    inst = random_instance(seed, n=8, days=2, slots_per_day=4, long_share=0.25)
    base = solve_ilp(inst, Config(require_optimal=True))
    cut = solve_ilp(inst, Config(require_optimal=True, symmetry_breaking=True))
    assert cut.options["n_symmetry_cuts"] > 0
    rel = abs(cut.Z_complete - base.Z_complete) / max(abs(base.Z_complete), 1e-9)
    assert rel < 1e-6, f"seed {seed}: {base.Z_complete} vs {cut.Z_complete}"
    # The cuts must actually hold on the returned schedule
    for d in range(inst.D):
        for length in set(inst.L[d]):
            occ = [len(cut.planning.get((d, t), [])) for t in range(inst.T[d]) if inst.L[d][t] == length]
            assert occ == sorted(occ, reverse=True)
