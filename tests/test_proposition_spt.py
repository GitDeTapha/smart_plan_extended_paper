"""Audit P4: Proposition 1 on saturated pairs, SPT dominance by exact enumeration."""
import numpy as np
import pytest

from data_driven_plan import Config, Instance, eval_slot_exact, solve_ilp


def _saturated_instance(seed):
    """Long patients in 60-minute slots: any pair (36 + 30 > 60 minutes) is saturated
    (delta != 0, gamma != 0); four patients in two slots, so the optimum doubles both."""
    rng = np.random.default_rng(seed)
    n = 4    # every patient served: at most two per slot
    s = np.where(np.arange(n) % 2 == 0, 36.0, 30.0)
    q = rng.uniform(0.6, 0.95, n)
    return Instance(n=n, D=1, T=[2], L=[[60.0, 60.0]], s=s, q=q, c_I=1.0, c_W=0.5, c_O=0.5)


@pytest.mark.parametrize("seed", range(5))
def test_proposition_1_on_saturated_pairs(seed):
    inst = _saturated_instance(seed)
    cont = solve_ilp(inst, Config(z_binary=False, require_optimal=True))
    binary = solve_ilp(inst, Config(z_binary=True, require_optimal=True))
    delta, gamma = cont.coeffs["delta"], cont.coeffs["gamma"]
    saturated = [(d, t) for (d, t), p in cont.planning.items() if len(p) == 2
                 and (abs(delta[min(p), max(p), d, t]) > 1e-9 or abs(gamma[min(p), max(p), d, t]) > 1e-9)]
    assert saturated, "instance does not exercise a saturated pair"
    assert cont.Z_complete == pytest.approx(binary.Z_complete, rel=1e-6)
    assert cont.options["fractional_z"] == 0


def test_spt_dominance_by_enumeration():
    rng = np.random.default_rng(3)
    for _ in range(300):
        q = rng.uniform(0.3, 0.99, 2)
        s = rng.uniform(5, 80, 2)
        L = float(rng.choice([30.0, 60.0]))
        long_first = [int(np.argmax(s)), int(np.argmin(s))]
        spt = eval_slot_exact(long_first, q, s, L, spt=True)
        lpt = eval_slot_exact(long_first, q, s, L, spt=False)
        assert spt[1] <= lpt[1] + 1e-12                 # waiting
        assert spt[0] == pytest.approx(lpt[0])          # idle does not depend on the order
        assert spt[2] == pytest.approx(lpt[2])          # nor does overtime
