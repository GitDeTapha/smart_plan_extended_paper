"""Audit P3: perturbations with common random numbers, regret evaluated exactly."""
import numpy as np
import pandas as pd

from conftest import random_instance
from ilp_campaign import perturbed_instance, sensitivity_table


def test_perturbations_share_their_draws_across_levels():
    inst = random_instance(0, n=20, days=1, slots_per_day=5)
    lo = perturbed_instance(inst, "misclass", 0.05, seed=7, s_A=36.0, s_B=7.0)
    hi = perturbed_instance(inst, "misclass", 0.20, seed=7, s_A=36.0, s_B=7.0)
    swapped_lo, swapped_hi = lo.s != inst.s, hi.s != inst.s
    assert np.all(swapped_hi[swapped_lo])                  # nested swaps: same uniforms
    n1 = perturbed_instance(inst, "q_noise", 0.05, seed=7, s_A=36.0, s_B=7.0)
    n2 = perturbed_instance(inst, "q_noise", 0.05, seed=7, s_A=36.0, s_B=7.0)
    np.testing.assert_array_equal(n1.q, n2.q)
    assert np.all((n1.q > 0) & (n1.q < 1))
    d = perturbed_instance(inst, "duration", 0.10, seed=7, s_A=36.0, s_B=7.0)
    np.testing.assert_allclose(d.s, inst.s * 1.1)


def test_table_flags_significance_only_with_a_paired_test():
    df = pd.DataFrame({"kind": ["q_shift"] * 6, "level": [0.1] * 6,
                       "Z_hat": [11, 12, 13, 14, 15, 16.0], "Z_opt": [10, 11, 12, 13, 14, 15.0],
                       "Zsim_hat": [11, 12, 13, 14, 15, 16.0], "Zsim_opt": [10, 11, 12, 13, 14, 15.0]})
    t = sensitivity_table(df)
    assert bool(t.loc[0, "Significatif (5 %)"])
    t0 = sensitivity_table(df.assign(Z_hat=df["Z_opt"], Zsim_hat=df["Zsim_opt"]))
    assert not bool(t0.loc[0, "Significatif (5 %)"])
