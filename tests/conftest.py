"""Shared fixtures: make the project module importable and keep plotting headless."""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest


@pytest.fixture
def illustrative_instance():
    """Table 1 instance of the manuscript with the historical durations (s_A=43, s_B=8)."""
    from data_driven_plan import Instance
    s_A, s_B = 43.0, 8.0
    return Instance(n=10, D=1, T=[4], L=[[30.0, 30.0, 60.0, 60.0]],
                    s=np.array([s_A] * 3 + [s_B] * 7),
                    q=np.array([0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45]),
                    c_I=1.0, c_W=0.5, c_O=0.5)


def random_instance(seed, n=30, days=4, slots_per_day=5, long_share=0.15,
                    s_A=36.0, s_B=7.0, pi_A=0.35, p_bar=0.2, c_W=0.5, c_O=0.5):
    """Self-contained random instance (independent of the notebook generator)."""
    from data_driven_plan import Instance
    rng = np.random.default_rng(seed)
    S = days * slots_per_day
    n_long = round(long_share * S)
    slot_len = rng.permutation([60.0] * n_long + [30.0] * (S - n_long)).reshape(days, slots_per_day)
    s = np.where(rng.random(n) < pi_A, s_A, s_B)
    q = np.clip(rng.beta(6.111 * (1 - p_bar), 6.111 * p_bar, n), 0.01, 0.999)
    return Instance(n=n, D=days, T=[slots_per_day] * days, L=[r.tolist() for r in slot_len],
                    s=s, q=q, c_I=1.0, c_W=c_W, c_O=c_O)
