"""Audit P2: Beta law of q in (mean, concentration) form, no floor."""
import numpy as np
import pandas as pd
import pytest

from ilp_campaign import (BETA_CONCENTRATION, Q_CLIP, beta_parameters, check_beta_mean,
                          check_holdout_scores, draw_q_empirical, generate_instance)


@pytest.mark.parametrize("p_bar", [0.03, 0.10, 0.20, 0.30])
def test_mean_of_q_is_one_minus_p_bar(p_bar):
    # assertion of the audit: |empirical mean - (1 - p_bar)| < 0.01 on 1e5 draws
    assert abs(check_beta_mean(p_bar, n=100_000) - (1 - p_bar)) < 0.01


def test_central_cell_keeps_its_former_spread():
    a, b = beta_parameters(0.80)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    assert sd == pytest.approx(0.15, abs=1e-3)
    assert a + b == pytest.approx(BETA_CONCENTRATION)


def test_former_floor_is_gone():
    # the old generator floored beta at 0.5, which gave mean 0.5 at p_bar = 0.03
    a, b = beta_parameters(0.97)
    assert b < 0.5


def test_empirical_variant_resamples_the_pool():
    pool = np.array([0.6, 0.9, 0.95])
    q = draw_q_empirical(np.random.default_rng(0), pool, 1000)
    assert set(np.round(q, 6)) <= set(pool)
    with pytest.raises(ValueError):
        draw_q_empirical(np.random.default_rng(0), np.array([0.5, 1.2]), 3)


def test_isotonic_end_blocks_are_clipped_when_drawn():
    # isotonic calibration can predict P(no-show) = 0 or 1 exactly
    q = draw_q_empirical(np.random.default_rng(0), np.array([0.0, 1.0]), 1000)
    assert set(q) == set(Q_CLIP)


def test_generator_uses_the_pool_and_shares_the_grid():
    kw = dict(days=2, slots_per_day=5, slot_lengths=(30, 60), ratio=1.5, long_share=0.2,
              cW_ratio=0.5, s_A=36.0, s_B=7.0, pi_A=0.35, c_O=0.5)
    beta = generate_instance(1, p_bar=0.2, **kw)
    emp = generate_instance(1, p_bar=0.2, q_pool=np.array([0.97]), **kw)
    assert np.allclose(emp.q, 0.97)
    assert beta.L == emp.L and np.array_equal(beta.s, emp.s)   # paired design


def test_holdout_scores_are_checked():
    ok = pd.DataFrame({"q_i": [0.9, 0.8], "group_pred": ["A", "B"],
                       "group_actual": ["B", "B"], "no_show": [0, 1]})
    check_holdout_scores(ok)
    with pytest.raises(ValueError):
        check_holdout_scores(ok.drop(columns="no_show"))
    check_holdout_scores(ok.assign(q_i=[1.0, 0.0]))     # isotonic end blocks, kept as predicted
    with pytest.raises(ValueError):
        check_holdout_scores(ok.assign(q_i=[1.2, 0.8]))
    with pytest.raises(ValueError):
        check_holdout_scores(ok.assign(q_i=[-0.1, 0.8]))
