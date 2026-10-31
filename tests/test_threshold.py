"""Every patient served (C1 as an equality) and the per-patient threshold 1/q_i
of the variant where patients may be left out."""
import numpy as np
import pytest

from data_driven_plan import Config, Instance, solve_ilp


def _two_patients(cw, q):
    # Two identical short patients, one slot of 60 min: no saturation.
    return Instance(n=2, D=1, T=[1], L=[[60.0]], s=np.array([20.0, 20.0]), q=np.array(q),
                    c_I=1.0, c_W=cw, c_O=0.0)


def test_every_patient_is_served():
    # The double costs more waiting than it saves idle (c_W / c_I > 1/q_i), yet
    # both patients are scheduled: leaving one out is no longer a decision.
    sol = solve_ilp(_two_patients(2.0, [0.9, 0.6]), Config(require_optimal=True))
    assert sol.n_served == 2 and sol.n_double == 1


def test_too_many_patients_raise():
    inst = Instance(n=3, D=1, T=[1], L=[[60.0]], s=np.full(3, 20.0), q=np.full(3, 0.8),
                    c_I=1.0, c_W=0.5, c_O=0.0)
    with pytest.raises(ValueError, match="cannot fit"):
        solve_ilp(inst, Config(require_optimal=True))


def test_threshold_is_individual_when_patients_may_be_left_out():
    # With C1 as an inequality the single slot keeps the patient with q = 0.9;
    # adding the other one saves c_I * 0.6 * 20 of idle and costs
    # c_W * 0.9 * 0.6 * 20 of waiting: the double pays iff c_W / c_I < 1 / 0.9.
    def doubled(cw):
        cfg = Config(require_optimal=True, serve_all=False)
        return solve_ilp(_two_patients(cw, [0.9, 0.6]), cfg).n_double == 1
    assert doubled(1.10) and not doubled(1.12)
