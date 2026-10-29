"""B5 (audit): costs are explicit and the solved objective is written in LaTeX."""
import numpy as np
import pytest

from data_driven_plan import Instance, objective_latex, validate_decisions


def _kwargs(**over):
    base = dict(n=2, D=1, T=[1], L=[[30.0]], s=np.array([10.0, 10.0]),
                q=np.array([0.8, 0.7]))
    base.update(over)
    return base


def test_instance_requires_every_cost():
    with pytest.raises(TypeError):
        Instance(**_kwargs())          # no silent c_O default
    with pytest.raises(TypeError):
        Instance(**_kwargs(c_I=1.0, c_W=0.5))


def test_instance_rejects_aberrant_values():
    with pytest.raises(ValueError):
        Instance(**_kwargs(c_I=1.0, c_W=0.5, c_O=-0.1))
    with pytest.raises(ValueError):
        Instance(**_kwargs(q=np.array([0.8, 1.0]), c_I=1.0, c_W=0.5, c_O=0.5))


def test_objective_latex_matches_the_costs():
    with_ot = objective_latex(1.0, 0.5, 0.5)
    assert "o_{it}" in with_ot and "1{,}5" in with_ot      # c_I + c_O = 1.5
    without_ot = objective_latex(1.0, 0.5, 0.0)
    assert "o_{it}" not in without_ot                       # manuscript eq. (9)


def test_decisions_are_validated():
    ok = validate_decisions("kmedians", "median", 0.5, (30, 60), "stratified", 0, False)
    assert ok["SLOT_LENGTHS"] == (30.0, 60.0)
    with pytest.raises(ValueError):
        validate_decisions("kmedian", "median", 0.5, (30, 60), "stratified", 0, False)
    with pytest.raises(ValueError):
        validate_decisions("kmedians", "mean", 0.5, (30, 60), "stratified", 0, False)


def test_table1_without_overtime(illustrative_instance):
    import dataclasses
    from data_driven_plan import Config, solve_ilp
    inst = dataclasses.replace(illustrative_instance, c_O=0.0)
    sol = solve_ilp(inst, Config(require_optimal=True))
    assert sol.Z_complete == pytest.approx(65.62, abs=5e-3)     # 70.82 with c_O = 0.5
    assert sol.E_OT == 0.0
