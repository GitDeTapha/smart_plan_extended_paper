"""Audit B4 and 2.3: the ILP parameters come only from plne_parameters.json, checked."""
import json

import pytest

from data_driven_plan import check_plne_parameters, load_plne_parameters

SLOTS = (30, 60)


def valid():
    return {"schema_version": 2, "duration_method": "kmedians", "duration_stat": "median",
            "cutoff_minutes": 21.5, "s_A": 36.0, "s_B": 7.0, "pi_A_actual": 0.346,
            "pi_A_predicted": 0.35, "noshow_rate": 0.032, "group_error_rate": 0.2,
            "n_train": 1000, "n_holdout": 100, "created": "2026-10-29T00:00:00",
            "git_commit": "abc", "source_run": "run_test", "smoke": True}


def test_valid_parameters_pass(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(valid()))
    assert load_plne_parameters(str(path), SLOTS)["s_A"] == 36.0


@pytest.mark.parametrize("key,value", [
    ("pi_A_actual", 0.04), ("pi_A_actual", 0.5),      # 0.05 < pi_A < 0.5
    ("s_B", 0.0), ("s_B", 40.0),                      # s_A > s_B > 0
    ("s_A", 64.25),                                   # s_A <= max(SLOT_LENGTHS): run of 26/09
    ("schema_version", 1), ("noshow_rate", 1.2), ("s_A", "36"),
])
def test_aberrant_values_stop_the_run(key, value):
    P = valid()
    P[key] = value
    with pytest.raises(ValueError):
        check_plne_parameters(P, SLOTS)


def test_missing_field_and_old_schema_stop_the_run():
    P = valid()
    del P["git_commit"]
    with pytest.raises(ValueError, match="missing"):
        check_plne_parameters(P, SLOTS)
    old = {"source_run": "run_20260926_2002.ipynb", "s_A": 64.2468, "s_B": 13.0161,
           "noshow_rate": 0.032, "pi_A_predicted": 0.4215, "pi_A_actual": 0.1548}
    with pytest.raises(ValueError):
        check_plne_parameters(old, SLOTS)
