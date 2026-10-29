"""Audit B3: exact 1-D K-medians fitted on training durations only."""
import itertools

import numpy as np
import pytest

from data_driven_plan import bootstrap_silhouette, fit_duration_groups, kmedians_1d_exact


def brute_force(v, k):
    x = np.sort(v)
    best = (np.inf, None)
    for cuts in itertools.combinations(range(1, len(x)), k - 1):
        segs = np.split(x, cuts)
        cost = sum(np.abs(sg - np.median(sg)).sum() for sg in segs)
        if cost < best[0] - 1e-9:
            best = (cost, cuts)
    return best[0]


@pytest.mark.parametrize("seed,k", [(0, 2), (1, 2), (2, 3), (3, 3)])
def test_dynamic_programme_is_exact(seed, k):
    v = np.random.default_rng(seed).integers(1, 60, 14).astype(float)
    assert kmedians_1d_exact(v, k)["cost"] == pytest.approx(brute_force(v, k))


def test_groups_and_cutoff():
    v = np.r_[np.full(700, 7.0), np.full(300, 36.0), [5.0, 9.0, 30.0, 40.0]]
    g = fit_duration_groups(v, 2, "kmedians", "median")
    assert g.s_by_group == {"A": 36.0, "B": 7.0}
    assert g.cutoff == pytest.approx(21.5)
    np.testing.assert_array_equal(g.labels([21.0, 22.0]), ["B", "A"])
    assert g.shares["A"] == pytest.approx(302 / 1004)


def test_kmeans_ablation_and_validation():
    v = np.r_[np.full(50, 5.0), np.full(50, 50.0)]
    g = fit_duration_groups(v, 2, "kmeans", "mean")
    assert g.s_by_group == {"A": 50.0, "B": 5.0}
    with pytest.raises(ValueError):
        fit_duration_groups(v, 2, "kmedoids")
    with pytest.raises(ValueError):
        kmedians_1d_exact(np.array([1.0, np.nan]), 2)


def test_bootstrap_uses_a_local_generator():
    v = np.random.default_rng(0).integers(1, 90, 5000).astype(float)
    g = fit_duration_groups(v, 2)
    np.random.seed(1); a = bootstrap_silhouette(v, g, 3, 500, random_state=4)
    np.random.seed(2); b = bootstrap_silhouette(v, g, 3, 500, random_state=4)
    assert a == b


def test_export_validates_and_records_provenance(tmp_path):
    import json
    import pandas as pd
    from data_driven_plan import export_holdout_scores, export_plne_parameters
    v = np.r_[np.full(700, 7.0), np.full(300, 36.0)]
    g = fit_duration_groups(v, 2)
    scores = pd.DataFrame({"q_i": [0.9, 0.95, 0.97, 0.99, 0.8], "group_pred": ["A", "B", "B", "B", "B"],
                           "group_actual": ["A", "A", "B", "B", "B"], "no_show": [0, 0, 1, 0, 0]})
    P = export_plne_parameters([str(tmp_path / "p.json")], g, scores, n_train=1000,
                               source_run="run_test", smoke=True, slot_lengths=(30, 60))
    assert json.loads((tmp_path / "p.json").read_text()) == P
    assert P["pi_A_actual"] == 0.4 and P["pi_A_predicted"] == 0.2 and P["group_error_rate"] == 0.2
    assert P["git_commit"]
    with pytest.raises(ValueError):                           # s_A = 36 > 30: B4 stops the export
        export_plne_parameters([str(tmp_path / "q.json")], g, scores, 1000, "run_test", True, (20, 30))
    export_holdout_scores(str(tmp_path / "h.parquet"), scores)
    assert len(pd.read_parquet(tmp_path / "h.parquet")) == 5
