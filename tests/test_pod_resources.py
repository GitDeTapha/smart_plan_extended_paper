"""Pod readiness: CPU count of a container, from the affinity mask and the cgroup quota."""
import os

import pytest

from data_driven_plan import available_cpus, resolve_n_jobs


def test_cgroup_v2_quota_caps_the_cpus(tmp_path):
    (tmp_path / "cpu.max").write_text("400000 100000\n")      # 4 CPUs
    assert available_cpus(str(tmp_path)) == min(4, len(os.sched_getaffinity(0)))
    (tmp_path / "cpu.max").write_text("max 100000\n")         # no quota
    assert available_cpus(str(tmp_path)) == len(os.sched_getaffinity(0))


def test_cgroup_v1_quota(tmp_path):
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("200000")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert available_cpus(str(tmp_path)) == min(2, len(os.sched_getaffinity(0)))


def test_n_jobs_is_resolved_to_a_count():
    assert resolve_n_jobs(-1) == available_cpus()
    assert resolve_n_jobs(10_000) == available_cpus()
    with pytest.raises(ValueError):
        resolve_n_jobs(0)


def test_cached_smote_is_identical_to_smote(tmp_path):
    import numpy as np
    from imblearn.over_sampling import SMOTE
    from data_driven_plan import CachedSMOTE
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 8)).astype(np.float32)
    y = (rng.random(3000) < 0.03).astype(int)
    ref = SMOTE(sampling_strategy=0.1, random_state=0).fit_resample(X, y)
    first = CachedSMOTE(sampling_strategy=0.1, random_state=0, cache_dir=str(tmp_path)).fit_resample(X, y)
    assert len(list(tmp_path.glob("smote_*.npz"))) == 1
    hit = CachedSMOTE(sampling_strategy=0.1, random_state=0, cache_dir=str(tmp_path)).fit_resample(X, y)
    for out in (first, hit):
        np.testing.assert_array_equal(out[0], ref[0])
        np.testing.assert_array_equal(out[1], ref[1])
    other = CachedSMOTE(sampling_strategy=0.1, random_state=1, cache_dir=str(tmp_path)).fit_resample(X, y)
    assert len(list(tmp_path.glob("smote_*.npz"))) == 2 and not np.array_equal(other[0], ref[0])


@pytest.mark.parametrize("n_grid,n_rows,k,L", [(144, 25_284_050, 3, 2), (36, 25_284_050, 3, 2),
                                               (144, 2_528_405, 3, 2), (20, 5000, 2, 2)])
def test_halving_schedule_is_the_one_sklearn_runs(n_grid, n_rows, k, L):
    """The planned schedule ends on all of fit with L finalists, and scikit-learn,
    given its parameters, computes the same iterations, rows and candidates."""
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.model_selection import HalvingRandomSearchCV
    from data_driven_plan import TrainingConfig, halving_schedule
    frac = 0.8 / 0.9
    cfg = TrainingConfig(halving_iterations=k, last_candidates=L,
                         min_resources=500_000 if n_rows > 10_000 else 400)
    sch = halving_schedule(n_grid, n_rows, frac, cfg)
    q = n_rows // sch["min_resources"]
    assert sch["iterations"] == k and sch["candidates"][-1] == L and sch["candidates"][0] <= n_grid
    assert 0 <= n_rows - sch["rows"][-1] <= q            # all of fit + valid but fewer than q rows
    # dry run of scikit-learn's own bookkeeping on a tiny problem with the same ratios
    small = 5000
    scale = small / n_rows
    # scikit-learn's own run with exactly the planned parameters (dummy model: only the
    # schedule bookkeeping and the row subsampling cost anything)
    search = HalvingRandomSearchCV(DummyClassifier(strategy="prior"),
                                   {"constant": list(range(n_grid))},
                                   n_candidates=sch["n_candidates"], factor=sch["factor"],
                                   min_resources=sch["min_resources"], max_resources=sch["max_resources"],
                                   resource="n_samples", cv=2, random_state=0, refit=False)
    X = np.zeros((n_rows, 1), dtype=np.float32)
    y = np.arange(n_rows) % 2
    search.fit(X, y)
    assert search.n_iterations_ == k and list(search.n_candidates_) == sch["candidates"]
    assert list(search.n_resources_) == sch["rows"]


def test_search_processes_are_capped_by_memory(monkeypatch):
    import data_driven_plan as ddp
    cfg = ddp.TrainingConfig(n_jobs=16, memory_safety=0.6)
    monkeypatch.setattr(ddp, "available_memory", lambda: 128 * 2 ** 30)
    cfg.n_jobs = 16                                    # as on the 16-vCPU pod
    rows, feats = 25_284_050, 200                      # fit + valid at full scale
    jobs, threads = ddp.search_jobs(cfg, rows, feats)
    assert jobs == int(0.6 * 128 * 2 ** 30 // (2.5 * rows * feats * 4)) and jobs * threads <= 16
    assert ddp.search_jobs(cfg, 100_000, feats) == (16, 1)          # small data: one process per CPU
    monkeypatch.setattr(ddp, "available_memory", lambda: 0)
    assert ddp.search_jobs(cfg, rows, feats)[0] == 1                 # never below one process


def test_available_memory_reads_the_cgroup_limit(tmp_path):
    from data_driven_plan import available_memory
    (tmp_path / "memory.max").write_text(str(8 * 2 ** 30))
    (tmp_path / "memory.current").write_text(str(2 * 2 ** 30))
    assert available_memory(str(tmp_path)) <= 6 * 2 ** 30
