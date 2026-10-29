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
