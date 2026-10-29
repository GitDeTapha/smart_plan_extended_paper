"""Audit 1.7: stages are resumable; a stage with a matching .done is not recomputed."""
import numpy as np
import pytest

from data_driven_plan import StageRunner


def test_done_stage_is_loaded_not_recomputed(tmp_path):
    kw = dict(root=str(tmp_path / "models"), status_dir=str(tmp_path / "runs"))
    run = StageRunner("t1", target="durations", **kw)
    run.configure("prep", smoke=True)
    assert run.compute("prep")
    run.finish("prep", X=np.ones((3, 2), dtype=np.float32), n=3)
    run.configure("durations", method="kmedians")
    assert run.compute("durations")
    run.finish("durations", s_A=36.0)
    assert (tmp_path / "runs" / "t1" / "STATUS.md").read_text().count("| done |") == 2

    again = StageRunner("t1", target="all", **kw)
    again.configure("prep", smoke=True)
    assert not again.compute("prep")                       # same config: skipped
    art = again.load("prep")
    assert art["n"] == 3 and art["X"].shape == (3, 2) and isinstance(art["X"], np.memmap)
    again.configure("durations", method="kmeans")          # changed config: recomputed
    assert again.compute("durations")
    again.configure("tune", models=["XGBoost"])
    assert again.compute("tune")


def test_upstream_change_invalidates_downstream(tmp_path):
    kw = dict(root=str(tmp_path / "m"), status_dir=str(tmp_path / "r"))
    run = StageRunner("t2", **kw)
    run.configure("prep", smoke=True); run.compute("prep"); run.finish("prep")
    run.configure("durations", method="kmedians"); run.compute("durations"); run.finish("durations")
    other = StageRunner("t2", **kw)
    other.configure("prep", smoke=False)
    other.configure("durations", method="kmedians")
    assert other.compute("durations")


def test_stages_beyond_the_target_are_skipped(tmp_path):
    run = StageRunner("t3", target="prep", root=str(tmp_path / "m"), status_dir=str(tmp_path / "r"))
    run.configure("durations", method="kmedians")
    assert not run.wanted("durations") and not run.compute("durations")
    with pytest.raises(ValueError):
        StageRunner("t4", target="train", root=str(tmp_path / "m"), status_dir=str(tmp_path / "r"))
