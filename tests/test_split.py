"""rec. 13: fit / valid / calib inside train; test and holdout unchanged; patient groups."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import data_driven_plan as ddp


@pytest.fixture
def frame(tmp_path, monkeypatch):
    monkeypatch.setattr(ddp, "INFERENCE_DIR", str(tmp_path))
    rng = np.random.default_rng(0)
    n = 20_000
    return pd.DataFrame({"patientsid": rng.integers(0, 3_000, n).astype("uint32"),
                         "NoShow": (rng.random(n) < 0.05).astype(int),
                         "service_time": rng.integers(1, 120, n), "x": rng.normal(size=n)})


def test_stratified_split_is_disjoint_and_keeps_test_and_holdout(frame):
    sp = ddp.make_master_split(frame, random_state=0)
    parts = [sp.fit, sp.valid, sp.calib, sp.test, sp.inference]
    idx = np.concatenate([p.index.to_numpy() for p in parts])
    assert len(idx) == len(set(idx)) == len(frame)
    assert set(sp.train.index) == set(sp.fit.index) | set(sp.valid.index) | set(sp.calib.index)
    assert len(sp.fit) / len(sp.train) == pytest.approx(0.8, abs=1e-3)
    # the former two-step split (same seeds) gives the same test and holdout rows
    df_s = frame.reset_index(drop=True)
    key = df_s["NoShow"].astype(str) + "_" + pd.qcut(df_s["service_time"], 10, duplicates="drop").astype(str)
    inf, mod = train_test_split(df_s, test_size=0.9, random_state=0, stratify=key)
    _, te = train_test_split(mod, test_size=0.2 / 0.9, random_state=0, stratify=key.loc[mod.index])
    assert set(te.index) == set(sp.test.index) and set(inf.index) == set(sp.inference.index)


def test_patient_group_split_keeps_patients_together(frame):
    sp = ddp.make_master_split(frame, split_mode="patient_group", random_state=0)
    tr, te, inf = (set(p["patientsid"]) for p in (sp.train, sp.test, sp.inference))
    assert not tr & te and not tr & inf and not te & inf
    assert 0.6 < len(sp.train) / len(frame) < 0.8


def test_patient_group_needs_the_identifier(frame):
    with pytest.raises(ValueError):
        ddp.make_master_split(frame.drop(columns="patientsid"), split_mode="patient_group")
