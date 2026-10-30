"""rec. 10: days built from holdout patients, judged on true groups and realised attendance."""
import dataclasses

import numpy as np
import pandas as pd

from ilp_campaign import (Q_CLIP, CampaignConfig, CampaignContext, _holdout_instance, holdout_day,
                          holdout_day_specs)


def _scores(q):
    rng = np.random.default_rng(0)
    n = len(q)
    return pd.DataFrame({"q_i": q,
                         "group_pred": np.where(rng.random(n) < 0.35, "A", "B"),
                         "group_actual": np.where(rng.random(n) < 0.35, "A", "B"),
                         "no_show": (rng.random(n) < 0.1).astype(int)})


def _context(tmp_path):
    cc = dataclasses.replace(CampaignConfig.smoke(n_workers=1), slots_per_day=4, holdout_days=3)
    return cc, CampaignContext(cc=cc, s_A=36.0, s_B=7.0, pi_A=0.35, c_O=0.5, slot_lengths=(30.0, 60.0),
                               cache_root=str(tmp_path))


def test_holdout_days(tmp_path):
    scores = _scores(np.random.default_rng(0).uniform(0.7, 0.99, 200))
    cc, ctx = _context(tmp_path)
    specs = holdout_day_specs(cc, scores, seed=1)
    patients = [tuple(s["q"]) for s in specs]
    assert len(specs) == 3 and len(set(patients)) == 3          # days do not share patients
    row = holdout_day(ctx, specs[0])
    assert row["n"] == 6
    # the oracle plans on the truth, so it cannot lose under the true model cost if proven optimal
    if row["ILP_oracle_status"] == "optimal":
        assert row["ILP_oracle_Zmodel"] <= row["ILP_Zmodel"] + 1e-6
    for k in ("ILP", "B1", "B3", "B5"):
        assert np.isfinite(row[f"{k}_Zreal"]) and row[f"{k}_Zday"] > 0


def test_isotonic_end_blocks_are_clipped_in_holdout_days(tmp_path):
    # isotonic calibration can predict q_i = 0 or 1 exactly; specs keep them, instances clip them
    scores = _scores(np.tile([0.0, 1.0], 100))
    cc, ctx = _context(tmp_path)
    spec = holdout_day_specs(cc, scores, seed=1)[0]
    spec["q"][:2] = [0.0, 1.0]                                    # both end blocks in the day
    q = _holdout_instance(ctx, spec, spec["group_pred"]).q
    assert q.min() == Q_CLIP[0] and q.max() == Q_CLIP[1]
    row = holdout_day(ctx, spec)
    assert np.isfinite(row["ILP_Zreal"])
