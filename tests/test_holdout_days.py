"""rec. 10: days built from holdout patients, judged on true groups and realised attendance."""
import dataclasses

import numpy as np
import pandas as pd

from ilp_campaign import CampaignConfig, CampaignContext, holdout_day, holdout_day_specs


def test_holdout_days(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    scores = pd.DataFrame({"q_i": rng.uniform(0.7, 0.99, n),
                           "group_pred": np.where(rng.random(n) < 0.35, "A", "B"),
                           "group_actual": np.where(rng.random(n) < 0.35, "A", "B"),
                           "no_show": (rng.random(n) < 0.1).astype(int)})
    cc = dataclasses.replace(CampaignConfig.smoke(n_workers=1), slots_per_day=4, holdout_days=3)
    ctx = CampaignContext(cc=cc, s_A=36.0, s_B=7.0, pi_A=0.35, c_O=0.5, slot_lengths=(30.0, 60.0),
                          cache_root=str(tmp_path))
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
