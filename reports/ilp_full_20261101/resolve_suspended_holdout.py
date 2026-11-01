"""Re-solve the holdout days whose ILPs ran while the laptop was suspended, in an empty cache.

Run from the repository root:
    python reports/ilp_full_20261101/resolve_suspended_holdout.py <out_dir>
The days are rebuilt with holdout_day_specs exactly as in the campaign (same
holdout sample, same seed); <out_dir>/holdout_days_resolved.csv has the columns of
results/holdout_days.csv.
"""
import functools
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from ilp_campaign import (CampaignConfig, CampaignContext, holdout_day, holdout_day_specs,
                          load_holdout_scores, run_lpt)

out = sys.argv[1]
P = json.load(open("data/inference/plne_parameters.json"))
holdout = load_holdout_scores("results/run_full_20261030/holdout_scores.parquet")
q_pool = np.random.default_rng(0).choice(holdout["q_i"].to_numpy(), 100_000, replace=False)
cc = CampaignConfig.production(n_workers=3)
ctx = CampaignContext(cc=cc, s_A=float(P["s_A"]), s_B=float(P["s_B"]), pi_A=float(P["pi_A_actual"]),
                      c_O=0.0, slot_lengths=(30.0, 60.0), cache_root=os.path.join(out, "cache"), q_pool=q_pool)
todo = [s for s in holdout_day_specs(cc, holdout, seed=0) if s["day"] in (24, 27, 28)]
t0 = time.time()
rows = pd.DataFrame(run_lpt(functools.partial(holdout_day, ctx), todo, lambda sp: 1.0, 3))
rows.to_csv(os.path.join(out, "holdout_days_resolved.csv"), index=False)
print(f"{time.time() - t0:.0f} s")
print(rows[["day", "ILP_status", "ILP_oracle_status", "ILP_Zmodel", "ILP_oracle_Zmodel"]])
