"""Re-solve the two DOE cells that reached the 600 s limit, on the same instances, with a longer limit.

Run from the repository root:
    python reports/ilp_full_20261030/resolve_time_limited.py 3600 r175 r200
The instances are rebuilt from their cell (label, seed, factors) exactly as in the
campaign; the B3 warm-start cost printed must equal warm_start_cost in results/doe_ilp.csv.
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from ilp_campaign import CampaignConfig, CampaignContext, cell_spec, solve_record

limit = float(sys.argv[1])
P = json.load(open("data/inference/plne_parameters.json"))
cc = dataclasses.replace(CampaignConfig.production(n_workers=1), time_limit=limit)
ctx = CampaignContext(cc=cc, s_A=float(P["s_A"]), s_B=float(P["s_B"]), pi_A=float(P["pi_A_actual"]),
                      c_O=0.0, slot_lengths=(30.0, 60.0), cache_root="/nonexistent")
specs = {"r175": cell_spec(cc, "ratio_1.75", 1016, ratio=1.75),
         "r200": cell_spec(cc, "ratio_2.0", 1016, ratio=2.0)}
for name in sys.argv[2:]:
    inst = ctx.instance(specs[name])
    rec = solve_record(inst, cc.solver_config())
    print(json.dumps({"cell": name, "n": inst.n, "limit": limit, **{k: rec[k] for k in (
        "status", "gap", "gap_complete", "Z_complete", "best_bound", "solve_time", "warm_start_cost")}}),
          flush=True)
