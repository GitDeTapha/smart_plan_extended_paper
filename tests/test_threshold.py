"""rec. 11: fine c_W / c_I grid and the per-patient threshold 1/q_i."""
import numpy as np

from data_driven_plan import Config, Instance, solve_ilp
from ilp_campaign import CampaignConfig, threshold_specs


def test_fine_grid_covers_one_to_one_and_a_half():
    cc = CampaignConfig.production(n_workers=1)
    specs = threshold_specs(cc)
    assert sorted({s["cW_ratio"] for s in specs}) == [round(1 + 0.05 * k, 2) for k in range(11)]
    assert len(specs) == 11 * cc.R


def test_threshold_is_individual():
    # Two identical short patients, one slot of 60 min: no saturation. With
    # q = (0.9, 0.6) the patient served first is the one listed first in SPT
    # ties (index order); the double pays iff c_W/c_I < 1/q_first.
    def doubled(cw, q):
        inst = Instance(n=2, D=1, T=[1], L=[[60.0]], s=np.array([20.0, 20.0]), q=np.array(q),
                        c_I=1.0, c_W=cw, c_O=0.0)
        return solve_ilp(inst, Config(require_optimal=True)).n_double == 1
    # The single slot keeps the patient with q = 0.9; adding the other one saves
    # c_I * 0.6 * 20 of idle and costs c_W * 0.9 * 0.6 * 20 of waiting: the double
    # pays iff c_W / c_I < 1 / 0.9 = 1.111, the threshold of the patient already there.
    q = [0.9, 0.6]
    assert doubled(1.10, q) and not doubled(1.12, q)
