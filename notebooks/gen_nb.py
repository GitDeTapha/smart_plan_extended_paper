"""Generate notebooks/01_ilp_validation.ipynb"""
import json, os

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

cells = []

def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":src})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src})
def params(src):
    """Code cell tagged `parameters`: papermill injects overrides right after it."""
    cells.append({"cell_type":"code","execution_count":None,"metadata":{"tags":["parameters"]},
                  "outputs":[],"source":src})

# ── Section 0 ──────────────────────────────────────────────────────────────────
md("# Notebook 01, Validation numérique du PLNE de rendez-vous ambulatoires\n\n"
   "**Sections 0–7** miroir des sections 11.1–11.4 du manuscrit (COR/EJOR).\n"
   "Toutes les durées sont en **minutes**. Ne pas modifier la formulation ; "
   "toute simplification computationnelle est signalée en tête de section.")

params("""\
# ── Décisions des auteurs (paramètres papermill) ─────────────────────────────
DURATION_METHOD = "kmedians"    # B3 ; "kmeans" en ablation
DURATION_STAT   = "median"      # s_A, s_B = centroïdes K-médianes
C_O_BASE        = 0.0           # V9 : pas d'overtime, comme le manuscrit (éq. 1, 8, 9) ; 0,5 pour l'extension §8.5
SLOT_LENGTHS    = (30, 60)      # B4 : assertion s_A <= max(SLOT_LENGTHS), sinon arrêt
SPLIT_MODE      = "stratified"  # "patient_group" en ablation
RANDOM_STATE    = 0
SMOKE           = False         # True : 1 % des données, budgets minimaux
PLNE_PARAMS_PATH = "data/inference/plne_parameters.json"  # exporté par data_driven_plan.ipynb
HOLDOUT_SCORES_PATH = None  # holdout_scores.parquet du même run (q_i, groupes, no-show) ; requis
CAMPAIGN_OVERRIDES = {}     # ex. {"R": 10, "time_limit": 600.0} ; voir ilp_campaign.CampaignConfig
N_WORKERS = None            # None : un processus par cœur physique
CACHE_DIR = "results/ilp_cache"  # solutions par (cellule, graine, méthode, configuration)
""")

code("""\
# ── 0. Environnement ──────────────────────────────────────────────────────────
import sys, platform, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from dataclasses import dataclass, field
from typing import Optional
import joblib
import os

# Cœur du modèle PLNE partagé avec data_driven_plan.py (racine du projet),
# garantit que ce notebook et data_driven_plan.ipynb résolvent EXACTEMENT
# le même modèle (pas de duplication de solve_ilp/Instance/Config). Ce
# notebook est exécuté avec le répertoire du projet comme répertoire courant.
sys.path.insert(0, os.getcwd())
from data_driven_plan import (
    Instance, Config, Solution, compute_coefficients, solve_ilp, SOLVER_NAME,
    objective_latex, validate_decisions,
    eval_slot_exact, eval_day_exact, evaluate_planning_exact,
    simulate_day, simulate_instance, draw_scenarios,
)
import pyomo.environ as pyo

DECISIONS = validate_decisions(DURATION_METHOD, DURATION_STAT, C_O_BASE, SLOT_LENGTHS,
                               SPLIT_MODE, RANDOM_STATE, SMOKE)
for _k, _v in DECISIONS.items():
    print(f"{_k:<16} {_v}")

RNG_SEED = RANDOM_STATE
_RNG = np.random.default_rng(RNG_SEED)

print(f"Python  : {sys.version.split()[0]}")
print(f"NumPy   : {np.__version__}")
print(f"Pandas  : {pd.__version__}")
import pyomo; print(f"Pyomo   : {pyomo.__version__}")
import psutil
def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            return next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or "unknown"
CPU_MODEL = _cpu_model()
print(f"Platform: {CPU_MODEL} | {platform.platform()}")
print(f"CPU     : {psutil.cpu_count(logical=False)} cœurs physiques, {psutil.cpu_count()} logiques")
print(f"RAM     : {psutil.virtual_memory().total / 2**30:.1f} Go")
print(f"Solveur : {SOLVER_NAME} ✓ (détecté par data_driven_plan.init_solver)")
""")

# ── Section 0.2 : paramètres issus du pipeline ───────────────────────────────
md("## 0.2 Paramètres issus du pipeline de prédiction\n\n"
   "Les durées $s_A$, $s_B$, la part du groupe long $\\pi_A$ et le taux de no-show sont lus "
   "**uniquement** dans `plne_parameters.json`, exporté par `data_driven_plan.ipynb` "
   "(K-médianes exactes ajustées sur l'apprentissage seul, provenance incluse). "
   "Aucun recalcul ici, aucune valeur par défaut : un champ absent ou hors domaine arrête "
   "le notebook (audit B4). Contrôles : $0{,}05 < \\pi_A < 0{,}5$, $s_A > s_B > 0$, "
   "$s_A \\le \\max$(`SLOT_LENGTHS`).")

code("""\
# ── 0.2 Lecture et contrôle de plne_parameters.json ──────────────────────────
from data_driven_plan import load_plne_parameters

PLNE = load_plne_parameters(PLNE_PARAMS_PATH, SLOT_LENGTHS)
s_A, s_B = float(PLNE["s_A"]), float(PLNE["s_B"])
pi_A = float(PLNE["pi_A_actual"])
L_SHORT, L_LONG = (float(v) for v in sorted(SLOT_LENGTHS))
if PLNE["duration_method"] != DURATION_METHOD or PLNE["duration_stat"] != DURATION_STAT:
    raise ValueError(f"plne_parameters.json built with {PLNE['duration_method']}/{PLNE['duration_stat']}, "
                     f"notebook expects {DURATION_METHOD}/{DURATION_STAT}")

print(f"Source           : {PLNE['source_run']} (commit {PLNE['git_commit']}, {PLNE['created']})")
print(f"Méthode          : {PLNE['duration_method']} / {PLNE['duration_stat']}, coupure {PLNE['cutoff_minutes']:.1f} min")
print(f"s_A, s_B         : {s_A:.2f} min, {s_B:.2f} min   (créneaux {L_SHORT:g} et {L_LONG:g} min)")
print(f"pi_A réel / prédit : {pi_A:.4f} / {PLNE['pi_A_predicted']:.4f}")
print(f"Taux de no-show  : {PLNE['noshow_rate']:.4f}")
print(f"Erreur du classifieur de groupe (holdout) : {PLNE['group_error_rate']:.4f}")
if PLNE["smoke"]:
    print("ATTENTION : paramètres d'un smoke test (1 % des patients), pas de résultats publiables.")
""")

# ── Section 1 : Modèle ────────────────────────────────────────────────────────
md("## 1. Modèle\n\n"
   "`Instance`, `Config`, `Solution`, `compute_coefficients` et `solve_ilp` sont "
   "importés de **data_driven_plan.py** (racine du projet), voir la cellule "
   "d'environnement ci-dessus. Ce cœur de modèle est ainsi partagé, à "
   "l'identique, avec `data_driven_plan.ipynb` (section 7).")

code("""\
# ── 1.1-1.3 Dataclasses + compute_coefficients + solve_ilp ───────────────────
# (importés de data_driven_plan, voir cellule 0 ; aucune redéfinition ici)
print("Instance/Config/Solution/compute_coefficients/solve_ilp importés depuis data_driven_plan.py")
""")

# ── Section 2 : Tests ─────────────────────────────────────────────────────────
md("## 2. Tests d'exactitude\n\n"
   "> Tous ces tests doivent passer **avant** toute expérience.")

code("""\
# ── 2.1 Ré-énumération Bernoulli (≤ 4 scénarios par créneau) ─────────────────
# eval_slot_exact (énumération des 2^k scénarios d'un créneau) est importé de
# data_driven_plan : une seule implémentation pour les tests, les benchmarks et le module.
def test_bernoulli_enum(inst: Instance, sol: Solution, tol=1e-9):
    # Compare E[Idle/Wait/OT] en minutes brutes (formule vs énumération Bernoulli).
    coeffs = sol.coeffs
    a, delta, w, o, gamma = (coeffs[k] for k in ("a","delta","w","o","gamma"))
    err_max = 0.0
    for (d,t), pats in sol.planning.items():
        L = inst.L[d][t]
        ei, ew, eo = eval_slot_exact(pats, inst.q, inst.s, L)   # minutes, non pondérées
        if len(pats) == 1:
            i = pats[0]
            mi = L - a[i,d,t]    # E[Idle] brut
            mw = 0.0
            mo = o[i,d,t]         # E[OT] brut
        else:
            pi, pj = (pats[0], pats[1]) if pats[0] < pats[1] else (pats[1], pats[0])
            mi = L - a[pi,d,t] - a[pj,d,t] - delta[pi,pj,d,t]
            mw = w[pi,pj]
            mo = o[pi,d,t] + o[pj,d,t] + gamma[pi,pj,d,t]
        err = max(abs(ei - mi), abs(ew - mw), abs(eo - mo))
        err_max = max(err_max, err)
        assert err < tol, f"Créneau ({d},{t}) pats={pats}: err={err:.2e}"
    return err_max
""")

code("""\
# ── 2.2 Monte Carlo sur un créneau double ─────────────────────────────────────
def test_montecarlo(q1=0.8, q2=0.7, s1=20.0, s2=30.0, L=40.0, M=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    show1 = rng.random(M) < q1
    show2 = rng.random(M) < q2
    # SPT : s1 < s2 → i1 first
    idle = np.zeros(M); wait = np.zeros(M); ot = np.zeros(M)
    # Both show
    both = show1 & show2
    svc_both = s1 + s2
    idle[both] = np.maximum(0, L - svc_both)
    wait[both] = s1    # j waits s1
    ot[both]   = np.maximum(0, svc_both - L)
    # Only 1
    only1 = show1 & ~show2
    idle[only1] = np.maximum(0, L - s1)
    ot[only1]   = np.maximum(0, s1 - L)
    # Only 2
    only2 = ~show1 & show2
    idle[only2] = np.maximum(0, L - s2)
    ot[only2]   = np.maximum(0, s2 - L)
    # Neither
    neither = ~show1 & ~show2
    idle[neither] = L

    mc_idle, mc_wait, mc_ot = idle.mean(), wait.mean(), ot.mean()

    # Closed-form
    cf_idle = L - q1*min(s1,L) - q2*min(s2,L) - q1*q2*(min(s1+s2,L)-min(s1,L)-min(s2,L))
    cf_wait = q1*q2*min(s1,s2)
    cf_ot   = q1*max(0,s1-L) + q2*max(0,s2-L) + q1*q2*(min(s1,L)+min(s2,L)-min(s1+s2,L))

    print(f"Monte Carlo vs Formule fermée (M={M:,}) :")
    print(f"  E[Idle] MC={mc_idle:.5f}  CF={cf_idle:.5f}  err={abs(mc_idle-cf_idle):.2e}")
    print(f"  E[Wait] MC={mc_wait:.5f}  CF={cf_wait:.5f}  err={abs(mc_wait-cf_wait):.2e}")
    print(f"  E[OT]   MC={mc_ot:.5f}  CF={cf_ot:.5f}  err={abs(mc_ot-cf_ot):.2e}")
    # Tolérance = 5×SE (SE_idle ≈ 0.012, SE_wait ≈ 0.010, SE_OT ≈ 0.005 pour M=1e6)
    assert abs(mc_idle - cf_idle) < 0.05, f"MC Idle diverge: {abs(mc_idle-cf_idle):.4f}"
    assert abs(mc_wait - cf_wait) < 0.05, f"MC Wait diverge: {abs(mc_wait-cf_wait):.4f}"
    assert abs(mc_ot   - cf_ot)   < 0.03, f"MC OT diverge: {abs(mc_ot-cf_ot):.4f}"
    print("  ✓ Monte Carlo converge vers les formules fermées.")

test_montecarlo()
""")

code("""\
# ── 2.3 Proposition 1 : z continus et z binaires, même optimum ───────────────
# Correction P4 : la proposition n'est mise à l'épreuve que si l'optimum contient
# au moins une paire saturée (delta != 0 ou gamma != 0). Instances construites
# pour cela : patients de 36 et 30 min dans des créneaux de 60 min (36 + 30 > 60).
def _saturated_instance(seed):
    rng = np.random.default_rng(seed)
    n = 6
    s = np.where(np.arange(n) % 2 == 0, 36.0, 30.0)
    q = rng.uniform(0.6, 0.95, n)
    return Instance(n=n, D=1, T=[2], L=[[60.0, 60.0]], s=s, q=q, c_I=1.0, c_W=0.5, c_O=C_O_BASE)

def test_z_integrality(n_instances=10):
    print("Proposition 1 (z continu vs z binaire) sur paires saturées :")
    for k in range(n_instances):
        inst = _saturated_instance(k)
        cont = solve_ilp(inst, Config(z_binary=False, require_optimal=True))
        binary = solve_ilp(inst, Config(z_binary=True, require_optimal=True))
        dl, gm = cont.coeffs["delta"], cont.coeffs["gamma"]
        n_sat = sum(1 for (d, t), p in cont.planning.items() if len(p) == 2
                    and max(abs(dl[min(p), max(p), d, t]), abs(gm[min(p), max(p), d, t])) > 1e-9)
        assert n_sat >= 1, f"instance {k} : aucune paire saturée à l'optimum"
        diff = abs(cont.Z_complete - binary.Z_complete)
        assert diff < 1e-6 * max(1.0, abs(binary.Z_complete)), f"Proposition 1 violée : diff={diff}"
        assert cont.options["fractional_z"] == 0
        print(f"  inst {k}: paires saturées={n_sat}, Z_cont={cont.Z_complete:.6f}, "
              f"Z_bin={binary.Z_complete:.6f}, diff={diff:.1e}  OK")
    print("Proposition 1 vérifiée.")

test_z_integrality()
""")

code("""\
# ── 2.4 Dominance SPT par énumération exacte ─────────────────────────────────
# Correction P4 : l'ancien test comparait q_i q_j min(s) à q_i q_j max(s), une
# tautologie. On évalue ici le même créneau sous les deux ordres avec eval_slot_exact.
def test_spt_dominance(n_pairs=500, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(n_pairs):
        q = rng.uniform(0.3, 0.99, 2)
        s = rng.uniform(5, 80, 2)
        L = float(rng.choice(SLOT_LENGTHS))
        long_first = [int(np.argmax(s)), int(np.argmin(s))]
        spt = eval_slot_exact(long_first, q, s, L, spt=True)
        lpt = eval_slot_exact(long_first, q, s, L, spt=False)
        assert spt[1] <= lpt[1] + 1e-12                       # attente
        assert abs(spt[0] - lpt[0]) < 1e-9 and abs(spt[2] - lpt[2]) < 1e-9  # idle, overtime
    print(f"Dominance SPT vérifiée sur {n_pairs} paires (attente plus faible, idle et overtime inchangés).")

test_spt_dominance()
print("\\n=== Tous les tests d'exactitude sont passés ===")
""")

# ── Section 3 : Exemple illustratif ──────────────────────────────────────────
md("## 3. Exemple illustratif, Table 1\n\n"
   "Instance : n=10, D=1, 4 créneaux (2×30, 2×60 min) ; patients 0 à 2 du groupe A, 3 à 9 du "
   "groupe B ; q=[0,90 ; 0,85 ; … ; 0,45] ; c_I=1, c_W=0,5, c_O=0,5 (coûts de la Table 1 du "
   "manuscrit, fixés ici indépendamment de `C_O_BASE`).\n\n"
   "Correction rec. 5 : l'exemple ne dépend plus des données. Variante **publiée** : $s_A=43$, "
   "$s_B=8$ min, $c_O=0{,}5$ (coûts de la Table 1 du manuscrit), doit redonner $Z^\\star = 70{,}82$. "
   "Variante **historique** : mêmes durées, $c_O$ = `C_O_BASE` ; avec $c_O = 0$ (modèle du "
   "manuscrit), $Z^\\star = 65{,}62$, valeur à reporter dans le manuscrit. Variante **données** : "
   "$s_A$, $s_B$ de `plne_parameters.json`, à titre d'illustration.")

code("""\
# ── 3.1 Construction et résolution des deux variantes ───────────────────────
ILLUS_Q = np.array([0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45])
ILLUS_L = [[30.0, 30.0, 60.0, 60.0]]
ILLUS_GROUPS = np.array(["A"] * 3 + ["B"] * 7)

def illustrative_instance(sA, sB, cO):
    return Instance(n=10, D=1, T=[4], L=ILLUS_L, s=np.where(ILLUS_GROUPS == "A", sA, sB),
                    q=ILLUS_Q, c_I=1.0, c_W=0.5, c_O=cO)

# "publie" : coûts de la Table 1 du manuscrit (c_O = 0,5), contrôle de non-régression de 70,82.
# "historique" et "donnees" : coût d'overtime C_O_BASE (0 : modèle du manuscrit, sans overtime).
ILLUS = {}
for _tag, (_sA, _sB, _cO) in {"publie": (43.0, 8.0, 0.5), "historique": (43.0, 8.0, C_O_BASE),
                              "donnees": (s_A, s_B, C_O_BASE)}.items():
    _inst = illustrative_instance(_sA, _sB, _cO)
    _sol = solve_ilp(_inst, Config(require_optimal=True))
    ILLUS[_tag] = (_inst, _sol)
    print(f"[{_tag}] s_A={_sA:.2f}, s_B={_sB:.2f} : statut {_sol.status}, gap {_sol.gap_complete:.1e}, "
          f"Z* = {_sol.Z_complete:.4f}  (E[Idle]={_sol.E_idle:.3f}, E[Wait]={_sol.E_wait:.3f}, "
          f"E[OT]={_sol.E_OT:.3f}), servis {_sol.n_served}/10, doubles {_sol.n_double}/4")

_Z_pub = ILLUS["publie"][1].Z_complete
assert abs(_Z_pub - 70.82) < 5e-3, f"Table 1 publiée : Z* = {_Z_pub:.4f}, attendu 70,82"
print("Table 1 publiée (c_O = 0,5) : Z* = 70,82 retrouvé.")
if C_O_BASE == 0:
    _Z_hist = ILLUS["historique"][1].Z_complete
    assert abs(_Z_hist - 65.62) < 5e-3, f"Table 1 sans overtime : Z* = {_Z_hist:.4f}, attendu 65,62"
    print("Table 1 sans overtime (c_O = 0) : Z* = 65,62, valeur à reporter dans le manuscrit.")
# Les sections 3.2 à 6.2 utilisent la variante historique (c_O = C_O_BASE).
_inst3, _sol3 = ILLUS["historique"]
""")

code("""\
# ── 3.2 Test de validation Bernoulli sur les deux variantes ──────────────────
for _tag, (_inst, _sol) in ILLUS.items():
    _err = test_bernoulli_enum(_inst, _sol)
    print(f"[{_tag}] erreur max formule fermée / énumération : {_err:.2e}")
""")

code("""\
# ── 3.3 Plannings et coefficients des paires (Table 1) ───────────────────────
def illustrative_tables(inst, sol):
    a, dl, w, o, gm = (sol.coeffs[k] for k in ("a", "delta", "w", "o", "gamma"))
    rows, pair_rows = [], []
    for (d, t) in sol.coeffs["slots"]:
        pats = sol.planning.get((d, t), [])
        ei, ew, eo = eval_slot_exact(pats, inst.q, inst.s, inst.L[d][t])
        rows.append({"Créneau": f"S{t}", "L (min)": inst.L[d][t],
                     "Patients (ordre SPT)": ", ".join(map(str, pats)) or "-",
                     "Groupes": "".join(ILLUS_GROUPS[i] for i in pats) or "-",
                     "E[Idle] (min)": inst.c_I * ei, "E[Wait] (min)": inst.c_W * ew,
                     "E[OT] (min)": inst.c_O * eo})
        if len(pats) == 2:
            i, j = min(pats), max(pats)
            pair_rows.append({"i": i, "j": j, "Créneau": f"S{t}", "L (min)": inst.L[d][t],
                              "q_i": inst.q[i], "q_j": inst.q[j], "s_i": inst.s[i], "s_j": inst.s[j],
                              "delta": dl[i, j, d, t], "w": w[i, j], "gamma": gm[i, j, d, t],
                              "coeff. z": inst.c_W * w[i, j] - (inst.c_I + inst.c_O) * dl[i, j, d, t]})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)

ILLUS_TABLES = {}
for _tag, (_inst, _sol) in ILLUS.items():
    ILLUS_TABLES[_tag] = illustrative_tables(_inst, _sol)
    print(f"=== Table 1, variante {_tag} (Z* = {_sol.Z_complete:.2f}) ===")
    print(ILLUS_TABLES[_tag][0].round(3).to_string(index=False))
    print(ILLUS_TABLES[_tag][1].round(4).to_string(index=False) if len(ILLUS_TABLES[_tag][1])
          else "Aucune paire double.")
    print()
""")

code("""\
# ── 3.4 Export LaTeX (escaped, booktabs) ────────────────────────────────────
from data_driven_plan import latex_table
for _tag, (_df_plan, _df_pairs) in ILLUS_TABLES.items():
    _Z = ILLUS[_tag][1].Z_complete
    latex_table(_df_plan, f"tables/table1_planning_{_tag}.tex",
                caption=f"Planning illustratif, durées {_tag} ($Z^\\\\star$ = {_Z:.2f}).",
                label=f"tab:planning-{_tag}", decimals=3)
    if len(_df_pairs):
        latex_table(_df_pairs, f"tables/table1_pairs_{_tag}.tex",
                    caption=f"Coefficients des paires doubles à l'optimum, durées {_tag}.",
                    label=f"tab:pairs-{_tag}", decimals=4)
print("Tables LaTeX exportées dans tables/")
""")

# ── Objectif réellement résolu (B5) ─────────────────────────────────────────
md("### Objectif résolu\n\n"
   "La cellule suivante écrit, à partir des coûts effectivement passés au solveur, "
   "l'objectif du PLNE en LaTeX (`tables/objective_solved.tex`). Avec `C_O_BASE = 0` "
   "on retrouve l'équation (9) du manuscrit ; sinon le terme d'overtime apparaît explicitement.")

code("""\
_obj_tex = objective_latex(c_I=1.0, c_W=0.5, c_O=C_O_BASE)
with open("tables/objective_solved.tex", "w", encoding="utf-8") as fh:
    fh.write("% Objectif du PLNE tel que résolu (c_I=1, c_W=0.5, c_O=C_O_BASE)\\n")
    fh.write("\\\\begin{equation}\\n" + _obj_tex + "\\n\\\\end{equation}\\n")
from IPython.display import Math, display
display(Math(_obj_tex))
print(_obj_tex)
""")

# ── Section 4 : Générateur + DOE ─────────────────────────────────────────────
md("## 4. Générateur d'instances et plan d'expériences\n\n"
   "Tailles, budgets et grilles viennent de `ilp_campaign.CampaignConfig` : production "
   "(une journée de S = 8 créneaux, R = 30 graines, gap cible 1e-6, limite de sécurité 600 s) "
   "ou smoke (S = 5, R = 2), modifiables par `CAMPAIGN_OVERRIDES`. Une journée est l'instance "
   "qu'un service planifie réellement ; elle est résolue à l'optimum prouvé en quelques secondes "
   "sur un ordinateur portable, alors que 5 jours × 8 créneaux (60 patients) gardaient un gap de "
   "11 à 12 % après 1 800 s, sans amélioration de la solution entre 600 et 1 800 s. Plan OFAT autour de la "
   "cellule centrale (n/S = 1,5 ; p̄ = 0,20 ; part de créneaux longs 0,15 ; c_W/c_I = 0,50). "
   "Chaque PLNE est résolu une fois (cache disque), démarré à chaud depuis le glouton B3, "
   "avec un fil HiGHS par processus et un processus par cœur physique, les instances les "
   "plus chargées en premier (LPT).")

code("""\
# ── 4.1 Générateur (ilp_campaign.generate_instance) ──────────────────────────
# Correction P2 : q ~ Beta de moyenne 1 - p̄ et de concentration fixe kappa = 6,111
# (écart-type 0,15 à la cellule centrale p̄ = 0,20, comme avant), sans le plancher
# max(., 0.5) qui déplaçait la moyenne (0,844 au lieu de 0,90 à p̄ = 0,10).
import ilp_campaign as campaign
from ilp_campaign import BETA_CONCENTRATION, Q_CLIP, check_beta_mean, load_holdout_scores

def generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15, cW_ratio=0.5,
                      c_I=1.0, c_O=C_O_BASE, s_A_=None, s_B_=None, pi_A_=None,
                      Q_bar=None, q_pool=None) -> Instance:
    return campaign.generate_instance(
        seed, days=CC.days, slots_per_day=CC.slots_per_day, slot_lengths=SLOT_LENGTHS,
        ratio=ratio, p_bar=p_bar, long_share=long_share, cW_ratio=cW_ratio,
        s_A=s_A if s_A_ is None else s_A_, s_B=s_B if s_B_ is None else s_B_,
        pi_A=pi_A if pi_A_ is None else pi_A_, c_O=c_O, c_I=c_I, Q_bar=Q_bar, q_pool=q_pool)
""")

code("""\
# ── 4.1b Contrôle de la loi de q à chaque niveau de p̄ (10^5 tirages) ────────
for _pb in (0.03, 0.10, 0.20, 0.30):
    _m = check_beta_mean(_pb, n=100_000, seed=RANDOM_STATE)
    print(f"p̄ = {_pb:.2f} : moyenne empirique de q = {_m:.4f} (attendu {1-_pb:.4f}, |écart| < 0,01)")
print(f"kappa = {BETA_CONCENTRATION}, q tronqué à {Q_CLIP}")

# Variante empirique : q_i calibrés du holdout (mêmes grilles et groupes, q rééchantillonnés)
if HOLDOUT_SCORES_PATH is None:
    raise ValueError("HOLDOUT_SCORES_PATH requis : holdout_scores.parquet exporté par data_driven_plan.ipynb")
HOLDOUT = load_holdout_scores(HOLDOUT_SCORES_PATH)
Q_POOL = HOLDOUT["q_i"].to_numpy()
print(f"Holdout : {len(HOLDOUT):,} patients, q̄ = {Q_POOL.mean():.4f}, "
      f"no-show observé = {HOLDOUT['no_show'].mean():.4f}")
# La calibration isotone est en escalier : ses paliers extrêmes peuvent valoir 0 ou 1.
# Les q_i sont stockés tels que prédits et tronqués à Q_CLIP à l'entrée d'une instance.
Q_OUTSIDE = int(((Q_POOL < Q_CLIP[0]) | (Q_POOL > Q_CLIP[1])).sum())
print(f"q_i hors de {Q_CLIP} (tronqués dans les instances) : {Q_OUTSIDE} sur {len(Q_POOL):,} "
      f"({100 * Q_OUTSIDE / len(Q_POOL):.4f} %)")
""")

code("""\
# ── 4.2 Configuration de la campagne ─────────────────────────────────────────
import dataclasses, functools, json
from ilp_campaign import (CampaignConfig, CampaignContext, doe_specs, run_lpt, solve_spec,
                          simulate_spec, spec_cost)

CC = (CampaignConfig.smoke if SMOKE else CampaignConfig.production)(n_workers=N_WORKERS)
CC = dataclasses.replace(CC, **CAMPAIGN_OVERRIDES)
# Les q_i du holdout voyagent vers chaque processus : on en garde au plus 100 000,
# tirés une fois pour toutes (la loi rééchantillonnée reste celle du holdout).
_rng_pool = np.random.default_rng(RANDOM_STATE)
Q_POOL_SAMPLE = Q_POOL if len(Q_POOL) <= 100_000 else _rng_pool.choice(Q_POOL, 100_000, replace=False)
CTX = CampaignContext(cc=CC, s_A=s_A, s_B=s_B, pi_A=pi_A, c_O=C_O_BASE,
                      slot_lengths=tuple(float(v) for v in SLOT_LENGTHS),
                      cache_root=CACHE_DIR, q_pool=Q_POOL_SAMPLE)
R, SEEDS = CC.R, CC.seeds
with open("results/ilp_campaign_config.json", "w", encoding="utf-8") as fh:
    json.dump({"campaign": CC.as_dict(), "decisions": DECISIONS, "plne_parameters": PLNE},
              fh, indent=2, default=str)
for _k, _v in CC.as_dict().items():
    print(f"{_k:<20} {_v}")
""")

code("""\
# ── 4.3 Exécution du plan d'expériences ──────────────────────────────────────
DOE_SPECS = doe_specs(CC, with_holdout_q=True)
_t0_doe = time.perf_counter()
doe_results = run_lpt(functools.partial(solve_spec, CTX), DOE_SPECS, spec_cost, CC.n_workers)
_t_doe = time.perf_counter() - _t0_doe
_df_doe = pd.DataFrame(doe_results)
_df_doe["occ"] = _df_doe["n_served"] / _df_doe["n"]
_df_doe["double_rate"] = _df_doe["n_double"] / CC.S
_df_doe.to_csv("results/doe_ilp.csv", index=False)
print(f"{len(DOE_SPECS)} PLNE en {_t_doe/60:.1f} min sur {CC.n_workers} processus")
_doe_summary = _df_doe.groupby("label")[["Z_complete", "E_idle", "E_wait", "E_OT", "n_served",
                                         "double_rate", "solve_time"]].mean()
print(_doe_summary.round(3))
latex_table(_doe_summary.reset_index(), "tables/table_doe.tex",
            caption=f"Plan d'expériences : moyennes par cellule (R = {R}).", label="tab:doe", decimals=3)
""")

code("""\
# ── 4.4 Statut de terminaison et gap (audit B1) ──────────────────────────────
print("Statuts HiGHS :", _df_doe["status"].value_counts().to_dict())
_unproven = _df_doe[_df_doe["status"] != "optimal"]
print(f"Optimalité prouvée : {100*(1-len(_unproven)/len(_df_doe)):.1f} % des PLNE")
print(f"Temps de résolution (s) : médiane {_df_doe['solve_time'].median():.1f}, "
      f"95e centile {_df_doe['solve_time'].quantile(0.95):.1f}, max {_df_doe['solve_time'].max():.1f}")
print(f"Mémoire résidente maximale d'un processus de résolution : {_df_doe['peak_rss_mb'].max():.0f} Mo")
if len(_unproven):
    print("Gap (coût complet) des solutions non prouvées : "
          f"médiane {_unproven['gap_complete'].median():.2e}, max {_unproven['gap_complete'].max():.2e}")
    print(_unproven.groupby("label").size().rename("non prouvés").to_string())
""")

code("""\
# ── 4.5 Figures du plan d'expériences (moyenne et IC à 95 %) ────────────────
from scipy import stats as sps

def _mean_ci(x):
    x = np.asarray(x, float)
    h = sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return x.mean(), h

_FACTORS = [("ratio", "Charge n/S", "ratio_", CC.central_ratio),
            ("p_bar", "Taux de no-show moyen p̄", "pbar_", CC.central_p_bar),
            ("long_share", "Part de créneaux longs", "ls_", CC.central_long_share),
            ("cW_ratio", "c_W / c_I", "cW_", CC.central_cw_ratio)]
_METRICS = [("Z_complete", "Coût espéré Z (min pondérées)"), ("E_idle", "E[Idle] pondéré (min)"),
            ("E_wait", "E[Wait] pondéré (min)"), ("E_OT", "E[OT] pondéré (min)")]
fig, axes = plt.subplots(len(_METRICS), len(_FACTORS), figsize=(14, 11))
for c, (fac, xlabel, prefix, center) in enumerate(_FACTORS):
    sub = _df_doe[(_df_doe["label"] == "central") | _df_doe["label"].str.startswith(prefix)]
    for r, (met, ylabel) in enumerate(_METRICS):
        g = sub.groupby(fac)[met].apply(_mean_ci)
        lv = np.array(g.index, float); mu = np.array([v[0] for v in g]); hw = np.array([v[1] for v in g])
        ax = axes[r, c]
        ax.errorbar(lv, mu, yerr=hw, fmt="o-", capsize=3)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/doe_metrics.pdf", bbox_inches="tight")
plt.savefig("figures/doe_metrics.png", dpi=300, bbox_inches="tight")
plt.show()
""")

# ── Section 5 : Benchmarks ───────────────────────────────────────────────────
md("## 5. Benchmarks B1 à B5\n\n"
   "Cellule centrale, R graines appariées. B1 réservation simple (même objectif, un patient "
   "par créneau) ; B2 approximation fluide (Jensen) ; B3 glouton ; B4 utilité de type "
   "Valenzuela (plafond de présence espérée 1,2 par créneau) ; B5 surréservation uniforme "
   "(rotation jusqu'à deux patients par créneau, sans regarder q ni s ; rec. 8). Chaque planning "
   "est évalué avec le coût exact du modèle, puis rejoué en simulation avec les mêmes tirages "
   "pour toutes les politiques (nombres aléatoires communs). Écart au PLNE : moyenne, IC à 95 % "
   "de Student, test des rangs signés de Wilcoxon apparié.")

code("""\
# ── 5.1 Exécution ────────────────────────────────────────────────────────────
from ilp_campaign import BENCHMARKS, bench_spec, cell_spec, paired_summary

BENCH_SPECS = [cell_spec(CC, "central", sd) for sd in SEEDS]
_df_bench = pd.DataFrame(run_lpt(functools.partial(bench_spec, CTX), BENCH_SPECS,
                                 spec_cost, CC.n_workers))
_df_bench.to_csv("results/benchmarks.csv", index=False)
print("Statut du PLNE par graine :", _df_bench["ILP_status"].value_counts().to_dict())
""")

code("""\
# ── 5.2 Comparaison appariée au PLNE ─────────────────────────────────────────
_rows = []
for _b, (_name, _) in BENCHMARKS.items():
    for _kind, _suffix in (("modèle exact", "_Z"), ("simulé, NAC", "_Zsim")):
        _ps = paired_summary(_df_bench[_b + _suffix], _df_bench["ILP" + _suffix])
        _rows.append({"Politique": f"{_b} {_name}", "Coût": _kind,
                      "ΔZ moyen": _ps["mean_diff"], "IC 95 % bas": _ps["ci_low"],
                      "IC 95 % haut": _ps["ci_high"], "Écart moyen (%)": _ps["rel_pct"],
                      "Bat le PLNE": f"{_ps['n_better']}/{_ps['n']}", "p (Wilcoxon)": _ps["p_wilcoxon"],
                      "Doubles moyens": _df_bench[f"{_b}_double"].mean()})
_df_bench_cmp = pd.DataFrame(_rows)
print(_df_bench_cmp.round(3).to_string(index=False))
print(f"PLNE : doubles moyens {_df_bench['ILP_double'].mean():.1f}, coût exact moyen {_df_bench['ILP_Z'].mean():.2f}")
latex_table(_df_bench_cmp, "tables/table_benchmarks.tex",
            caption=f"Benchmarks contre le PLNE, cellule centrale, R = {R} graines appariées.",
            label="tab:benchmarks", decimals=3)
_beaten = _df_bench_cmp[(_df_bench_cmp["Coût"] == "modèle exact") & (_df_bench_cmp["Bat le PLNE"].str[0] != "0")]
if len(_beaten):
    print("ATTENTION : une politique bat le PLNE sous son propre coût, donc au moins une "
          "solution du PLNE n'est pas optimale (voir les statuts ci-dessus).")
""")

# ── Section 6 : Simulation Lindley ───────────────────────────────────────────
md("## 6. Simulation hors hypothèse d'indépendance, dynamique séquentielle (Lindley)\n\n"
   "Chaque planning optimal est rejoué sur une journée réelle : retard propagé d'un créneau "
   "au suivant, idle terminal compté (correction P1). M scénarios par planning (`CC.M_sim`).")

code("""\
# ── 6.1 Simulateur Lindley (data_driven_plan.simulate_day / simulate_instance) ─
# Correction P1 : le simulateur compte désormais l'idle terminal, entre la fin du
# dernier patient et la fin de la journée ; sans lui, une journée qui finit tôt
# paraissait gratuite. simulate_instance renvoie le coût de chacun des M scénarios
# (moyenne, IC et tests appariés possibles) ; eval_day_exact donne la même
# espérance exactement, par énumération des 2^n profils de présence d'une journée.
print("simulate_day, simulate_instance, eval_day_exact importés de data_driven_plan")
""")

code("""\
# ── 6.2 Validation sur l'exemple illustratif ─────────────────────────────────
# Trois évaluations du même planning optimal :
#  - modèle      : créneaux indépendants, ce que le PLNE optimise (eval_slot_exact) ;
#  - jour exact  : dynamique de Lindley avec report et idle terminal, énumération 2^n ;
#  - simulé      : même dynamique, M scénarios ; doit retrouver le jour exact à 4 erreurs-types près.
_rng6 = np.random.default_rng(RANDOM_STATE)
_M_ILLUS = 10_000 if SMOKE else 200_000
_model_Z3 = evaluate_planning_exact(_inst3, _sol3.planning, day_dynamics=False)["Z"]
_exact_Z3 = evaluate_planning_exact(_inst3, _sol3.planning, day_dynamics=True)["Z"]
_sim3 = simulate_instance(_inst3, _sol3.planning, _M_ILLUS, _rng6, cv=0.0)
_se3 = _sim3.std(ddof=1) / np.sqrt(_M_ILLUS)
print(f"Exemple illustratif (M={_M_ILLUS:,}) :")
print(f"  Coût modèle (créneaux indépendants) : {_model_Z3:.4f}")
print(f"  Coût exact de la journée (Lindley)   : {_exact_Z3:.4f}")
print(f"  Coût simulé                          : {_sim3.mean():.4f} ± {1.96*_se3:.4f} (IC 95 %)")
assert abs(_sim3.mean() - _exact_Z3) < 4 * _se3, "simulateur et évaluation exacte divergent"
print(f"  Écart jour/modèle : {100*(_exact_Z3/_model_Z3-1):+.2f} % (erreur de l'hypothèse d'indépendance A6)")
""")

code("""\
# ── 6.3 Écart modèle / simulé sur le plan d'expériences ─────────────────────
# Les plannings viennent du cache de la section 4 (aucune nouvelle résolution).
# Durées déterministes (cv = 0) puis lognormales (cv = 0,3 et 0,5), même moyenne.
M_DOE_SIM = CC.M_sim
_t0_sim = time.perf_counter()
sim_results = run_lpt(functools.partial(simulate_spec, CTX), DOE_SPECS, spec_cost, CC.n_workers)
print(f"Simulation de {len(DOE_SPECS)} plannings (M={M_DOE_SIM:,}) en {(time.perf_counter()-_t0_sim)/60:.1f} min")
_df_sim = pd.DataFrame(sim_results)
_df_sim.to_csv("results/simulation_doe.csv", index=False)
print("Écart % simulé (cv = 0) / modèle, par cellule :")
print(_df_sim.groupby("label")["gap_pct"].agg(["mean", "std"]).round(2))
""")

code("""\
# ── 6.4 Figures : écart % selon la charge et la part de doubles ─────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
_sub = _df_sim[(_df_sim["label"] == "central") | _df_sim["label"].str.startswith("ratio_")]
g = _sub.groupby("ratio")["gap_pct"].apply(_mean_ci)
ax1.errorbar(g.index, [v[0] for v in g], yerr=[v[1] for v in g], fmt="o-", capsize=3)
ax1.set_xlabel("Charge n/S"); ax1.set_ylabel("Écart coût simulé / modèle (%)")
ax1.grid(True, alpha=0.3)
ax2.scatter(_df_sim["double_rate"], _df_sim["gap_pct"], alpha=0.4, s=15)
ax2.set_xlabel("Part de créneaux doublés"); ax2.set_ylabel("Écart coût simulé / modèle (%)")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/sim_gap_analysis.pdf", bbox_inches="tight")
plt.savefig("figures/sim_gap_analysis.png", dpi=300, bbox_inches="tight")
plt.show()
""")

code("""\
# ── 6.5 Variabilité des durées (lognormale, cv = 0,3 et 0,5) ────────────────
fig, ax = plt.subplots(figsize=(6.5, 4))
for cv in (0.0, 0.3, 0.5):
    col = f"Z_sim_cv{cv:g}"
    dev = 100 * (_sub[col] / _sub["Z_model"] - 1)
    g = dev.groupby(_sub["ratio"]).apply(_mean_ci)
    ax.errorbar(g.index, [v[0] for v in g], yerr=[v[1] for v in g], fmt="o-", capsize=3,
                label=f"cv = {cv:g}")
ax.set_xlabel("Charge n/S"); ax.set_ylabel("Écart coût simulé / modèle (%)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/sim_lognormal.pdf", bbox_inches="tight")
plt.savefig("figures/sim_lognormal.png", dpi=300, bbox_inches="tight")
plt.show()
""")

# ── Section 7 : Sensibilité ──────────────────────────────────────────────────
md("## 7. Analyse de sensibilité et robustesse\n\n"
   "Correction P3. Pour chaque perturbation, le PLNE est résolu sur l'instance perturbée "
   "(ce que verrait l'optimiseur), puis son planning est évalué sur l'instance vraie et comparé "
   "à l'optimum vrai (même cellule, même graine, solution du cache). Évaluation **exacte** "
   "(coût du modèle) et **simulée** avec des nombres aléatoires communs aux deux plannings. "
   "Les perturbations d'un même type partagent leurs tirages d'un niveau à l'autre. "
   "La mauvaise classification va jusqu'au taux d'erreur mesuré du classifieur de groupe. "
   "Aucune conclusion comparative n'est tirée sans test apparié significatif (Wilcoxon, 5 %).")

code("""\
# ── 7.1 Regret sous perturbation des entrées ─────────────────────────────────
from ilp_campaign import sensitivity_spec, sensitivity_specs, sensitivity_table

SENS_SPECS = sensitivity_specs(CC, PLNE["group_error_rate"])
_df_sens_raw = pd.DataFrame(run_lpt(functools.partial(sensitivity_spec, CTX), SENS_SPECS,
                                    spec_cost, CC.n_workers))
_df_sens_raw.to_csv("results/sensitivity_raw.csv", index=False)
_df_sens = sensitivity_table(_df_sens_raw)
_df_sens.to_csv("results/sensitivity_summary.csv", index=False)
print(_df_sens.round(4).to_string(index=False))
""")

code("""\
# ── 7.2 Figure : regret exact par perturbation (moyenne et IC à 95 %) ───────
_KIND_LABEL = {"q_shift": "Décalage de q", "q_noise": "Bruit gaussien sur q",
               "duration": "Erreur relative sur s", "misclass": "Taux de mauvaise classification"}
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, (kind, lab) in zip(axes, _KIND_LABEL.items()):
    sub = _df_sens_raw[_df_sens_raw["kind"] == kind]
    g = sub.groupby("level")["deg_pct"].apply(_mean_ci)
    ax.errorbar(g.index, [v[0] for v in g], yerr=[v[1] for v in g], fmt="o-", capsize=3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel(lab); ax.set_ylabel("Regret exact (% du coût optimal)"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/sensitivity_regret.pdf", bbox_inches="tight")
plt.savefig("figures/sensitivity_regret.png", dpi=300, bbox_inches="tight")
plt.show()
""")

code("""\
# ── 7.3 Plafond de présence Q̄ et part du groupe A (10 premières graines) ────
Q_BARS = [1.0, 1.2, 1.5, None]
PI_AS = sorted({0.05, 0.15, 0.30, round(pi_A, 4)})
QBAR_SPECS = [cell_spec(CC, "qbar", sd, Q_bar=qb, pi_A=pa)
              for qb in Q_BARS for pa in PI_AS for sd in SEEDS[:10]]
_df_qbar = pd.DataFrame(run_lpt(functools.partial(solve_spec, CTX), QBAR_SPECS, spec_cost, CC.n_workers))
_df_qbar["Q_bar"] = _df_qbar["Q_bar"].map(lambda v: "aucun" if pd.isna(v) else f"{v:g}")
_df_qbar.to_csv("results/sensitivity_qbar.csv", index=False)
print("Statuts :", _df_qbar["status"].value_counts().to_dict())
print(_df_qbar.pivot_table(values="Z_complete", index="Q_bar", columns="pi_A", aggfunc="mean").round(3))
""")

code("""\
# ── 7.4 Front idle / attente (balayage de c_W/c_I, 10 premières graines) ────
CW_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
PARETO_SPECS = [cell_spec(CC, "pareto", sd, cW_ratio=v) for v in CW_SWEEP for sd in SEEDS[:10]]
_df_pareto = pd.DataFrame(run_lpt(functools.partial(solve_spec, CTX), PARETO_SPECS, spec_cost, CC.n_workers))
# E_wait est pondéré par c_W : on revient aux minutes d'attente pour le front
_df_pareto["wait_min"] = _df_pareto["E_wait"] / _df_pareto["cW_ratio"]
_df_pareto.to_csv("results/pareto_idle_wait.csv", index=False)
grp_p = _df_pareto.groupby("cW_ratio")[["E_idle", "wait_min"]].mean()
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(grp_p["E_idle"], grp_p["wait_min"], c=np.log(grp_p.index), cmap="viridis", s=60, zorder=5)
for cwr, row in grp_p.iterrows():
    ax.annotate(f"{cwr:g}", (row["E_idle"], row["wait_min"]), fontsize=7,
                textcoords="offset points", xytext=(4, 4))
ax.plot(grp_p["E_idle"], grp_p["wait_min"], "k--", alpha=0.4)
ax.set_xlabel("Inactivité espérée E[Idle] (min)"); ax.set_ylabel("Attente espérée E[Wait] (min)")
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label="log(c_W / c_I)")
plt.tight_layout()
plt.savefig("figures/pareto_idle_wait.pdf", bbox_inches="tight")
plt.savefig("figures/pareto_idle_wait.png", dpi=300, bbox_inches="tight")
plt.show()
""")

code("""\
# ── 7.5 Synthèse : seules les différences significatives sont qualifiées ────
_sig = _df_sens[_df_sens["Significatif (5 %)"]]
_nsig = _df_sens[~_df_sens["Significatif (5 %)"]]
print(f"{len(_sig)} perturbation(s) sur {len(_df_sens)} dégradent significativement le coût (Wilcoxon, 5 %).")
for _, r in _sig.iterrows():
    print(f"  {r['Perturbation']} {r['Niveau']:+g} : regret {r['Regret exact (%)']:.2f} %, "
          f"IC 95 % de ΔZ [{r['IC 95 % bas']:.2f} ; {r['IC 95 % haut']:.2f}], p = {r['p exact']:.3g}")
if len(_nsig):
    print("Non significatif (aucune conclusion) : " +
          ", ".join(f"{r['Perturbation']} {r['Niveau']:+g}" for _, r in _nsig.iterrows()))
latex_table(_df_sens, "tables/table_sensitivity.tex",
            caption="Regret sous perturbation des entrées (IC à 95 %, tests de Wilcoxon appariés).",
            label="tab:sensitivity", decimals=4)
SENS_VERDICT = (f"{len(_sig)}/{len(_df_sens)} perturbations significatives ; regret exact maximal "
                f"{_df_sens['Regret exact (%)'].max():.2f} %")
""")

# ── Section 7.6 : seuil de double-booking ───────────────────────────────────
md("### 7.6 Seuil de double-booking (rec. 11)\n\n"
   "Placer j derrière i dans un créneau (sans saturation, durées égales) échange $c_I q_j s$ "
   "d'inactivité contre $c_W q_i q_j s$ d'attente : le double est rentable si "
   "$c_W/c_I < 1/q_i$, où $i$ est le patient servi en premier. Le seuil est **individuel** ; "
   "avec des $q_i$ dispersés, on attend une bande de transition centrée sur $1/\\bar q$ et non "
   "une bascule nette. Grille fine de $c_W/c_I$ de 1,0 à 1,5 par pas de 0,05, R graines.")

code("""\
# ── 7.6 Balayage fin de c_W/c_I ─────────────────────────────────────────────
from ilp_campaign import threshold_spec, threshold_specs
_df_thr = pd.DataFrame(run_lpt(functools.partial(threshold_spec, CTX), threshold_specs(CC),
                               spec_cost, CC.n_workers))
_df_thr.to_csv("results/threshold_cw_fine.csv", index=False)
_thr = _df_thr.groupby("cW_ratio").agg(instances_avec_double=("has_double", "mean"),
                                       doubles_moyens=("n_double", "mean"),
                                       patients_sous_seuil=("share_below_threshold", "mean"),
                                       doubles_conformes=("doubles_rule_ok", "mean"))
print("Statuts :", _df_thr["status"].value_counts().to_dict())
print(_thr.round(3).to_string())
latex_table(_thr.reset_index(), "tables/table_threshold_cw.tex",
            caption="Seuil de double-booking : grille fine de $c_W/c_I$.", label="tab:threshold", decimals=3)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(_thr.index, 100 * _thr["instances_avec_double"], "o-", label="Instances avec au moins un double")
ax.plot(_thr.index, 100 * _thr["patients_sous_seuil"], "s--", label="Patients avec c_W/c_I < 1/q_i")
ax.axvline(1 / (1 - CC.central_p_bar), color="grey", lw=0.8, ls=":", label="1 / q̄")
ax.set_xlabel("c_W / c_I"); ax.set_ylabel("Part (%)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/threshold_cw_fine.pdf", bbox_inches="tight")
plt.savefig("figures/threshold_cw_fine.png", dpi=300, bbox_inches="tight")
plt.show()
""")

# ── Section 8 : validation hors échantillon sur le holdout ──────────────────
md("## 8. Validation hors échantillon sur des patients du holdout (rec. 10)\n\n"
   "Journées construites à partir de patients du holdout, jamais vus à l'apprentissage : "
   "le PLNE planifie avec les $q_i$ et les groupes **prédits** ; chaque planning est ensuite jugé "
   "sur la vérité : groupes **réels** (durées), coût espéré du modèle, coût espéré exact de la "
   "journée (Lindley, idle terminal) et coût **réalisé** compte tenu des présences observées. "
   "Références : PLNE oracle (groupes réels), B1, B3, B5. Comparaisons appariées par journée.")

code("""\
# ── 8.1 Journées du holdout ──────────────────────────────────────────────────
from ilp_campaign import holdout_day, holdout_day_specs
HOLDOUT_SPECS = holdout_day_specs(CC, HOLDOUT, seed=RANDOM_STATE)
_df_hold = pd.DataFrame(run_lpt(functools.partial(holdout_day, CTX), HOLDOUT_SPECS,
                                lambda sp: 1.0, CC.n_workers))
_df_hold.to_csv("results/holdout_days.csv", index=False)
print(f"{len(_df_hold)} journées de {_df_hold['n'].iloc[0]} patients ; statut du PLNE :",
      _df_hold["ILP_status"].value_counts().to_dict())
_rows = []
for _ref in ("ILP_oracle", "B1", "B3", "B5"):
    for _kind, _suffix in (("espéré, modèle", "_Zmodel"), ("espéré, journée", "_Zday"), ("réalisé", "_Zreal")):
        _ps = paired_summary(_df_hold[_ref + _suffix], _df_hold["ILP" + _suffix])
        _rows.append({"Politique": _ref, "Coût": _kind, "ΔZ moyen vs PLNE": _ps["mean_diff"],
                      "IC 95 % bas": _ps["ci_low"], "IC 95 % haut": _ps["ci_high"],
                      "Écart moyen (%)": _ps["rel_pct"], "Fait mieux": f"{_ps['n_better']}/{_ps['n']}",
                      "p (Wilcoxon)": _ps["p_wilcoxon"]})
_df_hold_cmp = pd.DataFrame(_rows)
_df_hold_cmp.to_csv("results/holdout_days_summary.csv", index=False)
latex_table(_df_hold_cmp, "tables/table_holdout_days.tex",
            caption="Journées du holdout : écarts appariés au PLNE (IC à 95 %, Wilcoxon).",
            label="tab:holdout", decimals=3)
print(_df_hold_cmp.round(3).to_string(index=False))
""")

# ── RESULTS.md ────────────────────────────────────────────────────────────────
code("""\
# ── 9. Export RESULTS.md ─────────────────────────────────────────────────────
_central = _df_doe[_df_doe.label == "central"]
_sol_hist = ILLUS["historique"][1]
def _cmp_lines(df, kind):
    return "\\n".join(
        f"- {r['Politique']} : {r.iloc[2]:+.2f} [{r['IC 95 % bas']:.2f} ; {r['IC 95 % haut']:.2f}], "
        f"p = {r['p (Wilcoxon)']:.3g}" for _, r in df[df["Coût"] == kind].iterrows())

_md = f\"\"\"# RESULTS.md, chiffres clés pour le manuscrit

Paramètres : {PLNE['source_run']} (commit {PLNE['git_commit']}), s_A = {s_A:.2f}, s_B = {s_B:.2f},
pi_A = {pi_A:.4f}, c_O = {C_O_BASE}. Campagne : S = {CC.S}, R = {CC.R}, limite {CC.time_limit:g} s,
gap cible {CC.mip_gap:g}. Smoke : {SMOKE}.
Holdout : {len(Q_POOL):,} patients, {Q_OUTSIDE} q_i hors de {Q_CLIP}, tronqués dans les instances.

## Table 1 (section 3)
- Historique (s_A = 43, s_B = 8) : Z* = {_sol_hist.Z_complete:.4f} ({_sol_hist.status})
- Données : Z* = {ILLUS['donnees'][1].Z_complete:.4f}

## Plan d'expériences
- Optimalité prouvée : {100*(_df_doe['status'] == 'optimal').mean():.1f} % des PLNE
- Résolution : médiane {_df_doe['solve_time'].median():.1f} s, max {_df_doe['solve_time'].max():.1f} s,
  mémoire max {_df_doe['peak_rss_mb'].max():.0f} Mo par processus ({CPU_MODEL},
  {psutil.virtual_memory().total / 2**30:.0f} Go de RAM)
- Cellule centrale : Z = {_central['Z_complete'].mean():.3f} ± {_central['Z_complete'].std():.3f}

## Benchmarks (coût exact du modèle, Delta Z vs PLNE, IC 95 %)
{_cmp_lines(_df_bench_cmp.rename(columns={'ΔZ moyen': 'd'}), 'modèle exact')}

## Simulation (M = {M_DOE_SIM})
- Écart simulé / modèle, cellule centrale : {_df_sim[_df_sim.label == 'central']['gap_pct'].mean():.2f} %

## Robustesse
- {SENS_VERDICT}

## Hors échantillon (holdout, coût réalisé, Delta Z vs PLNE)
{_cmp_lines(_df_hold_cmp, 'réalisé')}
\"\"\"
with open("results/RESULTS.md", "w", encoding="utf-8") as fh:
    fh.write(_md)
print(_md)
""")

# ── Build notebook JSON ───────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = "01_ilp_validation.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook écrit : {out}  ({len(cells)} cellules)")
