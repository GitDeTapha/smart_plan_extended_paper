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
C_O_BASE        = 0.5           # V9 : overtime conservé ; c_O = 0 en ablation
SLOT_LENGTHS    = (30, 60)      # B4 : assertion s_A <= max(SLOT_LENGTHS), sinon arrêt
SPLIT_MODE      = "stratified"  # "patient_group" en ablation
RANDOM_STATE    = 0
SMOKE           = False         # True : 1 % des données, budgets minimaux
PLNE_PARAMS_PATH = "data/inference/plne_parameters.json"  # exporté par data_driven_plan.ipynb
""")

code("""\
# ── 0. Environnement ──────────────────────────────────────────────────────────
import sys, platform, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
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
print(f"Platform: {platform.processor()}")
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
   "Correction rec. 5 : l'exemple ne dépend plus des données. La **table historique** fixe "
   "$s_A=43$, $s_B=8$ min et doit redonner $Z^\\star = 70{,}82$ (assertion) ; la **table données** "
   "reprend $s_A$, $s_B$ de `plne_parameters.json`, à titre d'illustration.")

code("""\
# ── 3.1 Construction et résolution des deux variantes ───────────────────────
ILLUS_Q = np.array([0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45])
ILLUS_L = [[30.0, 30.0, 60.0, 60.0]]
ILLUS_GROUPS = np.array(["A"] * 3 + ["B"] * 7)

def illustrative_instance(sA, sB):
    return Instance(n=10, D=1, T=[4], L=ILLUS_L, s=np.where(ILLUS_GROUPS == "A", sA, sB),
                    q=ILLUS_Q, c_I=1.0, c_W=0.5, c_O=0.5)

ILLUS = {}
for _tag, (_sA, _sB) in {"historique": (43.0, 8.0), "donnees": (s_A, s_B)}.items():
    _inst = illustrative_instance(_sA, _sB)
    _sol = solve_ilp(_inst, Config(require_optimal=True))
    ILLUS[_tag] = (_inst, _sol)
    print(f"[{_tag}] s_A={_sA:.2f}, s_B={_sB:.2f} : statut {_sol.status}, gap {_sol.gap_complete:.1e}, "
          f"Z* = {_sol.Z_complete:.4f}  (E[Idle]={_sol.E_idle:.3f}, E[Wait]={_sol.E_wait:.3f}, "
          f"E[OT]={_sol.E_OT:.3f}), servis {_sol.n_served}/10, doubles {_sol.n_double}/4")

_Z_hist = ILLUS["historique"][1].Z_complete
assert abs(_Z_hist - 70.82) < 5e-3, f"Table 1 historique : Z* = {_Z_hist:.4f}, attendu 70,82"
print("Table 1 historique : Z* = 70,82 retrouvé.")
# Les sections 3.2 à 6.2 utilisent la variante historique, stable d'un run à l'autre.
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
# ── 3.4 Export LaTeX ─────────────────────────────────────────────────────────
for _tag, (_df_plan, _df_pairs) in ILLUS_TABLES.items():
    _Z = ILLUS[_tag][1].Z_complete
    _df_plan.to_latex(f"tables/table1_planning_{_tag}.tex", index=False, float_format="%.3f",
                      caption=f"Planning illustratif, durées {_tag} ($Z^\\\\star$ = {_Z:.2f}).",
                      label=f"tab:planning-{_tag}")
    if len(_df_pairs):
        _df_pairs.to_latex(f"tables/table1_pairs_{_tag}.tex", index=False, float_format="%.4f",
                           caption=f"Coefficients des paires doubles à l'optimum, durées {_tag}.",
                           label=f"tab:pairs-{_tag}")
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
   "**⚠ Simplification computationnelle :** S réduit à 20 créneaux (4 jours × 5) "
   "et R=10 réplications (vs S=40, R=30 dans le plan original) pour respecter le budget 45 min. "
   "Plan OFAT autour de la cellule centrale "
   "(ratio=1.5, p̄=0.20, long_share=0.15, c_W/c_I=0.50).")

code("""\
# ── 4.1 Générateur ────────────────────────────────────────────────────────────
S_TOTAL, D_DOE, SLOTS_PER_DAY = 20, 4, 5  # ⚠ réduit (40,5,8)→(20,4,5) budget 45 min

def generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15,
                       cW_ratio=0.5, c_I=1.0, c_O=C_O_BASE,
                       s_A_=None, s_B_=None, pi_A_=None,
                       Q_bar=None) -> Instance:
    \"\"\"
    ratio  : n/S
    p_bar  : taux de no-show moyen
    long_share : fraction de créneaux longs
    cW_ratio   : c_W / c_I
    \"\"\"
    rng = np.random.default_rng(seed)
    _sA = s_A_ if s_A_ is not None else s_A
    _sB = s_B_ if s_B_ is not None else s_B
    _piA = pi_A_ if pi_A_ is not None else pi_A

    n = max(1, round(ratio * S_TOTAL))
    c_W = cW_ratio * c_I

    # Grille de créneaux : D_DOE jours × SLOTS_PER_DAY
    T = [SLOTS_PER_DAY] * D_DOE
    n_long = round(long_share * S_TOTAL)
    slot_types = rng.permutation(
        np.array([L_LONG]*n_long + [L_SHORT]*(S_TOTAL-n_long))
    ).reshape(D_DOE, SLOTS_PER_DAY)
    L = [slot_types[d].tolist() for d in range(D_DOE)]

    # Patients
    groups = rng.random(n) < _piA
    s = np.where(groups, _sA, _sB)

    # q ~ Beta(α, β) de moyenne (1-p_bar) et écart-type 0.15
    mu_q = 1.0 - p_bar
    var_q = 0.15**2
    _a = mu_q * (mu_q*(1-mu_q)/var_q - 1)
    _b = (1-mu_q) * (mu_q*(1-mu_q)/var_q - 1)
    _a, _b = max(_a, 0.5), max(_b, 0.5)
    q = beta_dist.rvs(_a, _b, size=n, random_state=int(seed) % (2**31))
    q = np.clip(q, 0.05, 0.99)

    return Instance(n=n, D=D_DOE, T=T, L=L, s=s, q=q,
                    c_I=c_I, c_W=c_W, c_O=c_O, Q_bar=Q_bar)
""")

code("""\
# ── 4.2 Plan d'expériences OFAT ───────────────────────────────────────────────
R = 10  # ⚠ réduit de 30 → 10 pour budget 45 min
SEEDS = [1000 + r for r in range(R)]

DOE_CELLS = []
# Cellule centrale
for r in range(R):
    DOE_CELLS.append(dict(ratio=1.5, p_bar=0.20, long_share=0.15,
                          cW_ratio=0.50, seed=SEEDS[r], label="central"))
# Variations ratio
for ratio in [1.0, 1.25, 2.0]:
    for r in range(R):
        DOE_CELLS.append(dict(ratio=ratio, p_bar=0.20, long_share=0.15,
                              cW_ratio=0.50, seed=SEEDS[r], label=f"ratio_{ratio}"))
# Variations p_bar
for p_bar in [0.10, 0.30]:
    for r in range(R):
        DOE_CELLS.append(dict(ratio=1.5, p_bar=p_bar, long_share=0.15,
                              cW_ratio=0.50, seed=SEEDS[r], label=f"pbar_{p_bar}"))
# Variations long_share
for ls in [0.25, 0.50]:
    for r in range(R):
        DOE_CELLS.append(dict(ratio=1.5, p_bar=0.20, long_share=ls,
                              cW_ratio=0.50, seed=SEEDS[r], label=f"ls_{ls}"))
# Variations c_W/c_I
for cwr in [0.1, 0.25, 1.0, 2.0]:
    for r in range(R):
        DOE_CELLS.append(dict(ratio=1.5, p_bar=0.20, long_share=0.15,
                              cW_ratio=cwr, seed=SEEDS[r], label=f"cW_{cwr}"))

print(f"Plan DOE : {len(DOE_CELLS)} cellules × 1 solve")
print(f"Facteurs : ratio×{len([c for c in DOE_CELLS if 'ratio' in c['label'] or c['label']=='central'])} ...")
""")

code("""\
# ── 4.3 Exécution du DOE (joblib parallel) ───────────────────────────────────
# ⚠ Simplification computationnelle : S_TOTAL=20, R=10 (vs 40/30 dans le plan).
# Réduit pour respecter le budget 45 min (DOE S=40 R=30 dépassait 3600s).

def _solve_cell(cell):
    inst = generate_instance(**{k:v for k,v in cell.items() if k not in ("label",)})
    sol  = solve_ilp(inst, Config(time_limit=60.0))
    occ  = sol.n_served / (inst.n + 1e-9)
    double_rate = sol.n_double / (sum(inst.T) + 1e-9)
    return {
        "label": cell["label"], "seed": cell["seed"],
        "ratio": cell["ratio"], "p_bar": cell["p_bar"],
        "long_share": cell["long_share"], "cW_ratio": cell["cW_ratio"],
        "n": inst.n, "Z_reduced": sol.Z_reduced, "Z_complete": sol.Z_complete,
        "E_idle": sol.E_idle, "E_wait": sol.E_wait, "E_OT": sol.E_OT,
        "n_served": sol.n_served, "n_double": sol.n_double,
        "occ": occ, "double_rate": double_rate,
        "solve_time": sol.solve_time, "gap": sol.gap,
    }

_t0_doe = time.perf_counter()
doe_results = joblib.Parallel(n_jobs=-1, prefer="processes")(
    joblib.delayed(_solve_cell)(c) for c in DOE_CELLS
)
_t_doe = time.perf_counter() - _t0_doe
print(f"DOE terminé en {_t_doe/60:.1f} min ({_t_doe:.0f} s)")
if _t_doe > 2400:
    print("⚠ Budget dépassé, envisager R=10 pour les expériences ultérieures.")

_df_doe = pd.DataFrame(doe_results)
_df_doe.to_csv("results/doe_ilp.csv", index=False)
print(_df_doe.groupby("label")[["Z_complete","E_idle","E_wait","E_OT",
                                 "n_served","double_rate","solve_time"]].mean().round(3))
""")

code("""\
# ── 4.4 Figures DOE ──────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
metrics = ["Z_complete", "E_idle", "E_wait", "E_OT"]
titles  = ["Coût total Z", "E[Idle]", "E[Wait]", "E[OT]"]
factors = ["ratio", "p_bar", "long_share", "cW_ratio"]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

for ax, metric, title in zip(axes.flat, metrics, titles):
    for fac, col in zip(factors, colors):
        grp = _df_doe.groupby(fac)[metric].mean()
        ax.plot(grp.index, grp.values, "o-", label=fac, color=col, alpha=0.8)
    ax.set_title(title); ax.set_xlabel("Niveau"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/doe_metrics.pdf", bbox_inches="tight")
plt.savefig("figures/doe_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure doe_metrics sauvegardée.")
""")

# ── Section 5 : Benchmarks ───────────────────────────────────────────────────
md("## 5. Benchmarks B1–B4\n\n"
   "Mêmes instances et seeds. Tout est ré-évalué sous le coût exact du modèle.")

code("""\
# ── 5.1 B1 : Single booking (C2 : Σ_i x ≤ 1) ─────────────────────────────────
def solve_b1_single(inst: Instance, cfg: Config = None) -> Solution:
    \"\"\"Single booking : même PLNE mais C2 ≤ 1.\"\"\"
    if cfg is None: cfg = Config()
    coeffs = compute_coefficients(inst)
    a, delta, w, o, gamma = (coeffs[k] for k in ("a","delta","w","o","gamma"))
    slots, pairs = coeffs["slots"], coeffs["pairs"]
    I = list(range(inst.n))
    x_idx = [(i,d,t) for i in I for (d,t) in slots]

    m = pyo.ConcreteModel()
    m.x = pyo.Var(x_idx, domain=pyo.Binary, initialize=0)

    def _obj(m):
        return sum((inst.c_O*o[i,d,t] - inst.c_I*a[i,d,t])*m.x[i,d,t]
                   for (i,d,t) in x_idx)
    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)
    m.C1 = pyo.ConstraintList()
    for i in I: m.C1.add(sum(m.x[i,d,t] for (d,t) in slots) <= 1)
    m.C2 = pyo.ConstraintList()
    for (d,t) in slots: m.C2.add(sum(m.x[i,d,t] for i in I) <= 1)  # ≤ 1

    solver = pyo.SolverFactory(SOLVER_NAME)
    solver.options["time_limit"] = cfg.time_limit
    t0 = time.perf_counter()
    solver.solve(m, tee=False)
    solve_time = time.perf_counter() - t0

    x_val = {(i,d,t): pyo.value(m.x[i,d,t]) for (i,d,t) in x_idx}
    planning = {}
    for (d,t) in slots:
        pats = [i for i in I if (x_val.get((i,d,t),0) or 0) > 0.5]
        if pats: planning[d,t] = sorted(pats, key=lambda i: inst.s[i])

    # Évaluer sous coût exact
    E_idle=E_wait=E_OT=0.0
    for (d,t), pats in planning.items():
        L=inst.L[d][t]
        ei,ew,eo = eval_slot_exact(pats, inst.q, inst.s, L)
        E_idle+=inst.c_I*ei; E_wait+=inst.c_W*ew; E_OT+=inst.c_O*eo
    # Créneaux vides
    for (d,t) in slots:
        if (d,t) not in planning: E_idle += inst.c_I*inst.L[d][t]

    Z_reduced = pyo.value(m.obj)
    Z_complete = Z_reduced + inst.c_I*coeffs["total_L"]
    n_served = sum(len(v) for v in planning.values())
    return Solution(planning=planning, E_idle=E_idle, E_wait=E_wait, E_OT=E_OT,
                    Z_reduced=Z_reduced, Z_complete=Z_complete,
                    solve_time=solve_time, gap=0.0,
                    n_served=n_served, n_double=0,
                    x_val=x_val, z_val={}, coeffs=coeffs)
""")

code("""\
# ── 5.2 B2 : Approximation fluide (Jensen) ────────────────────────────────────
# Simplification signalée : la linéarisation fluide remplace E[idle] = (L - Σq_i*s_i*x)^+
# par u ≥ L - Σq_i*s_i*x, u ≥ 0 (relaxation convexe exacte pour variables continues).
def solve_b2_fluid(inst: Instance, cfg: Config = None) -> Solution:
    if cfg is None: cfg = Config()
    coeffs = compute_coefficients(inst)
    a, o = coeffs["a"], coeffs["o"]
    slots = coeffs["slots"]
    I = list(range(inst.n))
    x_idx = [(i,d,t) for i in I for (d,t) in slots]
    u_idx = list(slots)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(x_idx, domain=pyo.Binary, initialize=0)
    m.u = pyo.Var(u_idx, bounds=(0, None), initialize=0)

    def _obj(m):
        idle_part = inst.c_I * sum(m.u[d,t] for (d,t) in slots)
        ot_part   = sum(inst.c_O*o[i,d,t]*m.x[i,d,t] for (i,d,t) in x_idx)
        return idle_part + ot_part
    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)
    m.C1 = pyo.ConstraintList()
    for i in I: m.C1.add(sum(m.x[i,d,t] for (d,t) in slots) <= 1)
    m.C2 = pyo.ConstraintList()
    for (d,t) in slots: m.C2.add(sum(m.x[i,d,t] for i in I) <= 2)
    # u ≥ L - Σ q_i*s_i*x
    m.Cf = pyo.ConstraintList()
    for (d,t) in slots:
        L = inst.L[d][t]
        m.Cf.add(m.u[d,t] >= L - sum(inst.q[i]*inst.s[i]*m.x[i,d,t] for i in I))

    solver = pyo.SolverFactory(SOLVER_NAME)
    solver.options["time_limit"] = cfg.time_limit
    t0 = time.perf_counter()
    solver.solve(m, tee=False)
    solve_time = time.perf_counter() - t0

    x_val = {(i,d,t): pyo.value(m.x[i,d,t]) for (i,d,t) in x_idx}
    planning = {}
    for (d,t) in slots:
        pats = [i for i in I if (x_val.get((i,d,t),0) or 0) > 0.5]
        if pats: planning[d,t] = sorted(pats, key=lambda i: inst.s[i])

    # Ré-évaluer sous coût exact
    E_idle=E_wait=E_OT=0.0
    for (d,t), pats in planning.items():
        L=inst.L[d][t]
        ei,ew,eo = eval_slot_exact(pats, inst.q, inst.s, L)
        E_idle+=inst.c_I*ei; E_wait+=inst.c_W*ew; E_OT+=inst.c_O*eo
    for (d,t) in slots:
        if (d,t) not in planning: E_idle += inst.c_I*inst.L[d][t]

    Z_reduced  = pyo.value(m.obj)
    Z_complete = E_idle + E_wait + E_OT  # coût exact
    n_served = sum(len(v) for v in planning.values())
    n_double = sum(1 for v in planning.values() if len(v)==2)
    return Solution(planning=planning, E_idle=E_idle, E_wait=E_wait, E_OT=E_OT,
                    Z_reduced=Z_reduced, Z_complete=Z_complete,
                    solve_time=solve_time, gap=0.0,
                    n_served=n_served, n_double=n_double,
                    x_val=x_val, z_val={}, coeffs=coeffs)
""")

code("""\
# ── 5.3 B3 : Glouton ──────────────────────────────────────────────────────────
def solve_b3_greedy(inst: Instance) -> Solution:
    \"\"\"Trie par q_i*s_i décroissant, remplit simple puis double les créneaux
    à idle espéré résiduel maximal.\"\"\"
    coeffs = compute_coefficients(inst)
    slots = coeffs["slots"]
    I = list(range(inst.n))

    priority = sorted(I, key=lambda i: inst.q[i]*inst.s[i], reverse=True)
    occupancy = {(d,t): [] for (d,t) in slots}
    assigned  = set()

    # Passe 1 : simple booking
    slot_list = list(slots)
    slot_idx  = 0
    for i in priority:
        while slot_idx < len(slot_list):
            dt = slot_list[slot_idx]
            if len(occupancy[dt]) < 1:
                occupancy[dt].append(i); assigned.add(i); break
            slot_idx += 1
        if slot_idx >= len(slot_list): break

    # Passe 2 : doubler les créneaux à idle résiduel max
    remaining = [i for i in priority if i not in assigned]
    idle_residual = {}
    for (d,t), pats in occupancy.items():
        if pats:
            i = pats[0]; L = inst.L[d][t]
            idle_residual[d,t] = inst.c_I*(L - inst.q[i]*inst.s[i])
        else:
            idle_residual[d,t] = inst.c_I*inst.L[d][t]

    double_order = sorted([(d,t) for (d,t) in slots],
                           key=lambda dt: idle_residual[dt], reverse=True)
    rem_idx = 0
    for (d,t) in double_order:
        if rem_idx >= len(remaining): break
        if len(occupancy[d,t]) < 2:
            occupancy[d,t].append(remaining[rem_idx])
            assigned.add(remaining[rem_idx]); rem_idx += 1

    planning = {}
    for (d,t), pats in occupancy.items():
        if pats:
            planning[d,t] = sorted(pats, key=lambda i: inst.s[i])

    E_idle=E_wait=E_OT=0.0
    for (d,t), pats in planning.items():
        L=inst.L[d][t]
        ei,ew,eo = eval_slot_exact(pats, inst.q, inst.s, L)
        E_idle+=inst.c_I*ei; E_wait+=inst.c_W*ew; E_OT+=inst.c_O*eo
    for (d,t) in slots:
        if (d,t) not in planning: E_idle += inst.c_I*inst.L[d][t]

    Z_complete = E_idle + E_wait + E_OT
    n_served = sum(len(v) for v in planning.values())
    n_double = sum(1 for v in planning.values() if len(v)==2)
    return Solution(planning=planning, E_idle=E_idle, E_wait=E_wait, E_OT=E_OT,
                    Z_reduced=Z_complete, Z_complete=Z_complete,
                    solve_time=0.0, gap=0.0,
                    n_served=n_served, n_double=n_double,
                    x_val={}, z_val={}, coeffs=coeffs)
""")

code("""\
# ── 5.4 B4 : Utilité type Valenzuela ─────────────────────────────────────────
def solve_b4_valenzuela(inst: Instance, Q_cap=1.2, cfg: Config = None) -> Solution:
    \"\"\"max Σ q_i*x - 0.5*Σ B[d,t], cap propension 1.2/créneau, k=2.
    Ré-évalué sous coût exact.\"\"\"
    if cfg is None: cfg = Config()
    coeffs = compute_coefficients(inst)
    slots = coeffs["slots"]
    I = list(range(inst.n))
    x_idx = [(i,d,t) for i in I for (d,t) in slots]
    b_idx = list(slots)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(x_idx, domain=pyo.Binary, initialize=0)
    m.B = pyo.Var(b_idx, domain=pyo.Binary, initialize=0)

    def _obj(m):
        return -(sum(inst.q[i]*m.x[i,d,t] for (i,d,t) in x_idx)
                 - 0.5*sum(m.B[d,t] for (d,t) in slots))
    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)
    m.C1 = pyo.ConstraintList()
    for i in I: m.C1.add(sum(m.x[i,d,t] for (d,t) in slots) <= 1)
    m.C2 = pyo.ConstraintList()
    for (d,t) in slots: m.C2.add(sum(m.x[i,d,t] for i in I) <= 2)
    m.Cprop = pyo.ConstraintList()
    for (d,t) in slots:
        m.Cprop.add(sum(inst.q[i]*m.x[i,d,t] for i in I) <= Q_cap)
    # B[d,t] = 1 si créneau double
    m.CB = pyo.ConstraintList()
    for (d,t) in slots:
        m.CB.add(m.B[d,t] >= (sum(m.x[i,d,t] for i in I) - 1) / 2)

    solver = pyo.SolverFactory(SOLVER_NAME)
    solver.options["time_limit"] = cfg.time_limit
    t0 = time.perf_counter()
    solver.solve(m, tee=False)
    solve_time = time.perf_counter() - t0

    x_val = {(i,d,t): pyo.value(m.x[i,d,t]) for (i,d,t) in x_idx}
    planning = {}
    for (d,t) in slots:
        pats = [i for i in I if (x_val.get((i,d,t),0) or 0) > 0.5]
        if pats: planning[d,t] = sorted(pats, key=lambda i: inst.s[i])

    E_idle=E_wait=E_OT=0.0
    for (d,t), pats in planning.items():
        L=inst.L[d][t]
        ei,ew,eo = eval_slot_exact(pats, inst.q, inst.s, L)
        E_idle+=inst.c_I*ei; E_wait+=inst.c_W*ew; E_OT+=inst.c_O*eo
    for (d,t) in slots:
        if (d,t) not in planning: E_idle += inst.c_I*inst.L[d][t]

    Z_complete = E_idle + E_wait + E_OT
    n_served = sum(len(v) for v in planning.values())
    n_double = sum(1 for v in planning.values() if len(v)==2)
    return Solution(planning=planning, E_idle=E_idle, E_wait=E_wait, E_OT=E_OT,
                    Z_reduced=Z_complete, Z_complete=Z_complete,
                    solve_time=solve_time, gap=0.0,
                    n_served=n_served, n_double=n_double,
                    x_val=x_val, z_val={}, coeffs=coeffs)
""")

code("""\
# ── 5.5 Table comparative sur la cellule centrale (R réplications) ────────────
_BENCH_R = min(R, 10)  # 10 reps pour les benchmarks (budget)

def _bench_cell(seed):
    inst = generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15, cW_ratio=0.50)
    sol_ilp  = solve_ilp(inst)
    sol_b1   = solve_b1_single(inst)
    sol_b2   = solve_b2_fluid(inst)
    sol_b3   = solve_b3_greedy(inst)
    sol_b4   = solve_b4_valenzuela(inst)
    ref_Z    = sol_ilp.Z_complete
    def pct(z): return 100*(z/ref_Z - 1) if ref_Z > 1e-9 else 0.0
    return {
        "seed": seed,
        "ILP_Z":  ref_Z,
        "B1_Z":   sol_b1.Z_complete,  "B1_pct": pct(sol_b1.Z_complete),
        "B2_Z":   sol_b2.Z_complete,  "B2_pct": pct(sol_b2.Z_complete),
        "B3_Z":   sol_b3.Z_complete,  "B3_pct": pct(sol_b3.Z_complete),
        "B4_Z":   sol_b4.Z_complete,  "B4_pct": pct(sol_b4.Z_complete),
        "B1_double": sol_b1.n_double, "B2_double": sol_b2.n_double,
        "B3_double": sol_b3.n_double, "B4_double": sol_b4.n_double,
    }

bench_rows = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(_bench_cell)(SEEDS[r]) for r in range(_BENCH_R)
)
_df_bench = pd.DataFrame(bench_rows)
_df_bench.to_csv("results/benchmarks.csv", index=False)

_bench_summary = _df_bench[["B1_pct","B2_pct","B3_pct","B4_pct",
                              "B1_double","B2_double","B3_double","B4_double"]].mean().round(2)
print("=== Table comparative (moyennes sur", _BENCH_R, "réplications) ===")
print(_bench_summary.to_string())
print("\\nNote : B2 (fluide Jensen) sur-booké si B2_double > ILP_double moyenne.")
print(f"ILP double moyen : {_df_doe[_df_doe.label=='central']['n_double'].mean():.1f}")
""")

code("""\
# ── 5.6 Export LaTeX table comparative ───────────────────────────────────────
_cols_tex = ["ILP_Z","B1_Z","B2_Z","B3_Z","B4_Z","B1_pct","B2_pct","B3_pct","B4_pct"]
_df_bench[_cols_tex].describe().round(3).to_latex(
    "tables/table_benchmarks.tex",
    caption="Comparaison des heuristiques vs PLNE (cellule centrale, R=10).",
    label="tab:benchmarks")
print("table_benchmarks.tex exportée.")
""")

# ── Section 6 : Simulation Lindley ───────────────────────────────────────────
md("## 6. Simulation hors-échantillon, Dynamique séquentielle (Lindley)\n\n"
   "**Simplification signalée** : M=1 000 tirages pour le DOE (M=10 000 pour l'exemple "
   "illustratif). Réduire davantage si le budget est dépassé.")

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
# ── 6.3 Écart modèle/simulé sur le DOE (M=1000) ──────────────────────────────
M_DOE_SIM = 1_000   # ⚠ réduit à 1000 pour respecter le budget 45 min

def _sim_cell(cell):
    rng = np.random.default_rng(cell["seed"] + 9999)
    inst = generate_instance(**{k:v for k,v in cell.items() if k not in ("label",)})
    sol  = solve_ilp(inst, Config(time_limit=60.0))
    Z_model = sol.E_idle + sol.E_wait + sol.E_OT
    Z_sim   = simulate_instance(inst, sol.planning, M_DOE_SIM, rng, cv=0.0).mean()
    Z_sim_ln3 = simulate_instance(inst, sol.planning, M_DOE_SIM, rng, cv=0.3).mean()
    Z_sim_ln5 = simulate_instance(inst, sol.planning, M_DOE_SIM, rng, cv=0.5).mean()
    gap_pct   = 100*(Z_sim/Z_model - 1) if Z_model > 1e-9 else 0.0
    return {
        "label": cell["label"], "seed": cell["seed"],
        "ratio": cell["ratio"], "p_bar": cell["p_bar"],
        "long_share": cell["long_share"],
        "Z_model": Z_model, "Z_sim": Z_sim,
        "Z_sim_cv03": Z_sim_ln3, "Z_sim_cv05": Z_sim_ln5,
        "gap_pct": gap_pct,
        "n_double": sol.n_double,
        "double_rate": sol.n_double / (sum(inst.T) + 1e-9),
    }

_t0_sim = time.perf_counter()
sim_results = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(_sim_cell)(c) for c in DOE_CELLS
)
_t_sim = time.perf_counter() - _t0_sim
print(f"Simulation DOE terminée en {_t_sim/60:.1f} min")

_df_sim = pd.DataFrame(sim_results)
_df_sim.to_csv("results/simulation_doe.csv", index=False)
print("\\nÉcart % moyen par cellule :")
print(_df_sim.groupby("label")["gap_pct"].agg(["mean","std"]).round(2))
""")

code("""\
# ── 6.4 Figures : écart % vs n/S et part de doubles ─────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Écart vs ratio (n/S)
grp_ratio = _df_sim.groupby("ratio")["gap_pct"].mean()
ax1.plot(grp_ratio.index, grp_ratio.values, "o-", color="#2196F3", linewidth=2)
ax1.set_xlabel("Ratio n/S"); ax1.set_ylabel("Écart % (simulé vs modèle)")
ax1.set_title("Écart % vs charge n/S"); ax1.grid(True, alpha=0.3)

# Écart vs double_rate
ax2.scatter(_df_sim["double_rate"], _df_sim["gap_pct"], alpha=0.4, s=15, color="#FF9800")
ax2.set_xlabel("Part de créneaux doubles"); ax2.set_ylabel("Écart %")
ax2.set_title("Écart % vs part de doubles"); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/sim_gap_analysis.pdf", bbox_inches="tight")
plt.savefig("figures/sim_gap_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure sim_gap_analysis sauvegardée.")
""")

code("""\
# ── 6.5 Variante lognormale (CV=0.3 et 0.5) ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
cvs = ["gap_pct", "Z_sim_cv03", "Z_sim_cv05"]
lbls = ["CV=0 (déterministe)", "CV=0.3 (lognorm.)", "CV=0.5 (lognorm.)"]
for ax, col, lbl in zip(axes, cvs, lbls):
    if col == "gap_pct":
        vals = _df_sim.groupby("ratio")["gap_pct"].mean()
        ax.plot(vals.index, vals.values, "o-")
        ax.set_ylabel("Écart % coût simulé/modèle")
    else:
        ref = _df_sim["Z_model"]
        vals = _df_sim.groupby("ratio")[col].mean() / _df_sim.groupby("ratio")["Z_model"].mean()
        ax.plot(vals.index, (vals.values-1)*100, "o-", color="#E91E63")
        ax.set_ylabel("Dégradation % vs modèle")
    ax.set_title(lbl); ax.set_xlabel("Ratio n/S"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/sim_lognormal.pdf", bbox_inches="tight")
plt.savefig("figures/sim_lognormal.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ── Section 7 : Sensibilité ──────────────────────────────────────────────────
md("## 7. Analyse de sensibilité et robustesse")

code("""\
# ── 7.1 Miscalibration de q ───────────────────────────────────────────────────
EPSILONS = [-0.10, -0.05, 0.05, 0.10]
_SENS_R = min(R, 10)

def _miscal_cell(seed, eps):
    inst_true = generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15, cW_ratio=0.50)
    # Optimiser avec q̂ = clip(q + eps)
    q_hat = np.clip(inst_true.q + eps, 0.05, 0.99)
    inst_hat = Instance(n=inst_true.n, D=inst_true.D, T=inst_true.T, L=inst_true.L,
                        s=inst_true.s, q=q_hat,
                        c_I=inst_true.c_I, c_W=inst_true.c_W, c_O=inst_true.c_O)
    sol_hat = solve_ilp(inst_hat)

    # Évaluer sous q vrai
    rng = np.random.default_rng(seed+77777)
    Z_sim_true = simulate_instance(inst_true, sol_hat.planning, 1000, rng).mean()

    # Référence : solution optimale avec q vrai
    sol_true = solve_ilp(inst_true)
    rng2 = np.random.default_rng(seed+88888)
    Z_sim_opt  = simulate_instance(inst_true, sol_true.planning, 1000, rng2).mean()

    deg = 100*(Z_sim_true/Z_sim_opt - 1) if Z_sim_opt > 1e-9 else 0.0
    return {"seed": seed, "eps": eps, "Z_miscal": Z_sim_true,
            "Z_opt": Z_sim_opt, "deg_pct": deg}

miscal_rows = []
for eps in EPSILONS:
    rows = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_miscal_cell)(SEEDS[r], eps) for r in range(_SENS_R)
    )
    miscal_rows.extend(rows)

_df_miscal = pd.DataFrame(miscal_rows)

# Bruit gaussien
def _noise_cell(seed, sigma=0.05):
    inst_true = generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15, cW_ratio=0.50)
    rng = np.random.default_rng(seed+12345)
    q_hat = np.clip(inst_true.q + rng.normal(0, sigma, inst_true.n), 0.05, 0.99)
    inst_hat = Instance(n=inst_true.n, D=inst_true.D, T=inst_true.T, L=inst_true.L,
                        s=inst_true.s, q=q_hat,
                        c_I=inst_true.c_I, c_W=inst_true.c_W, c_O=inst_true.c_O)
    sol_hat = solve_ilp(inst_hat)
    rng2 = np.random.default_rng(seed+22222)
    Z_sim = simulate_instance(inst_true, sol_hat.planning, 1000, rng2).mean()
    sol_opt = solve_ilp(inst_true)
    rng3 = np.random.default_rng(seed+33333)
    Z_opt = simulate_instance(inst_true, sol_opt.planning, 1000, rng3).mean()
    return {"seed": seed, "eps": f"N(0,{sigma})", "Z_miscal": Z_sim,
            "Z_opt": Z_opt, "deg_pct": 100*(Z_sim/Z_opt-1) if Z_opt>1e-9 else 0.0}

noise_rows = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(_noise_cell)(SEEDS[r]) for r in range(_SENS_R)
)
_df_miscal = pd.concat([_df_miscal, pd.DataFrame(noise_rows)], ignore_index=True)
_df_miscal.to_csv("results/sensitivity_miscal.csv", index=False)

print("Dégradation % par niveau de miscalibration :")
print(_df_miscal.groupby("eps")["deg_pct"].agg(["mean","std"]).round(2))
""")

code("""\
# ── 7.2 Perturbation des durées ───────────────────────────────────────────────
DUR_DELTAS = [-0.20, -0.10, 0.10, 0.20]
MISCLASS_RATES = [0.05, 0.10, 0.20]

def _dur_cell(seed, delta_s):
    inst = generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15, cW_ratio=0.50)
    s_pert = inst.s * (1 + delta_s)
    inst_p = Instance(n=inst.n, D=inst.D, T=inst.T, L=inst.L,
                      s=s_pert, q=inst.q,
                      c_I=inst.c_I, c_W=inst.c_W, c_O=inst.c_O)
    sol = solve_ilp(inst_p)
    # Évaluer sous durées vraies
    rng = np.random.default_rng(seed+55)
    Z_eval = simulate_instance(inst, sol.planning, 1000, rng).mean()  # durées vraies
    sol_opt = solve_ilp(inst)
    rng2 = np.random.default_rng(seed+66)
    Z_opt = simulate_instance(inst, sol_opt.planning, 1000, rng2).mean()
    return {"seed": seed, "perturb": f"{int(delta_s*100):+d}%",
            "deg_pct": 100*(Z_eval/Z_opt-1) if Z_opt>1e-9 else 0.0}

def _misclass_cell(seed, rate):
    rng = np.random.default_rng(seed+111)
    inst = generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15, cW_ratio=0.50)
    swap = rng.random(inst.n) < rate
    s_swap = np.where(swap,
                      np.where(inst.s > (s_A+s_B)/2, s_B, s_A),  # swap A↔B
                      inst.s)
    inst_p = Instance(n=inst.n, D=inst.D, T=inst.T, L=inst.L,
                      s=s_swap, q=inst.q,
                      c_I=inst.c_I, c_W=inst.c_W, c_O=inst.c_O)
    sol = solve_ilp(inst_p)
    rng2 = np.random.default_rng(seed+222)
    Z_eval = simulate_instance(inst, sol.planning, 1000, rng2).mean()
    sol_opt = solve_ilp(inst)
    rng3 = np.random.default_rng(seed+333)
    Z_opt  = simulate_instance(inst, sol_opt.planning, 1000, rng3).mean()
    return {"seed": seed, "perturb": f"misclass {int(rate*100)}%",
            "deg_pct": 100*(Z_eval/Z_opt-1) if Z_opt>1e-9 else 0.0}

dur_rows = []
for d in DUR_DELTAS:
    dur_rows += joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_dur_cell)(SEEDS[r], d) for r in range(_SENS_R))
for r in MISCLASS_RATES:
    dur_rows += joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_misclass_cell)(SEEDS[rr], r) for rr in range(_SENS_R))

_df_dur = pd.DataFrame(dur_rows)
_df_dur.to_csv("results/sensitivity_durations.csv", index=False)
print("Dégradation %, perturbation des durées :")
print(_df_dur.groupby("perturb")["deg_pct"].agg(["mean","std"]).round(2))
""")

code("""\
# ── 7.3 Sensibilité à Q̄ et déséquilibre de classes ─────────────────────────
Q_BARS   = [1.0, 1.2, 1.5, None]
PI_AS    = [0.05, 0.15, 0.30]

def _qbar_cell(seed, Q_bar, pi_A_val):
    inst = generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15,
                             cW_ratio=0.50, pi_A_=pi_A_val, Q_bar=Q_bar)
    sol  = solve_ilp(inst)
    return {"seed": seed, "Q_bar": str(Q_bar), "pi_A": pi_A_val,
            "Z_complete": sol.Z_complete, "n_served": sol.n_served,
            "n_double": sol.n_double, "solve_time": sol.solve_time}

qbar_rows = []
for qb in Q_BARS:
    for pa in PI_AS:
        qbar_rows += joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_qbar_cell)(SEEDS[r], qb, pa) for r in range(min(_SENS_R, 5)))

_df_qbar = pd.DataFrame(qbar_rows)
_df_qbar.to_csv("results/sensitivity_qbar.csv", index=False)
print("Effet Q̄ et π_A sur Z_complete :")
print(_df_qbar.pivot_table(values="Z_complete", index="Q_bar",
                            columns="pi_A", aggfunc="mean").round(3))
""")

code("""\
# ── 7.4 Front de Pareto idle–attente (sweep c_W/c_I) ─────────────────────────
CW_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

def _pareto_cell(seed, cwr):
    inst = generate_instance(seed, ratio=1.5, p_bar=0.20, long_share=0.15, cW_ratio=cwr)
    sol  = solve_ilp(inst)
    return {"seed": seed, "cW_ratio": cwr,
            "E_idle": sol.E_idle, "E_wait": sol.E_wait, "E_OT": sol.E_OT}

pareto_rows = []
for cwr in CW_SWEEP:
    pareto_rows += joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_pareto_cell)(SEEDS[r], cwr) for r in range(min(_SENS_R, 5)))

_df_pareto = pd.DataFrame(pareto_rows)
_df_pareto.to_csv("results/pareto_idle_wait.csv", index=False)

# Figure Pareto
grp_p = _df_pareto.groupby("cW_ratio")[["E_idle","E_wait"]].mean()
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(grp_p["E_idle"], grp_p["E_wait"],
                c=np.log(grp_p.index), cmap="viridis", s=80, zorder=5)
for cwr, row in grp_p.iterrows():
    ax.annotate(f"cW={cwr}", (row["E_idle"], row["E_wait"]),
                fontsize=7, textcoords="offset points", xytext=(4,4))
ax.plot(grp_p["E_idle"], grp_p["E_wait"], "k--", alpha=0.4)
ax.set_xlabel("E[Idle]"); ax.set_ylabel("E[Wait]")
ax.set_title("Front de Pareto idle–attente"); ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label="log(c_W/c_I)")
plt.tight_layout()
plt.savefig("figures/pareto_idle_wait.pdf", bbox_inches="tight")
plt.savefig("figures/pareto_idle_wait.png", dpi=150, bbox_inches="tight")
plt.show()
print("Front de Pareto sauvegardé.")
""")

code("""\
# ── 7.5 Tableau de synthèse % déviation + verdict de stabilité ───────────────
_sens_summary = []

def _add_sens(label, df, col="deg_pct"):
    grp = df.groupby("perturb" if "perturb" in df.columns else "eps")[col]
    for k, v in grp:
        _sens_summary.append({"Source": label, "Perturbation": str(k),
                              "Déviation Z %": round(v.mean(), 2),
                              "Écart-type": round(v.std(), 2)})

_add_sens("Miscalibration q", _df_miscal, "deg_pct")
_add_sens("Durées/Misclassif.", _df_dur, "deg_pct")

_df_sens = pd.DataFrame(_sens_summary)
_df_sens.to_csv("results/sensitivity_summary.csv", index=False)

print("=== Synthèse de robustesse ===")
print(_df_sens.to_string(index=False))

_max_dev = _df_sens["Déviation Z %"].abs().max()
if _max_dev < 5:
    verdict = "STABLE (déviation max < 5%)"
elif _max_dev < 10:
    verdict = "MODÉRÉMENT STABLE (déviation max < 10%)"
else:
    verdict = f"INSTABLE (déviation max = {_max_dev:.1f}%)"
print(f"\\nVerdict : {verdict}")

_df_sens.to_latex("tables/table_sensitivity.tex", index=False, float_format="%.2f",
                  caption="Synthèse de robustesse (% déviation du coût simulé).",
                  label="tab:sensitivity")
""")

# ── RESULTS.md ────────────────────────────────────────────────────────────────
code("""\
# ── 8. Export RESULTS.md ─────────────────────────────────────────────────────
_central = _df_doe[_df_doe.label=="central"]
_bench_m  = _df_bench.mean()

_md = f\"\"\"# RESULTS.md, Chiffres clés pour le manuscrit

## Table 1 (Section 3)
- Z_complet instance illustrative : {_sol3.Z_complete:.4f}
- E[Idle] : {_sol3.E_idle:.4f}  E[Wait] : {_sol3.E_wait:.4f}  E[OT] : {_sol3.E_OT:.4f}
- Patients servis : {_sol3.n_served}/10  Créneaux doubles : {_sol3.n_double}/4
- Temps solveur : {_sol3.solve_time:.3f} s

## DOE, Cellule centrale (ratio=1.5, p̄=0.20, ls=0.15, c_W=0.50, R={R})
- Z_complet moyen : {_central['Z_complete'].mean():.3f} ± {_central['Z_complete'].std():.3f}
- E[Idle] moyen : {_central['E_idle'].mean():.3f}
- E[Wait] moyen : {_central['E_wait'].mean():.3f}
- E[OT]   moyen : {_central['E_OT'].mean():.3f}
- Patients servis / n : {_central['occ'].mean():.3f}
- Créneaux doubles / S : {_central['double_rate'].mean():.3f}
- Temps solveur moyen : {_central['solve_time'].mean():.2f} s

## Benchmarks (R={_BENCH_R})
- B1 (single) : +{_bench_m['B1_pct']:.1f}% vs ILP
- B2 (fluide Jensen) : +{_bench_m['B2_pct']:.1f}% vs ILP
- B3 (glouton) : +{_bench_m['B3_pct']:.1f}% vs ILP
- B4 (Valenzuela) : +{_bench_m['B4_pct']:.1f}% vs ILP

## Simulation (M={M_DOE_SIM})
- Écart % modèle/simulé (cellule centrale) : {_df_sim[_df_sim.label=='central']['gap_pct'].mean():.2f}%

## Robustesse
- Verdict : {verdict}

## Fichiers
- results/doe_ilp.csv, results/benchmarks.csv
- results/simulation_doe.csv, results/sensitivity_*.csv
- tables/table1_planning.tex, tables/table_benchmarks.tex, tables/table_sensitivity.tex
- figures/doe_metrics.pdf, figures/sim_gap_analysis.pdf, figures/pareto_idle_wait.pdf
\"\"\"

with open("RESULTS.md", "w") as f:
    f.write(_md)
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
