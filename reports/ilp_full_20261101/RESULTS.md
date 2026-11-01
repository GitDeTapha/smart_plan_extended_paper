# RESULTS.md, chiffres clés pour le manuscrit

Paramètres : run_full_20261030 (commit 8615e6564b67321c11e2ee6208425014f15d0c85), s_A = 36.00, s_B = 7.00,
pi_A = 0.3463, c_O = 0.0. Campagne : S = 8, R = 30, limite 600 s,
gap cible 1e-06. Smoke : False. Tous les patients servis (C1 en égalité).
Holdout : 4,013,341 patients, 2073 q_i hors de (0.01, 0.999), tronqués dans les instances.

## Table 1 (section 3)
- Historique (n = 7, s_A = 43, s_B = 8) : Z* = 65.6200 (optimal)
- Données : Z* = 79.7613

## Plan d'expériences
- Optimalité prouvée : 99.8 % des PLNE
- Résolution : médiane 21.2 s, max 600.6 s,
  mémoire max 597 Mo par processus (Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz,
  15 Go de RAM)
- Cellule centrale : Z = 135.268 ± 27.046

## Benchmarks (coût exact du modèle, Delta Z vs PLNE, IC 95 %)
- B1 Single booking : +10.66 [9.88 ; 11.44], p = 1.86e-09, served 8.0
- B2 Fluid (Jensen) : +5.47 [3.57 ; 7.36], p = 1.86e-09, served 12.0
- B3 Greedy : +3.50 [2.37 ; 4.63], p = 1.86e-09, served 12.0
- B4 Valenzuela utility : +32.10 [26.29 ; 37.90], p = 1.86e-09, served 8.0
- B5 Uniform overbooking : +19.26 [13.12 ; 25.40], p = 1.86e-09, served 12.0

## Simulation (M = 10000)
- Écart simulé / modèle, cellule centrale : 2.29 %

## Robustesse
- 8/13 perturbations significatives ; regret exact maximal 9.08 %

## Hors échantillon (holdout, coût réalisé, Delta Z vs PLNE)
- ILP_oracle : -24.61 [-30.79 ; -18.42], p = 2.44e-15, served 12.0
- B1 : -4.75 [-14.24 ; 4.73], p = 0.389, served 8.0
- B3 : +1.21 [-6.13 ; 8.54], p = 0.568, served 12.0
- B5 : +31.32 [18.41 ; 44.23], p = 4.04e-06, served 12.0
