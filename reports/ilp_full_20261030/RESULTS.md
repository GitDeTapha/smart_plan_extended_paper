# RESULTS.md, chiffres clés pour le manuscrit

Paramètres : run_full_20261030 (commit 8615e6564b67321c11e2ee6208425014f15d0c85), s_A = 36.00, s_B = 7.00,
pi_A = 0.3463, c_O = 0.0. Campagne : S = 8, R = 30, limite 600 s,
gap cible 1e-06. Smoke : False.
Holdout : 4,013,341 patients, 2073 q_i hors de (0.01, 0.999), tronqués dans les instances.

## Table 1 (section 3)
- Historique (s_A = 43, s_B = 8) : Z* = 65.6200 (optimal)
- Données : Z* = 79.7613

## Plan d'expériences
- Optimalité prouvée : 99.6 % des PLNE
- Résolution : médiane 15.0 s, max 600.6 s,
  mémoire max 565 Mo par processus (Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz,
  15 Go de RAM)
- Cellule centrale : Z = 135.193 ± 27.237

## Benchmarks (coût exact du modèle, Delta Z vs PLNE, IC 95 %)
- B1 Réservation simple : +10.73 [10.05 ; 11.42], p = 1.86e-09
- B2 Fluide (Jensen) : +5.94 [4.08 ; 7.80], p = 1.86e-09
- B3 Glouton : +3.58 [2.44 ; 4.72], p = 1.86e-09
- B4 Utilité Valenzuela : +32.17 [26.39 ; 37.96], p = 1.86e-09
- B5 Surréservation uniforme : +19.34 [13.15 ; 25.52], p = 1.86e-09

## Simulation (M = 10000)
- Écart simulé / modèle, cellule centrale : 1.74 %

## Robustesse
- 9/13 perturbations significatives ; regret exact maximal 10.08 %

## Hors échantillon (holdout, coût réalisé, Delta Z vs PLNE)
- ILP_oracle : -26.74 [-34.38 ; -19.10], p = 5.49e-13
- B1 : -0.52 [-8.23 ; 7.19], p = 0.0383
- B3 : +5.44 [-3.48 ; 14.36], p = 0.102
- B5 : +35.55 [21.59 ; 49.52], p = 4.05e-07
