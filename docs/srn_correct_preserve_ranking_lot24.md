# Lot 24 — Correct-and-Preserve Strategic Ranking

## Objet et protocole

Ce lot teste une modification contrôlée de l'objectif Policy, sans changer le moteur, le SRN, MCTS ni les données. Tous les candidats partent bit à bit de `G2-best`. La classification des paires est calculée une fois avec G2 gelé et ne devient jamais une cible mobile. `epsilon_gap=0.02` et les splits du Lot 23 sont conservés. Aucun self-play, Teacher ou Minimax n'est exécuté; Qdiag n'est jamais une cible Value.

Pour une paire Qdiag fiable `(a+,a-)`, de poids borné `w`, on définit `dθ=lθ(a+)−lθ(a−)`. Si G2 inverse l'ordre, la paire est une correction et

`Lcorr = Σ w·softplus(−dθ) / Σw`.

Si G2 respecte déjà l'ordre, sa marge `mG2>0` est protégée par

`Lpres = Σ w·relu(ρ·mG2−dθ) / Σw`.

L'objectif est `LD_RL-policy + LD_RL-value + 0.1·Lcorr + λpres·Lpres`, auquel C24-CPH ajoute `0.25·CE` sur la distribution historique MCTS128. La Value n'apprend que depuis le résultat terminal réel de D_RL. La calibration courte `ρ∈{0.25,0.50,0.75}`, `λpres∈{0.5,1,2}` a retenu `ρ=0.5`, `λpres=1.0`, sans consulter le test ni les arènes.

## Population de paires avant entraînement

| Split | Positions | Correction | Préservation | Quasi équivalentes | Instables exclues |
|---|---:|---:|---:|---:|---:|
| Train | 848 | 2 296 | 1 922 | 2 538 | 13 |
| Validation | 183 | 515 | 426 | 491 | 1 |
| Test strict | 169 | 408 | 420 | 520 | 3 |

Les distributions complètes des gaps, des poids et des marges parentales sont dans `pair_classification.json`. Le graphe de contraintes ne contient aucun cycle (ordre Qdiag scalaire). Les actions partagées entre contraintes de correction et de préservation sont au nombre de 1 272/290/261 sur train/validation/test. Le test unitaire de gradient contradictoire vérifie que `Lpres` oppose le gradient qui détruirait une relation protégée.

Le micro-overfit fait baisser `Lcorr` de 0,7523 à 0,7273 et porte la correction de 0 à 17,43 %. Les paires quasi équivalentes ont un gradient nul par exclusion. L'infrastructure apprend donc bien le signal visé; la préservation est jugée sur la validation complète, pas sur ce petit lot volontairement conflictuel.

## Résultats offline sur le test strict

| Modèle | Correction | Préservation | Damage | Efficacité | Acc. pondérée | SWI | Regret moyen | High-gap | Utility | Value MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G2 | 0,00 % | 100,00 % | 0,00 % | — | 44,05 % | 65,43 | 0,1015 | 44 | 0,00 | 0,7162 |
| C24-CORR | 0,00 % | 100,00 % | 0,00 % | — | 44,05 % | 65,43 | 0,1015 | 44 | 0,00 | 0,7162 |
| C24-CP | 5,88 % | 93,33 % | 6,67 % | 0,86 | 44,08 % | 65,66 | 0,1036 | 44 | 0,05 | 0,7480 |
| C24-CPH | 10,05 % | 92,38 % | 7,62 % | 1,28 | 48,43 % | 60,29 | 0,0973 | 37 | 8,24 | 0,7291 |
| Lot23-RANK | 55,64 % | 64,76 % | 35,24 % | 1,53 | 60,55 % | 45,66 | 0,0627 | 22 | 31,08 | 0,7379 |
| Lot23-HYBRID | 47,06 % | 65,24 % | 34,76 % | 1,32 | 60,42 % | 45,71 | 0,0634 | 22 | 30,83 | 0,7169 |

`damage=1−preservation`. L'efficacité est le nombre de corrections divisé par les nouvelles destructions. L'utilité stratégique descriptive vaut `Σ poids corrigés − Σ poids détruits`; elle n'est pas un oracle de force de jeu.

C24-CPH est le seul candidat qui satisfait tout le gate offline: préservation ≥90 %, correction positive, utilité positive, SWI inférieur à G2, regret non supérieur et Value préservée. Il déplace donc bien le compromis du Lot 23 vers une région sûre: moins de corrections (10,05 % contre 55,64 %), mais 92,38 % de préservation contre 64,76 %.

## Arènes contre G2

La short arena MCTS128 donne 26/3/34 sur 63 parties terminales, score 43,65 %, IC95 apparié `[32,81 %;53,91 %]`; le seuil de screening de 40 % est franchi.

| Budget | W/D/L terminal | Score | IC95 apparié | Troncatures | Longueur moyenne |
|---|---:|---:|---:|---:|---:|
| MCTS64 | 108/11/135 | 44,69 % | [39,45 %;50,00 %] | 2 | 89,12 |
| MCTS128 | 100/8/145 | 41,11 % | [35,35 %;46,68 %] | 3 | 96,52 |

Le scaling est déclaré sain selon le critère préfixé (la baisse reste inférieure à cinq points), mais C24-CPH est nettement inférieur à G2 aux deux budgets. Aucun contrôle MCTS256 n'est justifié et aucun G3 n'est promu.

## Réponses finales

1. Les volumes actifs sont 2 296/1 922 (train), 515/426 (validation) et 408/420 (test) pour correction/préservation.
2. Oui, la loss asymétrique corrige réellement des relations: C24-CPH atteint 10,05 % sur le test.
3. Oui, elle protège au moins 90 % des relations: 92,38 % pour C24-CPH.
4. Face aux 55,64 %/64,76 % de Lot23-RANK, le compromis devient 10,05 %/92,38 %: correction plus sélective, dommage fortement réduit.
5. Oui pour C24-CPH: SWI 65,43→60,29, regret 0,1015→0,0973 et inversions high-gap 44→37; l'amélioration est toutefois moindre que celle de RANK.
6. Oui, C24-CPH franchit le gate offline complet.
7. Non, l'arène ne confirme pas une progression: 44,69 % à MCTS64 et 41,11 % à MCTS128 contre G2.

Question scientifique: **oui**, on peut corriger sélectivement G2 tout en préservant au moins 90 % de ses relations fiables. Question de jeu: **non**, cette correction ne produit pas encore une Policy plus forte que G2 pour guider MCTS.

## Verdicts

```text
CORRECT_PRESERVE_IMPLEMENTATION_VALID = YES
CORRECTION_PAIRS_LEARNED = YES
PRESERVATION_PAIRS_PROTECTED = YES
SAFE_CORRECTION_REGION_FOUND = YES
HIGH_GAP_INVERSIONS_REDUCED = YES
NET_STRATEGIC_GAIN_POSITIVE = YES
VALUE_PRESERVED = YES
CORRECT_AND_PRESERVE_OBJECTIVE_INSUFFICIENT = NO
OFFLINE_STRATEGIC_METRICS_NOT_SUFFICIENT = YES
SEARCH_SCALING_HEALTHY = YES
G3_CANDIDATE = NONE
NEXT_ACTION = SEARCH_POLICY_INTERACTION_DIAGNOSIS
```

Conclusion: le Lot 24 résout le problème scientifique de correction sous contrainte de préservation, mais révèle que ces métriques Qdiag offline ne suffisent toujours pas à prédire la qualité d'une Policy dans MCTS. La prochaine expérience doit donc diagnostiquer l'interaction Policy–recherche, sans produire de nouveau self-play dans ce lot.
