# Lot 23 — Strategically Weighted Ranking Policy

## 1. Objectif et protocole

Le Lot 23 teste si une Policy qui privilégie les relations entre actions séparées par un grand strategic gap constitue un meilleur objectif que l'imitation distributionnelle uniforme. Trois candidats, tous initialisés exactement depuis G2, sont comparés :

- **C23-CE** : CE historique `D_RL`, CE MCTS128 contrôlée et Value `D_RL` ;
- **C23-RANK** : CE historique `D_RL`, ranking stratégique et Value `D_RL` ;
- **C23-HYBRID** : les deux composantes Policy et Value `D_RL`.

Qdiag n'est jamais utilisé comme cible Value. Aucun Teacher, Minimax, nouveau self-play ou changement du moteur, de MCTS ou du SRN n'intervient.

## 2. Construction de D_RANK

`D_RANK` réutilise les 1 200 positions Qdiag du Lot 22. Le split physique, déterministe et sans intersection contient :

| Split | Positions |
|---|---:|
| Train | 848 |
| Validation | 183 |
| Test strict | 169 |

Le corpus contient 620 positions issues de cibles MCTS128 et 580 positions issues de la batterie Lot 17. Après filtrage, 5 987 paires stratégiques sont disponibles. Deux cents positions possèdent simultanément Q256 et Q512 ; leurs paires dont le signe change sont exclues.

## 3. Calibration de la zone d'équivalence

| `|gap Q256|` | Paires | Stabilité de l'ordre Q256/Q512 |
|---|---:|---:|
| < 0,01 | 367 | 66,76 % |
| 0,01–0,02 | 201 | 78,11 % |
| 0,02–0,05 | 310 | 94,84 % |
| 0,05–0,10 | 280 | 99,64 % |
| 0,10–0,20 | 231 | 100 % |
| ≥ 0,20 | 340 | 100 % |

La plus petite frontière après laquelle chaque bin atteint au moins 90 % est :

```text
epsilon_gap = 0,02
```

Les paires sous ce seuil ne produisent aucune contrainte. Le contrôle d'incertitude reste `PARTIAL`, car Q512 n'est disponible que pour 200 positions.

## 4. Poids et loss

Pour une paire fiable `a+ > a-` :

```text
w = clip((|gap| - 0,02) / (q95_gap - 0,02), 0, 1)
L_rank = sum(w * softplus(-(logit(a+) - logit(a-)))) / sum(w)
```

Le quantile 95 des gaps vaut 0,52088. Les poids sont bornés dans `[0,1]`, avec moyenne 0,25385, médiane 0,13789 et P95 à 1.

La calibration des gradients compare `lambda_rank ∈ {0,10 ; 0,25 ; 0,50}`. Même la plus petite valeur produit 57,7 % de la norme du gradient Policy historique ; elle est retenue comme intervention minimale :

```text
lambda_rank = 0,10
```

Cette observation explique pourquoi une préservation explicite devra probablement faire partie de la reformulation suivante.

## 5. Validation technique

Le micro-overfit confirme que la loss diminue, que l'accuracy pondérée augmente, que les gradients sont finis, que les actions illégales sont absentes et qu'aucune cible Value Qdiag n'existe.

Les trois candidats reproduisent exactement les logits, probabilités et Value de G2 à l'epoch 0. Les meilleurs epochs, choisis sans arène, sont CE=16, RANK=18 et HYBRID=9.

## 6. Résultats sur le test strict

| Modèle | Accuracy pondérée | Correction | Préservation | Gain net | SWI | Regret moyen | High-gap inversions |
|---|---:|---:|---:|---:|---:|---:|---:|
| G2 | 44,05 % | 0 % | 100 % | 0 | 65,43 | 0,1015 | 44 |
| C23-CE | 56,51 % | 33,09 % | 78,57 % | +23,47 | 52,00 | 0,0787 | 31 |
| C23-RANK | **60,55 %** | **55,64 %** | 64,76 % | **+31,08** | **45,66** | **0,0627** | **22** |
| C23-HYBRID | 60,42 % | 47,06 % | **65,24 %** | +30,83 | 45,71 | 0,0634 | **22** |
| C20-DUAL | 51,07 % | 17,16 % | 86,90 % | +13,22 | 57,72 | 0,0936 | 35 |
| C21-B | 47,88 % | 16,18 % | 85,00 % | +7,21 | 61,23 | 0,1008 | 40 |

Le ranking apprend donc bien le signal attendu. Par rapport à G2, C23-RANK réduit SWI de 30,2 %, le regret moyen de 38,2 % et les inversions high-gap de 50 %.

## 7. Limite centrale : correction contre préservation

Le gain offline n'est pas une mise à jour sûre. C23-RANK corrige 55,6 % des relations erronées, mais ne conserve que 64,8 % des relations fiables que G2 ordonnait déjà correctement. HYBRID ne rétablit pratiquement pas la préservation.

Le gain stratégique net est défini par :

```text
NET_STRATEGIC_GAIN = poids des correction-pairs corrigées
                   - poids des relations G2 correctes cassées
```

Il est positif pour les trois candidats, mais masque le nombre important de relations détruites. La règle du lot exige les deux propriétés ; un gain agrégé positif ne suffit donc pas.

## 8. Value et Policy historique

| Modèle | MSE Value | CE D_RL |
|---|---:|---:|
| G2 | 0,71624 | 1,39338 |
| C23-CE | 0,75412 | 1,39195 |
| C23-RANK | 0,73791 | 1,39270 |
| C23-HYBRID | **0,71690** | 1,39302 |

La Value reste dans la tolérance de 10 % pour les trois candidats. HYBRID la préserve presque exactement.

## 9. Gate et absence d'arène

Le gate exige notamment un taux de préservation d'au moins 90 %. Aucun candidat ne le franchit :

- CE : 78,57 % ;
- RANK : 64,76 % ;
- HYBRID : 65,24 %.

Aucune arène courte ou principale n'a donc été lancée. Cette absence est une décision expérimentale, pas une donnée manquante. Il serait incorrect de conclure que le ranking est insuffisant en jeu : sa force MCTS n'a pas été testée, car sa sécurité offline est insuffisante.

## 10. Interprétation

Le Lot 22 avait correctement identifié un signal utile : pondérer les inversions par strategic gap améliore bien SWI et le regret. Mais la formulation actuelle optimise les corrections sans protéger assez fortement les relations high-gap que G2 possède déjà.

La prochaine reformulation ne doit pas être un sweep de `lambda_rank`. Elle doit rendre la préservation structurelle explicite, par exemple en distinguant dans l'objectif les `CORRECTION_PAIRS` des `PRESERVATION_PAIRS`, avec une contrainte ou un terme conservateur dédié aux secondes.

## 11. Réponses finales

1. **Qdiag est-il assez stable ?** Partiellement : `epsilon=0,02` isole une zone à ≥94,8 % de stabilité, mais Q512 ne couvre que 200 positions.
2. **Quelle zone d'équivalence ?** `|gap| ≤ 0,02`, sans contrainte ranking.
3. **Le réseau apprend-il les grands gaps ?** Oui : les inversions high-gap passent de 44 à 22 avec RANK/HYBRID.
4. **Corrige-t-il sans détruire ?** Non. RANK corrige 55,6 %, mais ne préserve que 64,8 %.
5. **SWI et regret diminuent-ils ?** Oui, nettement sur le test strict.
6. **Ces gains donnent-ils une meilleure force MCTS ?** Inconclusif : aucun candidat ne passe le gate d'arène.
7. **RANK ou HYBRID est-il supérieur à CE ?** RANK est meilleur offline ; aucune comparaison de jeu n'est autorisée.
8. **Existe-t-il un candidat G3 ?** Non.
9. **Prochaine étape unique ?** `RANKING_OBJECTIVE_REFORMULATION`.

### Question scientifique centrale

**INCONCLUSIVE.** Le ranking pondéré est meilleur sur le diagnostic stratégique, mais il n'est pas encore assez conservateur pour être testé comme guide MCTS.

## 12. Verdicts

```text
STRATEGIC_RANKING_IMPLEMENTATION_VALID = YES
QDIAG_UNCERTAINTY_CONTROLLED = PARTIAL
NEAR_EQUIVALENT_ACTIONS_TOLERATED = YES
STRATEGIC_RANKING_SIGNAL_LEARNED = YES
HIGH_GAP_INVERSIONS_REDUCED = YES
G2_STRATEGIC_RELATIONS_PRESERVED = NO
NET_STRATEGIC_GAIN_POSITIVE = YES
VALUE_PRESERVED = YES
RANKING_OBJECTIVE_INSUFFICIENT = INCONCLUSIVE
SEARCH_SCALING_HEALTHY = INCONCLUSIVE
BEST_OFFLINE_POLICY_OBJECTIVE = RANK
BEST_PLAYING_POLICY_OBJECTIVE = NONE
G3_CANDIDATE = NONE
NEXT_ACTION = RANKING_OBJECTIVE_REFORMULATION
```

## 13. Livrables

- script : [`run_srn_lot23.py`](../apps/trainer/scripts/run_srn_lot23.py) ;
- rapport : [`report.json`](../data/experiments/lot23_strategic_ranking/report.json) ;
- manifeste : [`d_rank_manifest.json`](../data/experiments/lot23_strategic_ranking/d_rank_manifest.json) ;
- calibration epsilon : [`epsilon_calibration.json`](../data/experiments/lot23_strategic_ranking/epsilon_calibration.json) ;
- statistiques des poids : [`strategic_weight_statistics.json`](../data/experiments/lot23_strategic_ranking/strategic_weight_statistics.json) ;
- évaluation offline : [`offline_strategic_evaluation.json`](../data/experiments/lot23_strategic_ranking/offline_strategic_evaluation.json) ;
- cas critiques Lot 22 : [`lot22_critical_cases_evaluation.json`](../data/experiments/lot23_strategic_ranking/lot22_critical_cases_evaluation.json) ;
- comparaison : [`candidate_comparison.json`](../data/experiments/lot23_strategic_ranking/candidate_comparison.json).
