# Lot 34 — génération et entraînement contrôlés de G4

## Résultat exécutif

Le Lot 34 a généré et entraîné les deux bras causaux prévus, mais il a été arrêté avant les arènes principales. Aucun des quatre candidats Policy ne satisfait le seuil de préservation stratégique de 85 % pré-enregistré au Lot 33. Continuer vers les grandes arènes aurait violé le protocole et transformé l'expérience en sélection post-hoc.

En conséquence, `CONTROL_G4` et `POOL_G4` ne sont pas des candidats G4 valides. L'effet causal du paradigme de données reste **INCONCLUSIVE**, G4 n'est pas promu et G2 demeure le champion officiel.

## Génération

| Mesure | CONTROL | POOL |
|---|---:|---:|
| Parties | 8 000 | 8 000 |
| Positions brutes | 704 816 | 714 700 |
| États physiques uniques | 659 742 | 668 971 |
| Taux de doublons | 6,395 % | 6,398 % |
| Parties terminales | 7 997 | 7 995 |
| Troncatures | 3 | 5 |
| Longueur moyenne | 88,102 | 89,338 |
| Longueur médiane | 71 | 72 |

CONTROL contient exactement 8 000 parties G2–G2. POOL respecte exactement le ratio 10/10/80 : 800 G2–G2, 800 G3–G3 et 6 400 parties croisées, elles-mêmes équilibrées en 3 200 parties par orientation.

Les audits ont trouvé zéro action illégale, zéro transition illégale, zéro défaut de conservation, zéro masque invalide et zéro défaut de provenance. Les exemples issus des troncatures restent utilisables pour Policy mais sont exclus de Value faute de résultat terminal réel.

## Comparabilité des entraînements

Les deux bras partent de la même Policy `G3_STRATEGIC` et de la même Value `V28_A`; l'écart maximal des paramètres avant l'update 1 est nul. Ils reçoivent chacun 32 000 updates AdamW avec la même configuration, le même batch size, le même clipping et les mêmes sources historique et de réanalyse.

Le batch discret de 256 donne les ratios effectifs identiques suivants :

- nouvelle génération : 179/256 = 69,921875 % ;
- historique RL : 51/256 = 19,921875 % ;
- réanalyse autonome : 26/256 = 10,15625 %.

Cette légère quantification est identique dans les deux bras. Aucun label Teacher ou Minimax n'a été utilisé.

## Métriques hors ligne et short gate

Le short gate a comparé les quatre combinaisons Policy×Value autorisées par bras, sur 64 parties appariées à MCTS64. Les couples provisoirement les mieux classés étaient :

- CONTROL : Policy update 28 000 + Value update 32 000, score court 92,97 % contre G2 ;
- POOL : Policy update 32 000 + Value update 32 000, score court 93,75 % contre G2.

Ces résultats ne constituent ni une promotion ni une preuve causale. La Policy POOL a une CE hors ligne plus faible (1,35884 contre 1,39507), tandis que sa Value est moins bonne sur la MSE de validation (0,54741 contre 0,47519).

## Gate stratégique pré-enregistré

La batterie indépendante est `data/d_scale_v1/d_strategic_sample/qdiag256.jsonl`, figée au Lot 25, contenant 2 000 positions. Son SHA-256 est `e854371afcc162b17e323a062ef0cde96bd7e1041ebe8b794028042b5c0abecc`.

| Bras | Policy | Préservation | Seuil | Passe |
|---|---:|---:|---:|---|
| CONTROL | update 32 000 | 66,47 % | 85 % | non |
| CONTROL | update 28 000 | 68,20 % | 85 % | non |
| POOL | update 32 000 | 69,56 % | 85 % | non |
| POOL | update 28 000 | 70,85 % | 85 % | non |

POOL améliore descriptivement la préservation par rapport à CONTROL, mais reste très loin du seuil obligatoire. Aucun finaliste éligible ne subsiste. Les arènes principales POOL–CONTROL, contre G2 et contre G3 n'ont donc pas été exécutées.

## Décision

```text
EXPERIMENT_VALID = NO
CONTROL_GENERATION_VALID = YES
POOL_GENERATION_VALID = YES
TRAINING_COMPARABILITY_VALID = YES
POLICY_SELECTION_VALID = NO
VALUE_SELECTION_VALID = YES
CONTROL_G4_VALID = NO
POOL_G4_VALID = NO
POOL_G4_BEATS_CONTROL_G4 = INCONCLUSIVE
POOL_DATA_PARADIGM_EFFECT = INCONCLUSIVE
SEARCH_ROBUSTNESS = INCONCLUSIVE
POPULATION_ROBUSTNESS = INCONCLUSIVE
G4_CHAMPION_PROMOTION = NO
G4_GENERATOR_ADMISSION = INCONCLUSIVE
OFFICIAL_CHAMPION = G2
G3_PROMOTED = NO
NEXT_ACTION = LOT34_CONTROLLED_RETRY
```

## Interprétation et suite

Le corpus POOL produit un signal Policy hors ligne légèrement meilleur et une préservation stratégique supérieure à CONTROL, mais le protocole ne permet pas d'en déduire que le cross-play améliore la force de jeu. La condition préalable de sûreté stratégique échoue dans les deux bras.

Le prochain lot doit être un retry explicitement re-préenregistré. Il devra corriger la cause de la destruction des relations stables — par exemple par une stratégie de replay ou un objectif de préservation décidés avant entraînement — sans réutiliser les résultats d'arène, puisqu'aucune arène principale n'a été lancée.

Les données détaillées et le verdict machine-readable se trouvent dans [`data/experiments/lot34_g4_training/report.json`](../data/experiments/lot34_g4_training/report.json).
