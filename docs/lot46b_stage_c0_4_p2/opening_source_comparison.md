# Comparaison

| Source | États complets | Relation | Verdict |
|---|---:|---|---|
| D_RL_LOT11 | 4 123 | entraînement direct | exclue |
| QDIAG256 | 2 000 | validation stratégique | exclue |
| LOT45_SELECTION complet | 100 000 | mélange majoritairement entraînement | exclu comme ensemble |
| LOT45_D_REAL_HUMAN_SUBSET | 491 | humain, 51 chevauchements connus | recommandé provisoirement après exclusions |
| LOT45_FINAL_TARGETS | 10 000 | cibles MCTS d'entraînement | exclue |
| B3_PAIRED_POSITIONS | 0 reconstructible | validation dépendante des candidats | exclue |

Le sous-ensemble humain est le seul candidat dépassant potentiellement 400 états sans être intrinsèquement une source d'entraînement ou de validation. Cette recommandation n'est ni une sélection finale ni une approbation.
