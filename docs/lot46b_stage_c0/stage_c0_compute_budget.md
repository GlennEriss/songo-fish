# Lot46B Stage C0 — budget de calcul

Aucun benchmark MCTS n'a été exécuté en C0. Le débit T4 pour MCTS512/MCTS4096 est donc **UNKNOWN**, et aucune durée précise n'est affirmée.

Pour une comparaison de 400 ouvertures : 800 parties. C1 totalise 6 comparaisons, donc 4 800 parties. C2 ajoute 800 parties. Le coût doit être calculé après un benchmark séparément autorisé : `GPU_seconds = sum(game_plies × measured_seconds_per_decision(candidate, budget, T4))`. Stockage : somme des traces compressées, manifests et journaux; à mesurer sur un pilote autorisé.

Avant C1, réserver avec marge la durée calculée sur débit mesuré, figer une taille fixe, et prévoir une reprise par identifiant `(experiment_id, comparison_id, opening_id, side)` afin d'éviter tout double comptage.
