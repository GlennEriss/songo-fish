# Lot46B Stage C0 — plan statistique preregistré

L'unité indépendante est l'`opening_id`; les deux inversions de côté forment un cluster et ne sont jamais traitées comme deux observations indépendantes. Le score primaire vaut 1/0,5/0 pour victoire/nulle/défaite terminale. Les troncatures et parties invalides sont rapportées séparément et exclues, jamais transformées en nulles.

C1 comporte six comparaisons primaires (trois paires, budgets 512 et 4096), chacune fixée à 400 ouvertures, soit 800 parties. L'IC à 95 % est un bootstrap percentile de 10 000 rééchantillonnages par cluster. Les six tests sont corrigés par Holm. Effet minimal interprétable preregistré : 5 points de score.

C2 teste la non-infériorité de `DEEP_POLICY_UNCONSTRAINED+MCTS512` face à `POOL_G4R+MCTS4096`. Marge fixée avant résultats : -0,05. Succès uniquement si la borne basse de l'IC apparié à 95 % de la différence de score dépasse -0,05. Taille fixe : 400 ouvertures/800 parties; aucun arrêt optionnel.

Les ouvertures devront provenir d'un corpus gelé indépendant des positions d'entraînement, avec empreintes physiques dédupliquées contre train/validation/holdout. Les métriques secondaires (W/D/L, durée, nœuds, évaluations réseau, longueur, troncatures) sont descriptives. Policy CE et accord MCTS32768 ne constituent jamais un résultat d'arène.
