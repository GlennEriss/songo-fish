# Évaluation des fuites

Empreinte : JSON ASCII canonique trié de `schema`, `board`, `player_to_move`, puis SHA256. Aucune canonicalisation par symétrie ou inversion de côté.

Pour les 491 états humains : 2 recouvrent `D_RL_LOT11`, 49 recouvrent `LOT45_FINAL_TARGETS`, soit 51 états distincts dans l'union d'entraînement connue; zéro recouvre QDIAG256. Ces 51 états doivent être exclus avant toute sélection.

Le contrôle B3 est `NOT_DETERMINABLE_NO_PHYSICAL_STATE`. La qualification reste donc provisoire. Il faut publier la table physique B3 gelée, ou une table de correspondance vérifiable vers les empreintes, puis refaire l'intersection avant approbation.
