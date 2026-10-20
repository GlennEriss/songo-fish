# Rapport final C0.4-P5

Les octets originaux ont été retrouvés dans Git et vérifiés contre le SHA-256 historique. Le membre gzip dérivé a été distingué de l'archive externe et du JSONL décompressé, puis vérifié. Les preuves publiées P3/P4 soutiennent 491 états uniques, 51 exclusions d'entraînement, 440 candidats reproductibles et zéro recouvrement exact avec les 994 états B3 raccordés.

Le contrat `SHA256_RANK_V1` spécifie désormais les encodages, le préimage, le tri, les départages, l'éligibilité et le gel. L'amendement exact a le SHA-256 `2e4dae3b58beeee4f15b17c72cf2c000729f3ae49e44147cc0a2601b29788662`. La recommandation est `CONDITIONAL_ACCEPT`; elle n'est pas une approbation. Toutes les huit décisions restent `PENDING`.

Le diagnostic P4 classe l'unique échec historique (simulation de terminaison d'un worker Lot45) comme course d'ordonnancement du test, sans incidence établie sur l'intégrité des ouvertures. Il n'a pas motivé de modification de logique de production.

Tests exécutés : `35 passed in 1.35s` sur les modules ciblés. La première suite complète a donné `809 passed in 174.97s`; trois tests de preuve ciblés ont ensuite été ajoutés et passent dans le lot ciblé. L'échec historique Lot45 ne s'est pas reproduit.

Aucun jeu, MCTS, entraînement, checkpoint, registre final, C0.4, C1 ou C2 n'a été exécuté. Prochaine action : demander une approbation scientifique conditionnelle explicite liée au hash de l'amendement.
