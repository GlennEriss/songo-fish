# Rapport final C0.4-P4

Le dérivé permet de reproduire exactement 491 états, 51 exclusions d’entraînement, 440 candidats et zéro intersection B3 après récupération exhaustive des 994 états. La recherche bornée a retrouvé la source originale suivie par Git; ses 586283 octets correspondent exactement au SHA-256 historique `17d9…ed18`. L’identité des octets et l’intégrité du dérivé sont donc vérifiées; l’origine humaine externe n’est pas certifiée par le hash seul.

La recommandation scientifique est `CONDITIONAL_ACCEPTANCE`, accompagnée du gel des hashes, empreintes candidates, règles, seed et procédure avant sélection, et de la divulgation permanente de la provenance partielle. `AMENDMENT_APPROVAL_READY = YES` signifie seulement que le paquet est prêt à être soumis; source, risque, seed et amendement restent non approuvés.

Le test Lot45 échoué dans la suite complète ne se reproduit pas en deux essais isolés. Le code montre une course d’ordonnancement dans le scénario de test. Il reste séparé de la décision scientifique.

Arrêt contrôlé : aucun registre final, choix des 400, jeu, MCTS, entraînement, C0.4, C1, C2 ou promotion.
