# Lot 36 — Qualification finale de G4 face à Minimax-Bidoua

## Objet et conclusion

Le Lot 36 mesure exclusivement la force externe des deux co-représentants G4
face à `MINIMAX_BIDOUA_REFERENCE_V1`. Il n'a entraîné aucun modèle, produit
aucune donnée d'entraînement et utilisé aucun choix Minimax comme label.

La conclusion est **G4_EXTERNAL_QUALIFICATION = STILL_BEHIND**. G4 a très
fortement réduit la distance historique depuis G2, mais ne bat pas Minimax au
sens du critère pré-enregistré. L'action suivante est
`MINIMAX_GAP_DIAGNOSIS_BEFORE_G5`.

## Intégrité du protocole

- Référence : `MINIMAX_BIDOUA_REFERENCE_V1`.
- Fingerprint observé et attendu :
  `70f3b5632da10cfa9df4c6f28c19f8c8b8927d514b515d1c31984243d61e334c`.
- Configuration inchangée : Negamax alpha-bêta, iterative deepening, PVS,
  aspiration windows, table de transposition, profondeur 14, 300 000 nœuds et
  5 secondes maximum par décision.
- Deux modèles distincts : `CONTROL_G4R` et `POOL_G4R`.
- Les empreintes Policy, Value et architecture ont été vérifiées avant et
  après les arènes et sont restées identiques.
- Six confrontations de 512 parties, soit 3 072 parties.
- Chaque confrontation contient 256 paires d'ouvertures avec inversion des
  côtés : 256 parties G4 en P1 et 256 en P2.
- Les ouvertures et seeds sont communes à CONTROL et POOL pour chaque budget.
- Aucun entraînement, MCTS512, tie-break G4, changement de registre ou travail
  G5 n'a été effectué.

## Résultats principaux

| Modèle | Budget | Score | IC95 apparié | Bat Minimax |
|---|---:|---:|---:|---|
| CONTROL_G4R | 64 | 6,26 % | [4,39 % ; 8,30 %] | Non |
| CONTROL_G4R | 128 | 7,40 % | [5,37 % ; 9,47 %] | Non |
| CONTROL_G4R | 256 | 13,31 % | [10,84 % ; 15,82 %] | Non |
| POOL_G4R | 64 | 6,23 % | [4,30 % ; 8,11 %] | Non |
| POOL_G4R | 128 | 7,48 % | [5,47 % ; 9,57 %] | Non |
| POOL_G4R | 256 | **13,33 %** | **[10,64 % ; 16,02 %]** | Non |

Le critère « bat Minimax » exige simultanément un score supérieur à 50 % et
une borne basse de l'IC95 supérieure à 50 %. Aucun budget ne s'en approche.
La robustesse forte sur deux budgets adjacents est donc également absente.

## Effet du budget de recherche

Le signal est sain et cohérent pour les deux modèles :

- CONTROL : 6,26 % → 7,40 % → 13,31 % ;
- POOL : 6,23 % → 7,48 % → 13,33 %.

Il n'y a pas de search collapse. Le gain le plus marqué apparaît entre 128 et
256 simulations. CONTROL et POOL restent cependant pratiquement confondus ;
le benchmark externe ne résout pas leur identité de champion.

## Côtés et robustesse

À MCTS256, CONTROL marque 39,5/256 comme P1 et 28,5/255 comme P2, soit environ
15,43 % contre 11,18 %. POOL marque 40,5/255 comme P1 et 27,5/255 comme P2,
soit environ 15,88 % contre 10,78 %. Un avantage P1 est visible, mais les deux
modèles restent nettement derrière Minimax des deux côtés. Les rares
troncatures par répétition sont conservées comme résultats techniques et non
transformées arbitrairement en nulles.

## Comparaison historique G2

Le score historique exact disponible pour G2 est 1,5625 % sur 128 parties à
MCTS64 face à la même empreinte Minimax. Le meilleur score G4 est 13,3333 %,
soit un gain indicatif de **+11,7708 points de pourcentage** et un score environ
8,5 fois supérieur. La comparaison reste qualifiée d'indicative parce que la
taille d'arène diffère. Le fossé est substantiellement réduit, mais demeure de
36,67 points par rapport à 50 %.

## Réponses au résumé obligatoire

1. Fingerprint Minimax exact : oui.
2. Configuration historique inchangée : oui.
3. Deux G4 complètement gelés : oui.
4. Parties jouées : 3 072.
5. CONTROL MCTS64 : 6,26 %.
6. CONTROL MCTS128 : 7,40 %.
7. CONTROL MCTS256 : 13,31 %.
8. POOL MCTS64 : 6,23 %.
9. POOL MCTS128 : 7,48 %.
10. POOL MCTS256 : 13,33 %.
11. Meilleur score : POOL MCTS256, 13,33 %.
12. Son IC95 : [10,64 % ; 16,02 %].
13. CONTROL bat statistiquement Minimax : non.
14. POOL bat statistiquement Minimax : non.
15. Supériorité robuste sur deux budgets adjacents : non.
16. Influence du budget : positive, surtout entre 128 et 256.
17. Équilibre des côtés : plan équilibré ; performance meilleure en P1, mais
    inférieure à Minimax dans les deux rôles.
18. Gain absolu maximal sur G2 : +11,77 points, comparaison indicative.
19. Fossé G2→Minimax substantiellement réduit : oui.
20. G4 compétitif avec Minimax : non au sens statistique ; il reste derrière.
21. G4 bat Minimax selon le critère : non.
22. Différence CONTROL/POOL : inconclusive, performances quasi identiques.
23. Entraînement effectué : non.
24. Label Minimax utilisé : non.
25. G5 entraîné : non.
26. `G4_EXTERNAL_QUALIFICATION = STILL_BEHIND`.
27. `NEXT_ACTION = MINIMAX_GAP_DIAGNOSIS_BEFORE_G5`.

## Artefacts

Les résultats complets, parties, analyses par côté, longueurs, causes
terminales, coûts de recherche, seeds et décisions sont sous
`data/experiments/lot36_g4_vs_minimax/`. Le script reproductible est
`apps/trainer/scripts/run_srn_lot36.py`.
