# Lot 14 — Génération G3 et benchmark Minimax-Bidoua

## Verdict exécutif

- **Progression relative G3/G2 : catégorie A — aucune progression détectable.**
- **G3_GENERATOR_CANDIDATE = NO.**
- G2 reste le dernier générateur autorisé.
- G2 et G3 sont tous deux très nettement dominés par la référence fixe
  Minimax-Bidoua.

Le passage G2→G3 n'a donc pas reproduit la progression G1→G2. Aucun G4 et
aucune promotion automatique n'ont été réalisés.

## A — Minimax-Bidoua

### 1. Implémentation utilisée

La référence utilise directement :

- `packages/songo_ai/search/negamax.py` ;
- `iterative_deepening` et `negamax_search` ;
- `default_evaluate`, sans réseau ;
- le moteur de règles existant, sans modification.

### 2. Configuration exacte

Nom figé : `MINIMAX_BIDOUA_REFERENCE_V1`.

| Paramètre | Valeur |
|---|---|
| Algorithme | Negamax alpha-bêta |
| Approfondissement | Itératif |
| Profondeur maximale | 14 |
| Nœuds maximaux | 300 000 par décision |
| Temps maximal | 5 s par décision |
| PVS | Actif |
| Fenêtre d'aspiration | 50,0 |
| Table de transposition | Active |
| Quiescence | Désactivée (`0`) |
| Ordonnancement | Coup TT, puis captures immédiates décroissantes |
| Magasin | différence × 10 |
| Territoire sûr | graines des cases à 5+, différence × 3 |
| Mobilité | différence × 1 |
| Terminal | différence finale × 1 000 |

Le bonus territoire est l'approximation historique Bidoua/Yinda du dépôt. Il
n'a pas été modifié.

### 3. Fingerprint

`70f3b5632da10cfa9df4c6f28c19f8c8b8927d514b515d1c31984243d61e334c`

### 4. Justification

Cette configuration correspond au niveau historique fort réellement exposé
par la table : `minimax:14:5:bidoua`. Le benchmark emploie 64 ouvertures
appariées — la baseline minimale autorisée — car la référence consomme souvent
son budget complet de cinq secondes. Les paires ont été parallélisées sans
affaiblir les limites propres à chaque décision.

Minimax n'apparaît dans aucun exemple D_RL et n'a fourni ni action cible, ni
score, ni PV, ni Value d'entraînement.

## B — Baseline G2 contre Minimax-Bidoua

5. **W/D/L :** 2 / 0 / 126 sur 128 vrais terminaux.
6. **Score G2 :** 1,5625 %.
7. **IC 95 % apparié :** [0 %, 3,9063 %].
8. **Par côté :** G2 comme P1 : 2/0/62 ; G2 comme P2 : 0/0/64.
9. **Coût :** Minimax a exploré 73 977 331 nœuds et consommé 9 044,54 s
   cumulées de recherche, soit 577 948 nœuds et 70,66 s par partie. G2/MCTS64
   a consommé 173 504 simulations et 213,06 s cumulées. Longueur moyenne 42,78,
   médiane 39, plage 1–139.

Les nombres de nœuds Minimax peuvent dépasser 300 000 par partie car la limite
s'applique à chaque décision, pas à la partie entière.

## C — Corpus autonome G2→G3

10. **Parties :** 500.
11. **Positions :** 42 876.
12. **Résultats :** P1 260 victoires, P2 222, 17 nulles, 1 répétition
    technique.
13. **Longueurs :** moyenne 85,752, médiane 66, minimum 21, maximum 322.
14. **Qualité Policy :** one-hot 5,089 %, support moyen 4,414, entropie
    1,274, actions légales moyennes 4,415. Tous les comptes de visites totalisent
    64.
15. **S0 :** 500 occurrences, JS 0,02091, accord argmax 0,1423. Les états
    répétés ont une JS moyenne de 0,02100.
16. **Hash :**
    `2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57`.
17. **Gate :** `TRAINING_G3_AUTHORIZED = YES`. Round-trip exact, provenance
    G2→G3 valide, aucune visite illégale et aucune métadonnée Minimax/teacher.

La partie tronquée fournit 212 positions dont `value_target=None`, conformément
au contrat ; les 42 664 autres positions possèdent un vrai label terminal.

## D — Entraînement G3

18. **Split :** 400 parties/35 111 positions train ; 100 parties/7 765
    positions validation ; intersection des `game_id` vide.
19. **Époque 0 — validation :** total 2,1175 ; Policy CE 1,3934 ; KL 0,1095 ;
    top-1 32,97 % ; Value MSE 0,7241 ; MAE 0,7097 ; sign 56,69 %.
20. **Courbe :** amélioration très faible. À l'époque 4, validation totale
    2,1121, Policy CE 1,3908, KL 0,1068, top-1 34,90 %, Value MSE 0,7213,
    MAE 0,7233, sign 54,18 %.
21. **Meilleure époque :** 4.
22. **Early stopping :** oui, époque 9, patience 5.
23. **G3-best :** `training/best_validation_checkpoint.pt`, SHA-256
    `ad7f0449dc646d3c238dd193639bac36628395c9b3daa30e7d87a523620d1da4`.
24. **G3-last :** `training/last_checkpoint.pt`, SHA-256
    `238a4a3c22246f47d3e9527150fc3b76d6aab7df01be482b0cbeaeb078eed41e`.

G3 était exactement égal à G2 avant le premier pas d'optimisation. La lignée,
le hash parent et le hash du corpus sont inscrits dans les deux checkpoints.
La baisse de loss est marginale et la précision de signe Value de la meilleure
époque est inférieure à celle de l'époque 0.

## E — G3 contre G2

### 25. MCTS32

- G3 W/D/L : 113 / 14 / 125 sur 252 terminaux ;
- 4 répétitions techniques ;
- score : 47,62 % ;
- IC 95 % : [41,80 %, 53,32 %] ;
- G3 comme P1 : 49/6/71 ;
- G3 comme P2 : 64/8/54.

### 26. MCTS64

- G3 W/D/L : 105 / 13 / 138 ;
- score : 43,55 % ;
- G3 comme P1 : 49/10/69 ;
- G3 comme P2 : 56/3/69 ;
- 256 vrais terminaux, aucune troncature.

### 27. Incertitude

IC 95 % MCTS64 : **[38,28 %, 48,83 %]**. L'intervalle est entièrement sous
50 %, ce qui fournit un signal défavorable cohérent au budget principal.

### 28. Résultats par côté

G3 ne dépasse pas G2 comme P1. Comme P2, son score reste également sous 50 % à
MCTS64. Le résultat global ne s'explique donc pas par un seul côté du plateau.

## F — G3 contre Minimax-Bidoua

29. **W/D/L :** 2 / 1 / 125.
30. **Score G3 :** 1,9531 %.
31. **IC 95 % :** [0 %, 4,6875 %].
32. **Par côté :** G3 comme P1 : 2/1/61 ; comme P2 : 0/0/64.
33. **Comparaison :** G2 marque 1,5625 %, G3 1,9531 %. Cette différence de
    0,39 point est minuscule, avec des intervalles fortement superposés, et ne
    constitue pas une progression absolue démontrée. Minimax explore 53 439 116
    nœuds et consomme 8 921,42 s cumulées dans cette seconde arène.

La référence, les 64 ouvertures et leurs inversions sont strictement identiques
pour G2 et G3.

## G — Conclusion

34. **Progression relative :** non. G3 perd contre G2, notamment à MCTS64 où
    l'IC 95 % est entièrement sous 50 %.
35. **Catégorie : A.**
36. **G3_GENERATOR_CANDIDATE = NO.**
37. **Position absolue :** G2 et G3 sont très loin derrière
    `MINIMAX_BIDOUA_REFERENCE_V1`. Le cycle autonome améliore la lignée face à
    son parent au passage G1→G2, mais n'a pas encore rejoint l'expert
    algorithmique historique.
38. **Anomalies/limites :** une partie self-play tronquée ; apprentissage G3
    presque plat ; dégradation de la Value sign accuracy à best-validation ;
    quatre répétitions techniques dans l'arène MCTS32 ; benchmark très coûteux
    et limité à 64 ouvertures ; temps cumulés de recherche parallèles supérieurs
    au temps mural.
39. **Tests :** fingerprint de référence, isolation benchmark/D_RL, provenance
    générationnelle, initialisation exacte, split sans fuite, hashes,
    checkpoints, arènes appariées et absence de labels Minimax.
40. **Pytest complet : 388 tests réussis** avant l'expérience ; une dernière
    exécution complète est conservée dans la livraison.
41. **Recommandation :** ne pas produire G4 et ne pas promouvoir G3. Conserver
    G2-best comme générateur autorisé. L'expérience minimale suivante devrait
    réentraîner un candidat G3 depuis G2 sur le même corpus immutable avec une
    seconde seed d'optimisation, sans régénérer les données ni changer les
    hyperparamètres. Cela isolera la variance d'optimisation de la qualité du
    corpus. Si ce second candidat reste inférieur, il faudra alors auditer la
    dérive de la tête Value et la diversité effective du signal G2→G3 avant de
    reprendre la boucle générationnelle.

## Artefacts

- `data/experiments/lot14_g3_seed_20261402/report.json`
- `data/experiments/lot14_g3_seed_20261402/minimax_bidoua_reference_v1.json`
- `data/experiments/lot14_g3_seed_20261402/g2_vs_minimax.json`
- `data/experiments/lot14_g3_seed_20261402/corpus_manifest.json`
- `data/experiments/lot14_g3_seed_20261402/quality_gate.json`
- `data/experiments/lot14_g3_seed_20261402/training/metrics.json`
- `data/experiments/lot14_g3_seed_20261402/training/metrics.csv`
- `data/experiments/lot14_g3_seed_20261402/g3_vs_g2_summaries.json`
- `data/experiments/lot14_g3_seed_20261402/g3_vs_g2_games.jsonl`
- `data/experiments/lot14_g3_seed_20261402/g3_vs_minimax.json`
- `data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl`
