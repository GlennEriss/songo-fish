# Lot 12 — Premier entraînement générationnel G2

## Verdict exécutif

La question centrale reçoit une réponse positive dans le cadre expérimental du
lot : **G2 apprend le nouveau signal autonome et présente une progression
stratégique reproductible face à G1-best**.

Catégorie finale : **C — progression cohérente dans les conditions
expérimentales testées**.

Ce verdict ne constitue pas une promotion automatique. Il repose sur 64
ouvertures appariées et deux budgets MCTS, avec des bornes basses d'IC 95 %
supérieures à 50 % dans les deux conditions.

## A — Génération

1. **Identité du générateur.** G1-best, checkpoint Lot 6 époque 8, SHA-256
   `09784ce81c7d21d005ccd3fced588c6ec59f65a973ef2a5b71ec27489d10bc36`.
2. **Configuration MCTS.** 64 simulations, `c_puct=1.5`, bruit Dirichlet actif,
   `alpha=0.3`, `epsilon=0.25`.
3. **Parties générées.** 500 nouvelles parties ; le pilote Lot 11 n'est pas
   inclus.
4. **Positions.** 42 847 exemples physiques D_RL.
5. **Temps de génération.** 1 051,21 s, dont 1 048,62 s mesurées dans le runner.
6. **Résultats.** P1 : 227 victoires ; P2 : 256 ; nulles : 17 ; troncatures : 0.
7. **Longueurs.** Moyenne 85,694 ; médiane 69 ; minimum 18 ; maximum 335 plies.

La politique de température est inchangée : cible 1,0 ; action 1,0 jusqu'au
ply 30, puis 0. Le corpus conserve les comptes de visites bruts, dont la somme
vaut exactement 64 pour chacun des 42 847 exemples. Les `value_target`
proviennent exclusivement des résultats terminaux moteur.

## B — Qualité du corpus

8. **One-hot.** 5,368 %.
9. **Support moyen.** 4,376 actions.
10. **Entropie moyenne.** 1,270 ; actions légales moyennes : 4,376 ; actions
    visitées moyennes : 4,376.
11. **États répétés.** 318 états répétés, 2 119 occurrences ; JS moyenne
    0,01390 ; accord argmax 0,1475.
12. **S0.** 500 occurrences, JS 0,00971, accord argmax 0,1458.
13. **Comparaison au pilote.** Le pilote avait 3,565 % one-hot, support 4,578,
    entropie 1,338 et JS(S0) 0,00901. Le corpus complet varie modérément, sans
    régression majeure au regard des marges préenregistrées.
14. **Hash/manifest.** SHA-256 du shard :
    `7d5657259cc9771a00b0380f8f30a1277888ad36383e97588faaea52bec474f4`.
    La relecture JSONL est exacte ; la provenance et les paramètres de recherche
    sont présents dans chaque exemple.
15. **TRAINING_AUTHORIZED.** **YES**. Les sept contrôles du gate passent avant
    le premier `optimizer.step`.

## C — Entraînement

16. **Parties.** 400 train, 100 validation.
17. **Positions.** 34 446 train, 8 401 validation.
18. **Hyperparamètres.** AdamW, LR 0,003, weight decay 0,0001, batch 256,
    gradient clipping 1,0, `lambda_policy=lambda_value=1`, maximum 30 époques,
    patience 5, seed 20261200. L'intersection des identifiants de parties train
    et validation est vide.
19. **Époque 0 réelle.** G1-best sans mise à jour :

| Split | Total | Policy CE | Policy KL | Top-1 | Value MSE | Value MAE | Value sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,3904 | 1,3788 | 0,1035 | 28,76 % | 1,0116 | 0,7645 | 57,61 % |
| Validation | 2,3508 | 1,3577 | 0,1077 | 29,73 % | 0,9931 | 0,7688 | 57,68 % |

20. **Courbes.** Les courbes complètes epoch 0–9 sont disponibles en JSON et
    CSV.
21. **Meilleure époque.** 4.
22. **Early stopping.** Oui, à l'époque 9 après cinq époques sans amélioration
    suffisante de la loss de validation.
23. **Best-validation, époque 4.** Validation : total 1,9023 ; Policy CE
    1,3532 ; KL 0,1033 ; top-1 33,34 % ; Value MSE 0,5491 ; MAE 0,5942 ; sign
    65,90 %.
24. **Last, époque 9.** Validation : total 1,9196 ; Policy CE 1,3522 ; KL
    0,1023 ; top-1 35,08 % ; Value MSE 0,5673 ; MAE 0,6059 ; sign 64,64 %.
25. **Temps d'entraînement.** 25,32 s, 1 215 pas d'optimisation.

L'amélioration de la loss provient surtout de la tête Value. La Policy CE
baisse peu, même si son top-1 augmente. Ce constat est un résultat
d'apprentissage, pas à lui seul une preuve de meilleure force de jeu.

## D — Checkpoints

26. **G2-best-validation.** `training/best_validation_checkpoint.pt`, SHA-256
    `eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753`.
27. **G2-last.** `training/last_checkpoint.pt`, SHA-256
    `f0ce81f10fd3d1c6bd89b766106e8387672ac0674bf4cd0494154d7f18080c54`.
28. **Reload exact.** Policy et Value du dernier checkpoint sont bit-à-bit
    identiques avant/après reload ; la lignée est exacte dans best et last.
29. **Parent/provenance.** `generation=G2`, `parent=G1-best`,
    `generator=G1-best`, hash du parent et hash D_RL sont inscrits dans les
    checkpoints. Le parent chargé n'a pas été muté.

Avant l'entraînement, les sorties du modèle initial G2 et de G1-best étaient
exactement égales sur le batch fixe, pour Policy comme pour Value.

## E — Arène G2-best-validation contre G1-best

30. **Ouvertures.** 64 positions indépendantes, chacune jouée dans les deux
    inversions de côté.
31. **Parties.** 128 par budget, 256 au total.
32. **Budget 32.** G2 : 98 victoires, 3 nulles, 25 défaites sur 126 terminaux ;
    score 78,97 %.
33. **Budget 64.** G2 : 91 victoires, 0 nulle, 37 défaites ; score 71,09 %.
34. **G2 en P1.** Budget 32 : 45–2–17 ; budget 64 : 43–0–21.
35. **G2 en P2.** Budget 32 : 53–1–8 sur 62 terminaux, avec 2 répétitions
    techniques ; budget 64 : 48–0–16.
36. **IC 95 % appariés par ouverture.** Budget 32 : [72,27 %, 85,94 %] ;
    budget 64 : [63,28 %, 78,13 %].
37. **Longueurs et troncatures.** Budget 32 : moyenne 82,45, médiane 71,5,
    plage 17–234, 126 terminaux et 2 répétitions. Budget 64 : moyenne 82,55,
    médiane 67, plage 9–265, 128 terminaux et aucune troncature. Les deux
    répétitions techniques sont exclues du dénominateur du score, jamais
    transformées en nulles.

Les empreintes G1 et G2 sont identiques avant et après l'arène. Dirichlet est
désactivé, la température vaut zéro et l'action est l'argmax des visites.

## F — Conclusion

38. **Niveau 1 — apprentissage.** Oui. La validation totale passe de 2,3508 à
    1,9023 et la Value MSE de 0,9931 à 0,5491. La Policy progresse plus
    modestement.
39. **Niveau 2 — changement comportemental.** Oui. Sur 12 positions fixes non
    sélectionnées selon les sorties, l'argmax Policy change dans 9 cas. La JS
    Policy moyenne reste faible (0,00093), signe de nombreux reclassements
    près d'égalités, tandis que l'écart absolu moyen de Value atteint 0,6415.
40. **Niveau 3 — progression stratégique.** Oui dans les deux conditions
    testées : G2 dépasse G1 avec des IC appariés entièrement au-dessus de 50 %.
41. **Catégorie.** **C**.
42. **Anomalies/limites.** Deux répétitions techniques au budget 32 ; légère
    asymétrie P1/P2 dans le corpus ; apprentissage principalement porté par la
    Value ; une seule seed d'entraînement et une seule batterie d'ouvertures.
43. **Tests ajoutés.** Provenance self-play par exemple et lignée persistante
    dans les checkpoints, en plus des contrôles runtime de hash, split,
    initialisation, reload, parent immuable et arène appariée.
44. **Pytest complet.** **383 tests réussis** avant l'expérience ; les tests
    ciblés renforcés passent également.
45. **Recommandation G3.** Ne pas lancer automatiquement G3. Conserver
    G2-best comme successeur expérimental, reproduire d'abord l'arène avec une
    seconde seed d'ouvertures (idéalement 128 ouvertures), puis, si le signal
    persiste, générer un corpus G2→G3 séparé avec G2-best et MCTS64. Ne pas
    fusionner rétroactivement les corpus.

## Artefacts de référence

- `data/experiments/lot12_g2_seed_20261200/report.json`
- `data/experiments/lot12_g2_seed_20261200/corpus_manifest.json`
- `data/experiments/lot12_g2_seed_20261200/quality_gate.json`
- `data/experiments/lot12_g2_seed_20261200/training/metrics.json`
- `data/experiments/lot12_g2_seed_20261200/training/metrics.csv`
- `data/experiments/lot12_g2_seed_20261200/sanity_fixed_positions.json`
- `data/experiments/lot12_g2_seed_20261200/arena_openings.json`
- `data/experiments/lot12_g2_seed_20261200/arena_games.jsonl`
- `data/experiments/lot12_g2_seed_20261200/arena_summaries.json`
- `data/d_rl/lot12_g1_to_g2_mcts64_seed_20261200.jsonl`
