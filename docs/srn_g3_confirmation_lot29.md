# Lot 29 — Confirmation indépendante de G3

## Décision finale

Le candidat immuable `Policy G3_STRATEGIC + Value V28_A` a reproduit un avantage directionnel contre G2 aux trois budgets. Il **n'est cependant pas promu**, car la baisse pré-enregistrée maximale autorisée entre MCTS64 et MCTS128 était de 5 points, tandis que la baisse observée est de **5,57 points**.

```text
G3_PROMOTED = NO
AUTHORIZED_GENERATOR = G2
NEXT_ACTION = REASSESS_MODEL_LEARNING_PARADIGM
```

Cette décision ne remet pas en cause le signal positif du candidat. Elle applique sans modification post-hoc la règle fixée avant les parties.

## Validité expérimentale

- checkpoint Policy : `g3_strategic_best.pt`, SHA-256 `99c3bea25d41daae2df405477c2b6d57811ae6aaecb35485c80a9a84eff2af05` ;
- checkpoint Value : `v28_a_best.pt`, SHA-256 `1e5175b312b81f4a62e973e1e2effea46eee7f2cb6ba2e6d2f42d4a0ac397ed6` ;
- checkpoint G2 : SHA-256 `eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753` ;
- différence maximale des logits Policy : `0.0` ;
- différence maximale de Value par rapport à V28-A : `0.0` ;
- seeds et ouvertures indépendantes du Lot 28 ;
- aucun entraînement, changement d'architecture ou changement moteur ;
- Dirichlet désactivé, température 0, `c_puct = 1.5` ;
- 20 000 réplications bootstrap par paire d'ouverture.

## Résultats indépendants

Le score protocolaire est `(W + 0,5 × D) / games`. Les troncatures restent donc dans le dénominateur.

| Budget | Parties | W/D/L | Troncatures | Score | IC95 apparié | P1 | P2 | Longueur moyenne/médiane |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MCTS64 | 512 | 279/18/213 | 2 | **56,25 %** | [52,44 %, 60,16 %] | 56,05 % | 56,45 % | 86,97 / 83 |
| MCTS128 | 512 | 251/17/242 | 2 | **50,68 %** | [46,88 %, 54,49 %] | 52,34 % | 49,02 % | 87,26 / 86 |
| MCTS256 | 256 | 139/9/106 | 2 | **56,05 %** | [50,98 %, 61,13 %] | 57,42 % | 54,69 % | 85,33 / 80,5 |

Le résultat combiné MCTS64 + MCTS128 porte sur 1 024 parties : 530 victoires, 35 nulles, 455 défaites et 4 troncatures. Le score est de **53,47 %**, IC95 apparié **[50,73 %, 56,20 %]**.

## Côtés et scaling

Les écarts P1/P2 restent faibles :

- MCTS64 : 0,39 point ;
- MCTS128 : 3,32 points ;
- MCTS256 : 2,73 points.

Ils sont très inférieurs au seuil pré-enregistré de 15 points. Il n'existe donc pas d'anomalie majeure de côté.

Le scaling observé est :

```text
64 → 128 : -5,5664 points
128 → 256 : +5,3711 points
```

La remontée à MCTS256 est saine. Toutefois, la première baisse dépasse de 0,5664 point la tolérance maximale de 5 points fixée avant résultat. Le critère global de scaling échoue donc mécaniquement.

## Comparaison avec le Lot 28

| Budget | Lot 28 | Lot 29 | Delta |
|---:|---:|---:|---:|
| 64 | 55,73 % | 56,25 % | +0,52 point environ |
| 128 | 52,35 % | 50,68 % | −1,67 point environ |
| 256 | 56,25 % | 56,05 % | −0,20 point environ |

La direction positive est reproduite aux trois budgets, avec une preuve statistique nette à 64, au combiné 64+128 et à 256. Le budget 128 reste incertain et la baisse 64→128 dépasse le seuil pré-engagé. L'effet du Lot 28 est donc classé **PARTIAL**, et non `YES`.

## Verdicts obligatoires

```text
CONFIRMATION_VALID = YES
CHECKPOINT_IMMUTABLE = YES
INDEPENDENT_SEEDS = YES
MCTS64_ADVANTAGE_REPLICATED = YES
MCTS128_ADVANTAGE_REPLICATED = YES
MCTS256_SCALING = HEALTHY
SIDE_BALANCE_ACCEPTABLE = YES
LOT28_EFFECT_REPLICATED = PARTIAL
SEARCH_SCALING_HEALTHY = NO

G3_PROMOTED = NO
AUTHORIZED_GENERATOR = G2
NEXT_ACTION = REASSESS_MODEL_LEARNING_PARADIGM
```

`MCTS128_ADVANTAGE_REPLICATED = YES` signifie que la direction du score reste supérieure à 50 %. Son IC95 recouvre néanmoins 50 %, ce qui rend l'avantage à ce budget incertain.

## Résumé final

1. **1 280 parties indépendantes** ont été jouées : 512 à MCTS64, 512 à MCTS128 et 256 à MCTS256.
2. À MCTS64 : 279/18/213, score 56,25 %, IC95 [52,44 %, 60,16 %].
3. À MCTS128 : 251/17/242, score 50,68 %, IC95 [46,88 %, 54,49 %].
4. À MCTS256 : 139/9/106, score 56,05 %, IC95 [50,98 %, 61,13 %].
5. Oui, les résultats sont équilibrés entre P1 et P2 ; l'écart maximal est de 3,32 points.
6. L'effet du Lot 28 est reproduit partiellement : direction positive partout, mais baisse 64→128 légèrement trop forte.
7. Non au sens de la règle globale pré-enregistrée ; le scaling 128→256 est néanmoins sain.
8. L'avantage est robuste à 64, au combiné 64+128 et à 256 ; il reste incertain à 128.
9. `G3_PROMOTED = NO`.
10. `AUTHORIZED_GENERATOR = G2`.
11. `NEXT_ACTION = REASSESS_MODEL_LEARNING_PARADIGM`.
