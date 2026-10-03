# Lot 28 — Value under search

## Décision

Le Lot 28 transforme le progrès Policy de G3-STRATEGIC en un candidat jouable en remplaçant sa Value par **V28-A**, une adaptation de la seule tête Value de G2 sur les résultats terminaux réels du grand corpus de self-play.

```text
BEST_POLICY = G3_STRATEGIC
BEST_VALUE = V28_A
G3_CANDIDATE = G3_VALUE_REWORK
NEXT_ACTION = INDEPENDENT_G3_CONFIRMATION
```

Le moteur, la Policy G3 et l'encodeur utilisé par cette Policy sont restés inchangés. L'identité numérique a été contrôlée sur une batterie fixe : `max_abs_policy_logit_delta = 0.0`.

## Expérience zéro : G3-HYBRID

`G3-HYBRID = Policy(G3-STRATEGIC) + Value(G2)`. Les deux sources sont évaluées séparément afin de garantir leurs identités exactes.

| Budget | W/D/L | Score | IC95 apparié | P1 W/D/L | P2 W/D/L | Troncatures | Longueur moyenne |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 125/4/127 | 49,61 % | [44,53 %, 54,49 %] | 58/4/66 | 67/0/61 | 0 | 88,71 |
| 128 | 124/11/118 | 51,19 % | [45,51 %, 57,23 %] | 61/6/59 | 63/5/59 | 3 répétitions | 92,62 |

Conclusion : l'hybride est valide et plus stable en recherche, mais il ne démontre pas à lui seul une supériorité robuste sur G2.

## Diagnostic Value sous recherche

La batterie comprend 400 positions, dont la moitié provient du scaling-failure set du Lot 27. Les recherches réelles à 64, 128 et 256 simulations ont produit 86 621 feuilles physiques uniques.

- désaccord absolu moyen `V_G2 / V_G3` : 0,0542 ;
- désaccord de signe : 3,46 % ;
- bascules 64→128 avec `V_G2` : 38,50 % ;
- bascules 64→128 avec `V_G3` : 50,00 % ;
- bascules 128→256 : 28,25 % avec `V_G2`, 22,50 % avec `V_G3`.

Les 169 cas où le changement de Value entraîne une action racine différente présentent un `|ΔV|` moyen de 0,0394, contre 0,0305 dans les cas stables. Leur `|ΔQ|` moyen vaut 0,0531 contre 0,0440, et leur divergence JS des visites 0,00276 contre 0,00133. Le mécanisme principal est donc une amplification décisionnelle de différences Value modestes dans les positions proches d'une frontière d'action.

La régularisation parent–enfant envisagée pour V28-B n'est pas justifiée : le résidu local moyen de G3 (0,0695) est déjà inférieur à celui de G2 (0,0867), sans discontinuité extrême. V28-B et V28-C n'ont donc pas été créées.

## Données et entraînement V28-A

Le stockage `D_SELFPLAY_LARGE` contient 441 563 positions. Parmi elles, 1 334 proviennent de parties tronquées et n'ont pas de résultat terminal. Elles ont été exclues, jamais converties artificiellement en nul. Le corpus utilisable contient donc 440 229 vrais labels `z`, répartis par `game_id` en 393 660 positions d'entraînement et 46 569 de validation.

Aucun label Teacher, Minimax, root-value ou Q de recherche n'est utilisé. Le modèle Value est initialisé depuis G2. Tous ses paramètres sont gelés sauf `value_mlp`; la Policy réellement jouée provient toujours du checkpoint G3-STRATEGIC indépendant.

| Epoch | MSE | MAE | Sign accuracy |
|---:|---:|---:|---:|
| 0 (G2) | 0,64738 | 0,67785 | 68,90 % |
| 1 | 0,63762 | 0,66963 | 69,93 % |
| 2 retenu | **0,63662** | **0,66649** | **70,14 %** |
| 3 | 0,63830 | 0,66955 | 70,00 % |

## Mini-search

| Value associée à Policy G3 | Flip 64→128 | Flip 128→256 |
|---|---:|---:|
| V_G3 | 50,00 % | 22,50 % |
| V_G2 | 38,50 % | 28,25 % |
| V28-A | **37,25 %** | **25,25 %** |

V28-A améliore la transition critique 64→128 par rapport à V_G2 et V_G3, sans optimiser une Value constante : sa MSE, sa variance et son accuracy de signe restent non dégénérées.

## Arènes finales V28-A contre G2

Dirichlet est désactivé, la température vaut 0, `c_puct = 1.5`, les côtés sont inversés et l'IC95 utilise 20 000 réplications bootstrap par paire d'ouverture.

| Budget | Parties | W/D/L | Score | IC95 apparié | P1 W/D/L | P2 W/D/L | Troncatures | Longueur moyenne |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 256 | 136/10/107 | **55,73 %** | [50,00 %, 60,74 %] | 68/6/54 | 68/4/53 | 3 répétitions | 85,89 |
| 128 | 256 | 128/11/116 | **52,35 %** | [47,27 %, 57,62 %] | 65/4/59 | 63/7/57 | 1 répétition | 88,01 |
| 256 | 128 | 67/10/51 | **56,25 %** | [49,22 %, 63,28 %] | 30/6/28 | 37/4/23 | 0 | 100,43 |

Le signal est cohérent aux trois budgets et ne présente pas d'effondrement à 256. Les intervalles à 128 et 256 recouvrent toutefois 50 % : le résultat constitue un **candidat G3**, pas encore une promotion définitive. La prochaine action est une confirmation indépendante avec nouvelles ouvertures et nouvelle seed.

## Verdicts obligatoires

```text
HYBRID_VALID = YES
HYBRID_BEATS_G2 = NO
G3_POLICY_PRESERVED = YES
VALUE_G2_MORE_SEARCH_STABLE_THAN_VALUE_G3 = YES
VALUE_RETRAINING_VALID = YES
VALUE_SEARCH_STABILITY_IMPROVED = YES
VALUE_REWORK_BEATS_HYBRID = YES
VALUE_REWORK_INSUFFICIENT = NO
SEARCH_SCALING_HEALTHY = YES
BEST_POLICY = G3_STRATEGIC
BEST_VALUE = V28_A
G3_CANDIDATE = G3_VALUE_REWORK
NEXT_ACTION = INDEPENDENT_G3_CONFIRMATION
```

## Réponses finales

1. L'hybride obtient 49,61 % à MCTS64 et 51,19 % à MCTS128 : il ne bat pas clairement G2.
2. V_G2 est plus stable que V_G3 sur la transition critique 64→128.
3. Les bascules supplémentaires proviennent de différences Value modestes qui modifient Q et les visites dans les positions proches d'une frontière de décision, et non d'une mauvaise cohérence parent–enfant générale.
4. Une seule nouvelle Value a été entraînée.
5. V28-A dépasse V_G2 en stabilité mini-search et en force d'arène avec Policy G3 gelée.
6. La meilleure combinaison est `P_G3_STRATEGIC + V28_A`.
7. Elle bat G2 à MCTS64 avec un score de 55,73 %.
8. Elle est devant G2 à MCTS128 avec 52,35 %, mais l'IC95 recouvre 50 %.
9. Le scaling reste acceptable à MCTS256 : 56,25 % sur 128 parties.
10. `G3_CANDIDATE = G3_VALUE_REWORK`.
11. `NEXT_ACTION = INDEPENDENT_G3_CONFIRMATION`.

La réponse à la question scientifique est donc **oui, provisoirement** : corriger l'interaction Value–MCTS transforme le progrès Policy en progression cohérente de force aux trois budgets. Nous avons désormais un véritable candidat G3, qui doit encore être confirmé indépendamment avant promotion officielle.
