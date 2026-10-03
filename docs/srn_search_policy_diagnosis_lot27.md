# Lot 27 — Search–Policy Interaction Diagnosis

## Objet

Ce lot explique pourquoi G3-STRATEGIC passe de 62,55 % contre G2 à MCTS64 à 54,90 % à MCTS128. Aucun entraînement, checkpoint, changement moteur ou changement d'architecture n'a été effectué.

La batterie principale contient 2 000 positions uniques reconstruites depuis les trajectoires indépendantes des arènes du Lot 26. G2, G3-SCALE et G3-STRATEGIC sont exécutés avec une trace déterministe MCTS256, Dirichlet désactivé et `c_puct=1.5`. Les points 8, 16, 32, 64, 128 et 256 sont extraits de cette même recherche. Une batterie profonde de 400 positions couvre 200 cas de failure set et 200 contrôles.

## Priors et divergence initiale

Les Policies G2 et G3-STRATEGIC sont très proches en distribution : JS moyenne 0,000096, L1 moyenne 0,0189, accord argmax 91,55 % et accord top-2 94,58 %. Leur première divergence de sélection racine apparaît néanmoins tôt : médiane simulation 9, moyenne 12,88.

Il n'existe pas de changement global de calibration expliquant le phénomène. L'entropie moyenne vaut 1,25693 pour G2 et 1,25670 pour G3-STRATEGIC; la probabilité top-1 moyenne vaut respectivement 0,3443 et 0,3433. Support effectif, masse top-2 et masse de queue sont également comparables.

## Évolution de la recherche

| Budget | Accord action G2/G3-STRATEGIC | JS moyenne des visites |
|---:|---:|---:|
| 8 | 87,05 % | 0,00973 |
| 16 | 85,15 % | 0,00415 |
| 32 | 83,90 % | 0,00195 |
| 64 | 84,20 % | 0,00189 |
| 128 | 85,60 % | 0,00227 |
| 256 | 88,70 % | 0,00322 |

La recherche ne produit pas une amplification monotone des erreurs G3. Après le maximum de désaccord vers 32–64 simulations, les actions convergent de nouveau. Les différences Policy corrigées par la recherche (catégorie B) passent de 133 à MCTS64 à 148 à MCTS256, tandis que les différences persistantes Policy+MCTS (catégorie D) passent de 36 à 21. Cela soutient `SEARCH_CORRECTS_G2=YES`.

## Masse de prior pondérée par le regret

RWPM moyen sur les 2 000 positions Qdiag256 :

- G2 : 0,03836;
- G3-SCALE : 0,03891;
- G3-STRATEGIC : 0,03813.

G3-STRATEGIC présente également une masse moyenne sur les actions de regret supérieur à 0,1 légèrement plus faible que G2 : 0,07738 contre 0,07814. Ses P95/P99 et son maximum RWPM sont eux aussi légèrement meilleurs. Le recul à budget élevé n'est donc pas expliqué par un excès global de masse sur des actions à fort regret.

## Ablations racine, descendants et Value

Sur 400 positions instrumentées :

| Configuration | Accord avec G2 @64 | Accord avec G2 @128 | Bascule 64→128 |
|---|---:|---:|---:|
| G3 Policy racine seule, descendants/Value G2 | 87,50 % | 92,00 % | 40,25 % |
| G3 Policy descendants seuls, racine/Value G2 | 91,25 % | 90,50 % | 36,50 % |
| G3 Policy complète + Value G2 | 82,75 % | 87,25 % | 38,50 % |
| G3 complet Policy + Value | 77,50 % | 79,50 % | 50,00 % |

L'effet racine est progressivement corrigé par davantage de recherche. L'effet descendant persiste davantage. Mais l'observation principale est l'augmentation des bascules 64→128 de 38,50 % à 50,00 % lorsque la Value G3 remplace la Value G2. À Policy G3 identique, les versions Value G2 et Value G3 ne choisissent la même action que dans 83,50 % des cas à MCTS64 et 85,75 % à MCTS128.

La localisation est donc `BOTH` pour la Policy, avec une interaction Value dominante. La divergence ne vient pas d'une Value catastrophique offline — elle était préservée au Lot 26 — mais de son interaction répétée avec les feuilles et la sélection PUCT.

## Distribution des arbres

Le Jaccard des feuilles visitées vaut 0,802 entre G2 et G3-STRATEGIC, contre 0,687 entre G2 et G3-SCALE. Le déplacement propre à G3-STRATEGIC reste mesurable mais ne franchit pas le seuil retenu pour déclarer un changement auto-induit majeur. `SELF_INDUCED_SEARCH_SHIFT=NO`.

## Explication du scaling

L'avantage MCTS64 vient d'un petit déplacement utile des priors/rankings. Lorsque le budget augmente, G2 compense une partie de ce désavantage initial : les actions G2/G3 convergent de 84,20 % à MCTS64 vers 88,70 % à MCTS256. G3-STRATEGIC ne se dégrade pas principalement par masse de regret ou mauvaise calibration.

Cependant, la Value G3-STRATEGIC perturbe davantage la trajectoire de recherche à mesure que les feuilles s'accumulent. L'ablation avec Value G2 réduit nettement les bascules de budget et rapproche la recherche de G2. La cause principale mesurée est donc `VALUE_INTERACTION`, avec `SEARCH_CORRECTS_G2` et `DESCENDANT_POLICY_ACCUMULATION` comme causes secondaires.

## Verdicts

```text
SEARCH_POLICY_DIAGNOSIS_VALID = YES
ROOT_PRIOR_ADVANTAGE = YES
PRIOR_MASS_PROBLEM = NO
EARLY_PRIOR_LOCKIN = NO
SEARCH_CORRECTS_G2 = YES
SEARCH_AMPLIFIES_G3_ERRORS = NO
DESCENDANT_POLICY_ACCUMULATION = YES
VALUE_INTERACTION_PROBLEM = YES
POLICY_CALIBRATION_SHIFT = NO
POLICY_CALIBRATION_SENSITIVITY = INCONCLUSIVE
SELF_INDUCED_SEARCH_SHIFT = NO
STRATEGIC_RANKING_MISSES_PRIOR_MASS_EFFECT = NO
POLICY_DAMAGE_LOCATION = BOTH
PRIMARY_CAUSE = VALUE_INTERACTION
SECONDARY_CAUSES = SEARCH_CORRECTS_G2, DESCENDANT_POLICY_ACCUMULATION
G3_CANDIDATE = NONE
NEXT_ACTION = POLICY_VALUE_INTERACTION_REWORK
```

## Réponses finales

1. G2 et G3-STRATEGIC divergent typiquement dès la simulation 9 à la racine.
2. Les différences augmentent jusqu'à 32–64 simulations, puis sont partiellement réduites; elles ne sont pas amplifiées monotoniquement.
3. Non. G3-STRATEGIC a un RWPM et une masse high-regret légèrement inférieurs à G2.
4. La Policy agit à la racine et dans les descendants; l'effet racine est davantage corrigé, l'effet descendant persiste.
5. Oui. La Value G3 fait passer le taux de bascule 64→128 de 38,50 % à 50,00 % par rapport à la même Policy avec Value G2.
6. Elle conduit à des arbres différents, mais pas suffisamment pour déclarer un déplacement auto-induit majeur selon le seuil retenu.
7. Les deux mécanismes existent, mais les mesures montrent surtout G2 qui récupère avec davantage de recherche, combiné à une interaction Value G3 défavorable.
8. Cause principale : `VALUE_INTERACTION`.
9. Tester séparément la robustesse et la calibration de la Value sous distribution de feuilles MCTS, en conservant la Policy stratégique.
10. `NEXT_ACTION = POLICY_VALUE_INTERACTION_REWORK`.
