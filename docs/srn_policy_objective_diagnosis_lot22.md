# Lot 22 — Diagnostic de l'objectif Policy

## 1. Objet et intégrité

Le Lot 22 cherche à expliquer pourquoi plusieurs modèles améliorent les mesures Policy hors ligne tout en jouant moins bien que G2. Aucun réseau n'a été entraîné, aucun checkpoint n'a été créé ou modifié, et aucun Teacher ou Minimax n'a été utilisé.

Les modèles analysés sont G2, G3-A, G3-B, C20-DUAL et C21-B. Le fingerprint des cinq réseaux est identique avant et après le diagnostic.

## 2. Batterie diagnostique

La batterie contient 6 001 positions physiques uniques :

- 4 013 positions de validation `D_REANALYSIS`, avec cible autonome MCTS128 ;
- 1 988 positions de la batterie indépendante du Lot 17.

Elle contient 3 021 positions P1 et 2 980 positions P2, tous les branchements de 1 à 7 actions et toutes les plages de graines. Les positions sont stratifiées selon joueur, branchement, graines, magasins, marge et entropie G2, provenance et criticité Lot 17.

Les changements d'argmax sur cette batterie sont nombreux : 2 093 pour G3-A, 1 874 pour G3-B, 1 732 pour C20-DUAL et 1 872 pour C21-B.

## 3. Distribution contre ranking

| Modèle | CE MCTS128 | JS moyenne depuis G2 | L1 moyenne | argmax changé | top-2 changé | Kendall moyen |
|---|---:|---:|---:|---:|---:|---:|
| G3-A | 1,13995 | 0,000599 | 0,04451 | 34,88 % | 37,19 % | 0,500 |
| G3-B | 1,14046 | 0,000250 | 0,02945 | 31,23 % | 32,86 % | 0,585 |
| C20-DUAL | 1,13931 | 0,000492 | 0,04092 | 28,86 % | 31,21 % | 0,599 |
| C21-B | **1,13898** | 0,000694 | 0,04692 | 31,19 % | 31,94 % | 0,562 |

Une distance distributionnelle minuscule coexiste ainsi avec 29 à 35 % de changements d'action dominante. C21-B possède la meilleure CE MCTS128 du groupe, mais reste un joueur rejeté à 35,16 % contre G2. G3-B a la plus faible JS sans être validé en jeu. Les métriques distributionnelles moyennes n'ordonnent donc pas la force connue.

## 4. Référence stratégique autonome

Un sous-échantillon stratifié de 1 200 positions est analysé action par action. Chaque action légale est forcée par le moteur, puis son successeur est recherché par G2/MCTS256 sans Dirichlet. La Value est reconvertie dans la perspective du joueur parent.

Pour une action `a` :

```text
regret(S,a) = max_b Qdiag(S,b) - Qdiag(S,a)
```

Un flip est bénéfique si son delta de regret est inférieur à −0,02, neutre dans `[-0,02 ; 0,02]`, et nuisible au-dessus de 0,02.

Qdiag reste une estimation, pas un oracle. Sur 200 cas critiques, le classement top-1 MCTS256/MCTS512 est stable dans 83 % des cas ; le Kendall moyen est 0,809, mais le changement Q maximal moyen vaut 0,045. Les conclusions sont donc solides au niveau agrégé, avec une incertitude réelle au niveau de certains cas individuels.

## 5. Criticité des flips

| Modèle | flips | bénéfiques | neutres | nuisibles | regret nuisible moyen | maximum |
|---|---:|---:|---:|---:|---:|---:|
| G3-A | 321 | 37,38 % | 28,66 % | 33,96 % | 0,294 | 1,257 |
| G3-B | 252 | 36,51 % | 31,75 % | 31,75 % | 0,253 | 1,072 |
| C20-DUAL | 261 | 37,16 % | 31,42 % | 31,42 % | 0,248 | 1,391 |
| C21-B | 258 | 33,72 % | 31,01 % | **35,27 %** | 0,242 | **1,617** |

Les changements d'argmax ne sont donc pas intrinsèquement mauvais : environ un tiers est bénéfique, un tiers quasi neutre et un tiers nuisible. Le problème est l'incapacité de la CE à distinguer ces classes.

La forme du coût est `MIXED`. Les 10 % pires flips nuisibles concentrent environ 32 à 40 % du coût total, et les 5 % pires environ 19 à 27 %. Il existe donc une queue importante, mais pas une concentration assez extrême pour qualifier tout le phénomène de purement heavy-tail.

## 6. Quelle métrique reconnaît les flips nuisibles ?

| Modèle | AUC JS | AUC L1 | AUC inversions brutes | AUC SWI |
|---|---:|---:|---:|---:|
| G3-A | 0,503 | 0,495 | 0,458 | **0,729** |
| G3-B | 0,495 | 0,498 | 0,447 | **0,720** |
| C20-DUAL | 0,581 | 0,575 | 0,451 | **0,736** |
| C21-B | 0,488 | 0,482 | 0,442 | **0,702** |

La JS moyenne obtient une AUC moyenne de 0,517, proche du hasard. Le nombre brut d'inversions est lui aussi peu informatif. La somme des inversions pondérées par `|Qdiag(a)-Qdiag(b)|`, ou SWI, atteint une AUC moyenne de 0,722.

Ce résultat est central : ce n'est pas le fait d'inverser des actions qui explique le mieux le danger, mais l'inversion d'actions stratégiquement éloignées.

## 7. Hypothèse top-2

La préservation brute du top-2 n'est pas confirmée. Dans les quatre modèles, les flips conservant le même ensemble top-2 présentent même une fraction nuisible supérieure aux flips changeant cet ensemble. Le top-2 ne contient aucune information sur l'écart stratégique entre ses deux actions ; il ne peut donc pas remplacer SWI.

Verdict : `TOP2_PRESERVATION_MATTERS = INCONCLUSIVE`.

## 8. Amplification par la recherche

L'ablation compare `Policy G2 + Value G2` à `Policy C21-B + Value G2`, sur 100 cas critiques, avec MCTS8/32/64/128/256. La Value est strictement la même : seule la Policy varie.

| Budget | JS moyenne des visites | changement d'action après recherche |
|---:|---:|---:|
| 8 | 0,02099 | 23 % |
| 32 | 0,00416 | 14 % |
| 64 | 0,00335 | 8 % |
| 128 | 0,00446 | 9 % |
| 256 | 0,00545 | 10 % |

Vingt-sept occurrences satisfont le critère `SEARCH_AMPLIFIED_POLICY_ERRORS` : faible JS initiale, mais divergence élevée des visites ou changement de l'action MCTS. Davantage de simulations réduit fortement le phénomène initial, sans le faire disparaître : 10 % des cas critiques changent encore d'action à MCTS256.

## 9. Adéquation de l'objectif actuel

L'objectif CE distributionnel est inadéquat comme objectif unique :

- il récompense une amélioration moyenne même lorsque des décisions critiques empirent ;
- une faible JS ne garantit ni la conservation de l'argmax ni celle du ranking ;
- il traite presque de la même manière une inversion entre actions quasi équivalentes et une inversion à grand strategic gap ;
- MCTS peut amplifier une partie de ces petites différences de prior.

Les actions quasi équivalentes expliquent environ 29 à 32 % de flips neutres : leurs inversions ne devraient pas être fortement pénalisées. À l'inverse, SWI montre que les inversions entre actions à grand écart concentrent l'information utile.

## 10. Objectif recommandé pour le Lot 23

La famille recommandée est :

```text
RECOMMENDED_POLICY_OBJECTIVE = STRATEGICALLY_WEIGHTED_RANKING
```

L'expérience suivante devra rester simple : paires d'actions légales, tolérance explicite pour les actions quasi équivalentes et poids croissant avec un strategic gap autonome suffisamment stable. Qdiag ne devra pas être traité comme une vérité exacte. Une petite composante distributionnelle ne devra être ajoutée que comme contrôle si le ranking seul s'avère insuffisant.

Aucune loss de ce type n'a été implémentée ou entraînée dans ce lot.

## 11. Réponses finales

1. **Pourquoi JS=0,000617 peut-elle accompagner une forte perte ?** Parce qu'une moyenne faible masque des changements d'ordre près des frontières de décision : environ 29 % des argmax D_RE changent dans C21-B.
2. **Les changements d'argmax sont-ils associés aux pertes ?** Oui. Environ 31 à 35 % des flips sont nuisibles et certains atteignent un delta de regret de 1,62.
3. **Les changements de top-2 sont-ils plus informatifs que CE/JS ?** Pas sous forme binaire. Le top-2 brut ne sépare pas correctement les flips nuisibles.
4. **Les inversions quasi équivalentes sont-elles généralement sans conséquence ?** Une fraction importante est neutre, mais pas toutes ; il faut une tolérance fondée sur le strategic gap plutôt qu'une exemption globale.
5. **Les grands strategic gaps concentrent-ils le coût ?** Oui. SWI est nettement plus prédictif des flips nuisibles que JS, L1 ou le nombre brut d'inversions.
6. **MCTS amplifie-t-il certaines petites erreurs ?** Oui. L'effet diminue avec le budget, mais persiste dans 10 % des cas critiques à MCTS256.
7. **Quelle famille explique le mieux les modèles rejetés ?** Le ranking pondéré par criticité stratégique, et non la divergence distributionnelle moyenne.
8. **Quel objectif unique tester au Lot 23 ?** Une loss de ranking pondérée par strategic gap.

### Question scientifique centrale

**YES.** Les échecs des quatre candidats sont mieux expliqués par des inversions d'actions stratégiquement éloignées que par leur CE, JS ou L1 moyenne.

## 12. Verdicts

```text
DISTRIBUTIONAL_METRICS_EXPLAIN_FAILURE = NO
ARGMAX_FLIPS_EXPLAIN_FAILURE = YES
TOP2_PRESERVATION_MATTERS = INCONCLUSIVE
PAIRWISE_RANKING_MORE_INFORMATIVE_THAN_CE = YES
STRATEGIC_GAP_MATTERS = YES
SEARCH_AMPLIFIES_POLICY_ERRORS = YES
POLICY_FAILURE_SHAPE = MIXED
CURRENT_POLICY_OBJECTIVE_ADEQUATE = NO
RECOMMENDED_POLICY_OBJECTIVE = STRATEGICALLY_WEIGHTED_RANKING
NEXT_ACTION = CONTROLLED_POLICY_OBJECTIVE_EXPERIMENT
```

## 13. Livrables

- script : [`run_srn_lot22.py`](../apps/trainer/scripts/run_srn_lot22.py) ;
- rapport : [`report.json`](../data/experiments/lot22_policy_objective/report.json) ;
- batterie : [`diagnostic_battery.json`](../data/experiments/lot22_policy_objective/diagnostic_battery.json) ;
- métriques distributionnelles : [`distribution_metrics.json`](../data/experiments/lot22_policy_objective/distribution_metrics.json) ;
- ranking : [`ranking_metrics.json`](../data/experiments/lot22_policy_objective/ranking_metrics.json) ;
- flips : [`argmax_flip_analysis.json`](../data/experiments/lot22_policy_objective/argmax_flip_analysis.json) ;
- regret : [`action_regret.json`](../data/experiments/lot22_policy_objective/action_regret.json) ;
- SWI : [`strategic_weighted_inversions.json`](../data/experiments/lot22_policy_objective/strategic_weighted_inversions.json) ;
- sensibilité MCTS : [`search_sensitivity.json`](../data/experiments/lot22_policy_objective/search_sensitivity.json) ;
- stabilité Qdiag : [`qdiag_stability.json`](../data/experiments/lot22_policy_objective/qdiag_stability.json) ;
- recommandation : [`future_loss_recommendation.json`](../data/experiments/lot22_policy_objective/future_loss_recommendation.json).
