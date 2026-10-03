# Lot 17 — Ranking, regret et importance stratégique de la Policy

## Verdict exécutif

- **POLICY_RANKING_REGRESSION = YES** — des changements G2→G3 stables
  augmentent fortement le regret sur une minorité de positions.
- **RARE_HIGH_REGRET_ERRORS = YES** — les 100 régressions les plus coûteuses,
  soit 5 % de l'échantillon, concentrent plus de 81 % du coût régressif.
- **POLICY_CE_MISALIGNED_WITH_PLAY_STRENGTH = YES** — la CE moyenne baisse
  alors que la Policy isolée perd en arène, et CE améliorée coexiste localement
  avec regret aggravé.
- **MCTS_TARGET_NOISE_PROBLEM = YES** — sur les cas critiques, les cibles
  MCTS64 avec bruit Dirichlet n'ont que 70,6 % d'accord argmax inter-seeds.
- **OUT_OF_DISTRIBUTION_POLICY_PROBLEM = INCONCLUSIVE** — les changements de
  ranking existent sur `D_REAL`, mais aucun coup humain n'est utilisé comme
  vérité et aucun regret externe n'a été calculé.
- **NEXT_EXPERIMENT = REGRET_WEIGHTED_POLICY_LOSS** — à tester dans un lot
  ultérieur, sans l'implémenter ici.

## 1. Intégrité et méthode

Le corpus immutable G2→G3 est vérifié par son SHA-256 :
`2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57`.
G2-best, G3-A-best et G3-B-best sont utilisés en lecture seule ; leurs
empreintes sont identiques avant et après le diagnostic. Aucun entraînement,
checkpoint, self-play ou corpus RL n'a été créé.

Les Policies sont extraites sur les 42 876 positions. Le regret est mesuré sur
2 000 positions sélectionnées reproductiblement avec la seed 20261717 et
stratifiées par ply (0–30, 31–90, 91+), nombre d'actions légales et présence
d'un changement d'argmax G2→G3.

### Définition de Q_search

Pour chaque action légale `a` :

1. le moteur applique `a` ;
2. si l'état est terminal, sa valeur exacte est utilisée ;
3. sinon, une recherche indépendante G2/MCTS128, sans Dirichlet, évalue
   l'état successeur ;
4. la valeur est reconvertie dans la perspective du joueur au trait en `S`.

Chaque action reçoit le même budget. Ainsi :

`regret(S,a) = max_b Q_search(S,b) - Q_search(S,a)`.

Cette grandeur reste une estimation G2/MCTS128, pas une valeur exacte ni une
évaluation Minimax.

## 2. Changements de ranking sur tout le corpus

Seuils de séparation MCTS documentés : faible `< 0,10`, moyenne
`[0,10 ; 0,25)`, forte `≥ 0,25`, avec la marge
`π_top1 - π_top2`.

| Catégorie | G3-A | G3-B |
|---|---:|---:|
| Même top-1 que G2 | 29 507 (68,82 %) | 30 806 (71,85 %) |
| Changement vers top-1 MCTS | 4 209 | 3 639 |
| Abandon du top-1 MCTS | 3 323 | 2 936 |
| Ni G2 ni G3 au top-1 MCTS | 5 837 | 5 495 |

Les changements ne sont donc pas univoquement mauvais : les passages vers le
top-1 MCTS sont plus nombreux que les abandons. G3-A augmente en moyenne de
0,00352 la probabilité de l'action préférée MCTS ; G3-B de 0,00184. Les
changements de marge sont également modestes, respectivement +0,01255 et
+0,00428.

## 3. Regret ponctuel

| Policy choisissant l'action | Moyenne | Médiane | P75 | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| G2 | 0,08892 | 0,02287 | 0,10697 | 0,25820 | 0,39197 | 1,34756 |
| G3-A | 0,07198 | 0,01091 | 0,07480 | 0,21565 | 0,36203 | 1,34756 |
| G3-B | 0,07214 | 0,01147 | 0,07812 | 0,22137 | 0,36754 | 1,34756 |
| π_MCTS64 | 0,02185 | 0 | 0,02336 | 0,06715 | 0,10678 | 1,01834 |

Sur cette mesure ponctuelle, G3-A et G3-B sont meilleurs que G2 en moyenne.
Ce résultat interdit une conclusion simpliste selon laquelle « G3 choisit
globalement de moins bonnes actions ». Il coexiste toutefois avec des erreurs
locales importantes et avec la défaite des Policies G3 en arène séquentielle.

### Changements G2→G3

| Mesure | G3-A | G3-B |
|---|---:|---:|
| Changements d'action | 849 | 749 |
| Améliorations de regret | 523 | 458 |
| Régressions de regret | 325 | 290 |
| Δ regret moyen, toutes positions | −0,01693 | −0,01677 |
| Coût moyen parmi les régressions | +0,08412 | +0,06461 |
| Coût médian parmi les régressions | +0,03278 | — |
| Régression maximale | +1,31892 | +0,71856 |

Les améliorations sont plus nombreuses et plus fortes en moyenne. Le défaut
n'est donc pas une majorité de mauvais changements, mais la présence d'une
queue de régressions coûteuses pouvant être amplifiées au cours d'une partie.

## 4. Concentration des erreurs coûteuses

| Part supérieure des régressions | Part du coût G3-A | Part du coût G3-B |
|---|---:|---:|
| Top 10 cas | 27,17 % | 20,75 % |
| Top 50 cas | 61,79 % | 59,10 % |
| Top 100 cas | 82,11 % | 81,43 % |

Les 100 cas représentent seulement 5 % des 2 000 positions diagnostiquées,
mais plus de quatre cinquièmes du regret positif cumulé. Cela justifie
`RARE_HIGH_REGRET_ERRORS = YES`, même si les régressions au sens large ne sont
pas rarissimes (16,25 % et 14,50 % de l'échantillon).

## 5. Cross-entropy contre importance stratégique

G3-A réduit la CE moyenne de 0,00294 et G3-B de 0,00203 sur l'échantillon.
Leur regret moyen baisse également, mais l'alignement position par position
est imparfait :

- G3-A : 121 positions (6,05 %) ont une CE améliorée et un regret aggravé ;
- G3-B : 110 positions (5,50 %) ont le même conflit.

La corrélation entre ΔCE et Δregret est seulement modérée : Pearson 0,313 / 
Spearman 0,218 pour G3-A, et 0,304 / 0,213 pour G3-B. L'accord argmax avec
MCTS est plus informatif sur le regret, avec Spearman −0,480 et −0,490.

Le constat complet est donc : la CE fournit un signal moyen utile, mais elle
pondère uniformément des positions dont le coût décisionnel diffère fortement.
Elle peut s'améliorer tout en dégradant certaines décisions importantes et,
comme établi au Lot 16, tout en réduisant la force de jeu de la Policy isolée.

**La réponse à la question clé sur la loss est YES.**

## 6. Confiance MCTS, branching, ply et actions

La fréquence des régressions diminue lorsque la cible MCTS est fortement
séparée :

| Marge MCTS | G3-A | G3-B |
|---|---:|---:|
| Faible | 19,06 % | 17,75 % |
| Moyenne | 17,82 % | 15,93 % |
| Forte | 11,00 % | 8,83 % |

Les cas incertains sont donc plus exposés. Les forced moves ont naturellement
zéro regret. Pour 2 à 7 actions légales, aucune croissance monotone simple
n'apparaît. Les trois plages de ply présentent entre 13 % et 17 % de
régressions, sans concentration suffisante pour désigner une plage unique.

L'action locale 1 concentre la fréquence la plus élevée : 24,1 % pour G3-A et
22,4 % pour G3-B. Cela décrit une association empirique ; ce n'est pas une
preuve qu'une case physique particulière ou une règle Songo est défectueuse.

## 7. Stabilité de recherche

Les 100 cas critiques ont été réévalués avec cinq seeds.

- recherche diagnostique sans bruit : accord de la meilleure action 100 % et
  écart-type Q nul ; le regret mesuré est déterministe dans ce protocole ;
- reproduction des cibles self-play G2/MCTS64 avec Dirichlet : divergence JS
  inter-seeds moyenne 0,02084, médiane 0,01976, P95 0,04505 ; accord argmax
  moyen seulement **70,6 %**.

Le premier résultat exclut un artefact de tie-breaking du diagnostic. Le
second montre qu'une cible MCTS64 bruitée unique peut changer d'action préférée
sur près de trois paires de runs sur dix dans ces positions critiques. Ce bruit
est suffisamment important pour être considéré comme un problème de cible,
sans prétendre qu'il explique à lui seul toute la baisse en arène.

## 8. Batterie externe D_REAL

Sur les 575 positions physiques uniques :

- G3-A change le top-1 de G2 dans 31,83 % des positions ;
- G3-B le change dans 31,13 %.

Ces taux sont proches des 31,18 % et 28,15 % observés sur le corpus self-play
complet. Le phénomène de déplacement du ranking existe donc hors du corpus
d'entraînement. Toutefois, sans cible stratégique indépendante ni regret
calculé sur ces états, il est impossible de conclure que ces changements sont
moins bons. Le coup humain n'est jamais traité comme vérité optimale.

## 9. Réponses finales

1. **G3 change-t-il de mauvaises actions importantes malgré une meilleure CE
   moyenne ?** Oui : 325 et 290 régressions, dont certaines atteignent 1,319 et
   0,719, avec respectivement 121 et 110 conflits CE améliorée/regret aggravé.

2. **Erreurs nombreuses ou rares et coûteuses ?** Les régressions concernent
   14,5–16,25 % des positions, mais leur coût est très concentré : les 5 % de
   positions les plus coûteuses portent plus de 81 % du regret régressif.

3. **Les cibles MCTS64 sont-elles stables ?** Les Q diagnostiques sans bruit
   sont stables, mais les cibles MCTS64 avec Dirichlet ne présentent que 70,6 %
   d'accord argmax sur les cas critiques.

4. **Le phénomène apparaît-il sur les positions réelles ?** Les changements de
   ranking oui ; leur caractère régressif reste inconclusif.

5. **Modification unique suivante ?** `REGRET_WEIGHTED_POLICY_LOSS`.

## 10. Expérience suivante recommandée

La prochaine expérience devra comparer, à corpus et architecture constants,
la CE actuelle à une loss Policy pondérée par une estimation bornée de
l'importance/regret de l'action. Ce choix vise directement la concentration
mesurée des erreurs coûteuses. Il devra être testé contre une baseline stricte
et ne devra pas convertir Q_search en vérité exacte.

Une pondération par confiance MCTS constitue une alternative crédible au vu de
l'instabilité des cibles, mais elle n'est pas retenue comme expérience
principale afin de ne modifier qu'un mécanisme à la fois.

## Artefacts

- `data/experiments/lot17_policy_regret/report.json`
- `ranking_statistics.json`
- `regret_statistics.json`
- `correlations.json`
- `regression_cases.jsonl`
- `improvement_cases.jsonl`
- `real_positions_diagnostic.json`
- `search_stability.json`
- `sample_manifest.json`
- `regret_search_cache.jsonl`
