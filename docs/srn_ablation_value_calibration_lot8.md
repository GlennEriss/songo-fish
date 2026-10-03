# Lot 8 — Ablation Policy/Value et calibration de Value

## 1. Question et conclusion

Le changement stratégique observé après le premier entraînement provient
principalement de la **Value apprise**. La Policy apprise change souvent le
classement local d'actions presque ex æquo, mais n'apporte aucun avantage
détectable lorsqu'elle est isolée ; avec V1 fixée, elle est même moins
performante que P0 dans les conditions testées.

V1 contient un signal utile, surtout près du terminal, mais elle n'est pas
encore suffisamment fiable pour guider sans précaution les générations
suivantes. Elle est fortement saturée, se dégrade loin du terminal et semble
avoir appris une dépendance excessive à l'identité physique P1/P2.

Ce lot n'a effectué aucun entraînement et n'a modifié aucun checkpoint.

## 2. Évaluateurs testés

- `P0V0` : Policy G0, Value G0 ;
- `P1V1` : Policy G1-best, Value G1-best ;
- `P0V1` : Policy G0, Value G1-best ;
- `P1V0` : Policy G1-best, Value G0 ;
- `P0Vneutral` : Policy G0, Value non terminale égale à zéro ;
- `P1Vneutral` : Policy G1-best, Value non terminale égale à zéro.

Un état utilise un seul forward lorsque Policy et Value viennent du même
réseau. Un hybride P0/V1 ou P1/V0 utilise deux forwards. Les terminaux sont
toujours évalués par le moteur, y compris avec Value neutre.

## 3. Batterie D_LAB élargie

La batterie comprend 350 positions du split test D_LAB v001, sélectionnées par
round-robin déterministe sur des bins structurels : phase proxy, nombre de
coups légaux, graines en jeu et total des magasins. Aucun label teacher et
aucune sortie de modèle n'intervient dans la sélection.

- P1/P2 : 175/175 ;
- ouverture/milieu/fin : 40/195/115 ;
- actions légales 1 à 7 ;
- ply 2 à 211 ;
- graines en jeu 4 à 70 ;
- total des magasins 0 à 66.

## 4. Diagnostic Policy

| Comparaison | JS moyenne | KL gauche→droite | Argmax différents | Variation de P(argmax gauche) |
|---|---:|---:|---:|---:|
| G0 / G1-best | 0,000215 | 0,000875 | 218/350 (62,29 %) | 0,01079 |
| G0 / G1-last | 0,000199 | 0,000807 | 170/350 (48,57 %) | 0,00990 |
| Best / Last | 0,0000563 | 0,000224 | 132/350 (37,71 %) | 0,00435 |

Les entropies moyennes sont presque identiques : `1,32883`, `1,32843` et
`1,32836`. Sur les 325 positions ayant plusieurs coups légaux, la marge
top1-top2 moyenne vaut seulement :

- G0 : `0,00438`, médiane `0,00310` ;
- G1-best : `0,00412`, médiane `0,00201` ;
- G1-last : `0,00713`, médiane `0,00511`.

Le changement fréquent d'argmax est donc compatible avec des distributions
très plates : l'ordre change, mais les probabilités ne se séparent presque pas.

## 5. Distribution et saturation de Value

| Modèle | Moyenne | Médiane | Écart-type | Moyenne |V| | |V|>0,5 | |V|>0,8 | |V|>0,95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | -0,0404 | -0,0397 | 0,0108 | 0,0404 | 0 % | 0 % | 0 % |
| G1-best | -0,0382 | -0,0768 | 0,7316 | 0,6709 | 72,86 % | 46,86 % | 16,57 % |
| G1-last | 0,0362 | -0,0171 | 0,7099 | 0,6430 | 68,29 % | 43,71 % | 11,14 % |

Pour G1-best, 14,57 % des positions sont dans `[-1;-0,95]` et 2 % dans
`[0,95;1]`. G1-last présente 11,14 % de saturation négative et aucune
saturation positive sur cette batterie.

Après seulement 16 parties d'entraînement, cette confiance est élevée. Elle
n'est pas automatiquement fausse, mais nécessite une validation stricte.

## 6. Calibration indépendante

Une nouvelle batterie `D_EVAL_VALUE` a été produite exclusivement pour
l'évaluation : 32 parties G0 + MCTS à 2 simulations, bruit racine activé et
seed indépendante `2026092408`. Elle contient 4 174 positions et 0 troncature.
Elle n'a pas été ajoutée à D_RL.

### 6.1 Calibration globale

| Modèle | MSE | MAE | Sign accuracy |
|---|---:|---:|---:|
| G0 | 1,0061 | 1,0021 | 49,78 % |
| G1-best | 0,9292 | 0,6993 | 67,92 % |
| G1-last | 0,8671 | 0,6907 | 69,19 % |

V1 contient donc un signal réel : elle améliore les trois métriques globales.
G1-last est légèrement meilleur que G1-best sur cette batterie indépendante,
ce qui confirme que la meilleure loss de validation n'est pas nécessairement
le meilleur critère de sélection stratégique.

### 6.2 Calibration par horizon observé

La distance est le nombre de coups restants dans la trajectoire réellement
jouée, pas une distance optimale.

| Horizon | Modèle | MSE | MAE | Sign accuracy |
|---|---|---:|---:|---:|
| 0–5 | G0 | 1,0188 | 1,0087 | 41,25 % |
|  | G1-best | 0,4613 | 0,3554 | 83,75 % |
|  | G1-last | 0,3597 | 0,3075 | 88,13 % |
| 6–15 | G0 | 1,0060 | 1,0023 | 50,00 % |
|  | G1-best | 0,5633 | 0,4217 | 80,63 % |
|  | G1-last | 0,4710 | 0,3874 | 83,44 % |
| 16–30 | G0 | 1,0015 | 0,9999 | 52,92 % |
|  | G1-best | 0,7409 | 0,5627 | 75,42 % |
|  | G1-last | 0,6456 | 0,5329 | 75,83 % |
| 31–60 | G0 | 1,0068 | 1,0025 | 49,83 % |
|  | G1-best | 0,7798 | 0,6121 | 70,31 % |
|  | G1-last | 0,6975 | 0,5814 | 71,92 % |
| >60 | G0 | 1,0059 | 1,0019 | 49,68 % |
|  | G1-best | 1,1049 | 0,8209 | 62,69 % |
|  | G1-last | 1,0640 | 0,8309 | 63,58 % |

V1 est particulièrement informative près du terminal. Au-delà de 60 coups,
son MAE et sa précision de signe restent meilleures que G0, mais sa MSE devient
plus mauvaise : quelques erreurs très confiantes sont fortement pénalisées.

### 6.3 Dépendance P1/P2

Sur D_LAB, pourtant exactement équilibré :

| Modèle | Value moyenne P1 | Value moyenne P2 |
|---|---:|---:|
| G0 | -0,0498 | -0,0311 |
| G1-best | 0,4057 | -0,4821 |
| G1-last | 0,3547 | -0,2822 |

Sur les trajectoires de calibration, les cibles moyennes valent `+0,188` pour
P1 au trait et `-0,181` pour P2, tandis que G1-best prédit en moyenne `+0,562`
et `-0,447`. Le réseau exploite donc un signal réel présent dans le petit
corpus, mais l'amplifie fortement. Une dépendance à l'identité physique plutôt
qu'à la seule structure stratégique est une hypothèse prioritaire à tester.

## 7. Arène d'ablation

Le protocole utilise 32 ouvertures légales déterministes, deux orientations par
ouverture, `c_puct=1,5`, Dirichlet désactivé, température zéro et bootstrap
apparié à 10 000 réplications. Les W/N/D et scores sont ceux du premier agent.

### 7.1 Effet Value avec Policy P0

| Budget | P0V0 vs P0V1 | Score P0V0 | IC 95 % |
|---:|---:|---:|---:|
| 8 | 16/1/47 | 25,78 % | [17,19 ; 34,38] |
| 32 | 21/2/40, 1 troncature | 34,92 % | [24,22 ; 44,53] |

P0V1 est nettement supérieur à P0V0 aux deux budgets. Remplacer seulement V0
par V1 produit donc un effet stratégique positif robuste.

### 7.2 Effet Policy avec Value V0

| Budget | P0V0 vs P1V0 | Score P0V0 | IC 95 % |
|---:|---:|---:|---:|
| 8 | 35/0/29 | 54,69 % | [43,75 ; 65,63] |
| 32 | 36/1/27 | 57,03 % | [45,31 ; 68,75] |

Aucun avantage détectable de P1 sur P0 lorsque V0 est fixée.

### 7.3 Effet Policy avec Value V1

| Budget | P1V1 vs P0V1 | Score P1V1 | IC 95 % |
|---:|---:|---:|---:|
| 8 | 20/2/42 | 32,81 % | [22,66 ; 42,97] |
| 32 | 24/1/39 | 38,28 % | [28,13 ; 48,44] |

Avec V1 fixée, P0V1 bat P1V1 aux deux budgets. La Policy apprise n'explique
donc pas l'amélioration ; dans cette expérience, elle réduit au contraire la
force fournie par V1.

### 7.4 Effet Value avec Policy P1

| Budget | P1V1 vs P1V0 | Score P1V1 | IC 95 % |
|---:|---:|---:|---:|
| 8 | 40/2/22 | 64,06 % | [52,34 ; 75,00] |
| 32 | 37/3/22, 2 troncatures | 62,10 % | [50,00 ; 73,44] |

V1 améliore également l'agent lorsque P1 est fixée. Le résultat à 32 touche la
borne 0,5 mais reste cohérent avec le budget 8.

## 8. Contrôles Value neutre, budget 8

| Confrontation | W/N/D du premier | Score | IC 95 % |
|---|---:|---:|---:|
| P0V0 / P0Vneutral | 30/0/34 | 46,88 % | [39,06 ; 54,69] |
| P1V1 / P1Vneutral | 40/2/21, 1 troncature | 65,08 % | [53,91 ; 77,34] |
| P0Vneutral / P1Vneutral | 35/0/29 | 54,69 % | [43,75 ; 65,63] |

V0 ne se distingue pas de zéro. V1 apporte un gain détectable par rapport à
zéro. Avec Value neutralisée, P0 et P1 ne sont pas distinguables.

## 9. Propagation dans MCTS

Cinq positions à très faible divergence Policy mais argmax brut différent ont
été analysées avec les six évaluateurs.

Exemple 1 : `JS=0,00000375`, argmax brut G0=5 et G1=4. Au budget 32 :

- P0V0 sélectionne 2 ;
- P1V0 sélectionne 2 ;
- P0Vneutral et P1Vneutral sélectionnent 2 ;
- P1V1 sélectionne 5 ;
- P0V1 sélectionne 5.

La source Value, et non la source Policy, détermine ici l'action finale.

Exemple 2 : `JS=0,00000145`, argmax brut G0=1 et G1=2. Au budget 32 :

- les deux variantes V0 sélectionnent 0 ;
- les deux variantes V1 sélectionnent 1 ;
- les deux variantes neutres sélectionnent 1.

Exemple 3 : `JS=0,00000160`. P0V0 et P1V0 sélectionnent 3, P1V1 sélectionne
0 et P0V1 sélectionne 1. Ce cas montre une interaction locale entre priors,
Value très confiante et ordre des visites.

Les priors, visites et Q complets sont conservés dans
`mcts_propagation.json`.

## 10. Coût, terminaisons et intégrité

- parties d'arène : 704 ;
- parties terminales : 700 ;
- répétitions techniques : 1 ;
- max plies : 3 ;
- simulations MCTS : 1 022 912 ;
- états évalués par les hybrides : 1 046 817 ;
- forwards réels des réseaux sources : 1 489 093 ;
- durée totale : 545,19 secondes sur CPU ;
- paramètres avant/après : identiques.

La différence entre états évalués et forwards réels mesure explicitement le
coût des combinaisons Policy/Value provenant de modèles distincts.

## 11. Interprétation scientifique

Les observations convergent :

1. **Value explique l'essentiel du changement stratégique.** V1 bat V0 à
   Policy constante et bat la Value neutre.
2. **Policy n'apporte pas encore de progrès démontrable.** P0 et P1 sont
   indiscernables avec V0 ou Vneutral ; avec V1, P0 est supérieur.
3. **L'interaction n'est pas favorable au modèle complet actuel.** P1 semble
   réduire une partie du bénéfice produit par V1.
4. **V1 apprend un signal réel mais fragile.** Sa calibration est bonne près
   du terminal, puis se dégrade avec l'horizon.
5. **La confiance est excessive au regard du corpus.** Environ 16,6 % des
   positions D_LAB dépassent `|V|=0,95` pour G1-best.
6. **Un biais de perspective physique est plausible.** La séparation P1/P2
   est beaucoup plus forte que les cibles moyennes observées.

La Value ne doit donc pas être rejetée : elle est le premier composant ayant
appris un signal utile. Mais elle n'est pas encore assez calibrée et robuste
pour devenir sans contrôle le guide des prochaines générations.

## 12. Recommandation pour le Lot 9

Le Lot 9 devrait traiter la **robustesse de Value et la faiblesse des cibles
Policy avant G2** :

1. audit de symétrie P1/P2 sur des paires d'états physiquement symétriques ;
2. équilibrage ou augmentation symétrique du futur D_RL ;
3. production d'un D_RL plus grand avec au moins 8 simulations MCTS, car les
   cibles Policy à 2 simulations sont trop pauvres ;
4. comparaison expérimentale de mécanismes de calibration/regularisation de
   Value, sans changer plusieurs variables à la fois ;
5. sélection des checkpoints par calibration indépendante et arène, jamais
   uniquement par loss ;
6. entraînement de G2 seulement après fixation de ce protocole.

## 13. Artefacts

- `packages/songo_ai/evaluation/hybrid_evaluator.py` ;
- `packages/songo_ai/evaluation/value_diagnostics.py` ;
- `packages/songo_ai/evaluation/srn_arena.py` ;
- `packages/songo_ai/evaluation/srn_benchmark.py` ;
- `apps/trainer/scripts/evaluate_srn_lot8.py` ;
- `packages/songo_ai/tests/test_srn_ablation.py` ;
- `data/experiments/lot8_ablation_seed_20260924/report.json` ;
- `data/experiments/lot8_ablation_seed_20260924/expanded_raw_diagnostics.json` ;
- `data/experiments/lot8_ablation_seed_20260924/value_calibration.json` ;
- `data/experiments/lot8_ablation_seed_20260924/mcts_propagation.json` ;
- `data/experiments/lot8_ablation_seed_20260924/arena_games.jsonl`.

