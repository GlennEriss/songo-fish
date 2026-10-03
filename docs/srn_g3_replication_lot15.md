# Lot 15 — Réplication contrôlée G2→G3 et signal MCTS

## Verdict exécutif

- **OPTIMIZATION_VARIANCE_EXPLAINS_G3A = INCONCLUSIVE**
- **G3B_PROGRESS_OVER_G2 = NO**
- **G3B_GENERATOR_CANDIDATE = NO**
- **NEXT_ACTION =** ne promouvoir ni G3-A ni G3-B, arrêter les réplications de
  seed et concevoir un diagnostic ciblé du mécanisme d'amélioration avant toute
  nouvelle génération.

G3-B est approximativement au niveau de G2 aux deux budgets testés. La nouvelle
seed évite la régression statistiquement établie de G3-A à MCTS64, mais elle ne
démontre aucune progression. Le résultat correspond donc au scénario 2 : signal
G2→G3 faible ou plateau, et non preuve que G3-A était seulement une mauvaise
seed.

## A — Intégrité de la réplication

### Corpus et parent

| Élément | Valeur vérifiée |
|---|---|
| Corpus | `data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl` |
| SHA-256 corpus avant/après | `2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57` |
| Corpus inchangé | oui |
| Checkpoint parent | G2-best-validation, époque 4 du Lot 12 |
| SHA-256 G2 | `eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753` |
| Seed G3-A | 20261402 |
| Seed G3-B | 20261515 |

Le split est strictement identique à celui de G3-A : 400 parties et 35 111
positions en entraînement, 100 parties et 7 765 positions en validation. Les
listes de `game_id` sont égales des deux côtés et leur intersection est vide.

Avant tout pas d'optimisation, les sorties Policy et Value du clone G3-B sont
exactement égales à celles du checkpoint G2 sur la batterie fixe. Les métriques
d'époque 0 de G3-A et G3-B sont également identiques :

| Validation époque 0 | G3-A | G3-B |
|---|---:|---:|
| Loss totale | 2,117495 | 2,117495 |
| Policy CE | 1,393383 | 1,393383 |
| Policy KL | 0,109459 | 0,109459 |
| Policy top-1 | 32,968 % | 32,968 % |
| Value MSE | 0,724111 | 0,724111 |
| Value MAE | 0,709744 | 0,709744 |
| Value sign accuracy | 56,693 % | 56,693 % |

Aucun label Minimax/teacher n'est présent. Les datasets 10k et 100k n'ont pas
été utilisés. Aucun self-play n'a été régénéré.

## B — Entraînement G3-B

Les paramètres du Lot 14 ont été repris sans modification, à l'exception de la
seed : 30 époques maximum, batch 256, Adam avec taux `0.003`, weight decay
`0.0001`, gradient clipping `1.0`, coefficients Policy/Value égaux à 1 et
patience 5.

G3-B atteint sa meilleure validation à l'époque 2 et s'arrête à l'époque 7.
La courbe complète est conservée dans `training/metrics.csv` et
`training/metrics.json`.

| Meilleure validation | G3-A (ép. 4) | G3-B (ép. 2) |
|---|---:|---:|
| Loss totale | 2,112058 | 2,114022 |
| Policy CE | 1,390771 | 1,391406 |
| Policy KL | 0,106847 | 0,107481 |
| Policy top-1 | 34,900 % | 34,630 % |
| Value MSE | 0,721287 | 0,722616 |
| Value MAE | 0,723304 | 0,722689 |
| Value sign accuracy | 54,177 % | 55,355 % |

Les deux optimisations aboutissent à des métriques proches. G3-A est légèrement
meilleur sur les pertes Policy et Value, tandis que G3-B a une meilleure
précision de signe Value. Ces écarts ne suffisent pas à prédire la force en jeu.

Checkpoints G3-B :

- best-validation : SHA-256
  `a38dd48d0d80cac2c3202c14e2442a6b2c588e39075f1fd2074c8e48f0116d4b` ;
- last : SHA-256
  `28809883c1920a61e61ed0bbf036554e28386a81228a69454ea78b6a227e3156`.

## C — Signal Policy G2 → MCTS64

Le diagnostic passif porte sur l'intégralité des 42 876 positions. Il compare
la Policy brute du parent G2 à la cible MCTS64 déjà stockée, sans modifier le
corpus ni l'entraînement.

| Mesure | Moyenne | Médiane | Q05 | Q25 | Q75 | Q95 |
|---|---:|---:|---:|---:|---:|---:|
| JS | 0,02649 | 0,01494 | 0,00000 | 0,00717 | 0,03173 | 0,09468 |
| KL(π MCTS ‖ P G2) | 0,10942 | 0,06078 | 0,00000 | 0,02879 | 0,12882 | 0,39716 |
| Accord argmax | 32,39 % | — | — | — | — | — |
| Action MCTS dans le top-2 G2 | 57,14 % | — | — | — | — | — |
| Recouvrement top-2 | 56,28 % | 50,00 % | 0 % | 50 % | 100 % | 100 % |
| Entropie Policy G2 | 1,38450 | 1,60147 | 0,00000 | 1,09846 | 1,78969 | 1,94458 |
| Entropie MCTS64 | 1,27381 | 1,35684 | 0,00000 | 1,04516 | 1,60110 | 1,87325 |
| Δ proba sur l'action préférée MCTS | +0,15111 | +0,12684 | 0,00000 | +0,07846 | +0,19247 | +0,38476 |

MCTS64 transforme donc réellement les préférences : son argmax ne coïncide
avec celui de G2 que dans 32,39 % des positions et il ajoute en moyenne 15,11
points de probabilité à son action préférée. Toutefois, la divergence JS reste
faible en moyenne et décroît nettement avec le ply. Le signal est une
réorganisation mesurée d'une distribution déjà structurée, pas une cible
radicalement différente sur toutes les positions.

### Sous-groupes

| Segment | N | JS moy. | Accord argmax | H(G2) | H(MCTS) | Δ action MCTS |
|---|---:|---:|---:|---:|---:|---:|
| Ply 0–30 | 15 362 | 0,0339 | 22,56 % | 1,6996 | 1,5557 | +0,1691 |
| Ply 31–90 | 17 214 | 0,0261 | 31,44 % | 1,3668 | 1,2592 | +0,1525 |
| Ply 91+ | 10 300 | 0,0161 | 48,65 % | 0,9441 | 0,8779 | +0,1221 |
| Joueur 1 | 21 567 | 0,0271 | 33,34 % | 1,3877 | 1,2739 | +0,1533 |
| Joueur 2 | 21 309 | 0,0259 | 31,43 % | 1,3813 | 1,2737 | +0,1489 |

La divergence augmente avec le nombre de coups légaux : JS vaut 0 quand une
seule action est disponible, puis 0,0120 à deux actions et 0,0366 à sept. Il
n'existe pas d'asymétrie importante entre joueurs. Le gain informationnel de
MCTS est surtout visible tôt dans la partie et lorsque le choix est large.

## D — Arène G3-B contre G2

Protocole : 128 ouvertures appariées avec inversion P1/P2, seed indépendante
20261516, `c_puct=1.5`, Dirichlet désactivé, température nulle, choix par argmax
des visites et 20 000 réplications bootstrap par paire d'ouverture.

| Budget | W/D/L G3-B | Terminaux | Score | IC 95 % apparié | Troncatures |
|---|---:|---:|---:|---:|---:|
| MCTS32 | 114/13/124 | 251 | 48,01 % | [42,97 %, 53,71 %] | 5 répétitions |
| MCTS64 | 115/6/132 | 253 | 46,64 % | [41,60 %, 52,93 %] | 3 répétitions |

| Budget/côté de G3-B | W/D/L | Troncatures |
|---|---:|---:|
| MCTS32, P1 | 61/8/55 | 4 |
| MCTS32, P2 | 53/5/69 | 1 |
| MCTS64, P1 | 53/4/69 | 2 |
| MCTS64, P2 | 62/2/63 | 1 |

À MCTS32, les parties durent en moyenne 95,56 plies (médiane 83, plage
2–330). À MCTS64, elles durent 96,36 plies en moyenne (médiane 89, plage
2–326). Les deux intervalles incluent 50 % : ni supériorité ni infériorité de
G3-B ne sont établies dans cette expérience.

## E — Comparaison G2, G3-A et G3-B

| Indicateur face à G2 | G3-A | G3-B |
|---|---:|---:|
| MCTS32 score | 47,62 % | 48,01 % |
| MCTS32 IC 95 % | [41,80 %, 53,32 %] | [42,97 %, 53,71 %] |
| MCTS64 score | 43,55 % | 46,64 % |
| MCTS64 IC 95 % | [38,28 %, 48,83 %] | [41,60 %, 52,93 %] |
| Best epoch | 4 | 2 |
| Validation Policy KL | 0,10685 | 0,10748 |
| Validation Policy top-1 | 34,90 % | 34,63 % |
| Validation Value MSE | 0,72129 | 0,72262 |
| Validation Value sign | 54,18 % | 55,35 % |

G2 demeure la référence à 50 % et le dernier générateur autorisé. G3-A était
significativement inférieur à MCTS64 ; G3-B remonte vers le niveau de G2 mais
ne le dépasse pas. La seed influe donc sur le résultat d'arène, sans expliquer
à elle seule l'absence de progrès générationnel.

Le benchmark Minimax n'a pas été relancé, conformément au protocole. Les
références historiques restent G2 à 1,56 % et G3-A à 1,95 % contre
`MINIMAX_BIDOUA_REFERENCE_V1`.

## F — Verdict et suite

**OPTIMIZATION_VARIANCE_EXPLAINS_G3A = INCONCLUSIVE.** La réplication ne
reproduit pas la régression statistiquement nette de G3-A, mais ne produit pas
non plus un candidat supérieur à G2. Une seule seconde seed ne permet pas
d'attribuer principalement l'échec de G3-A à la variance.

**G3B_PROGRESS_OVER_G2 = NO.** Les scores MCTS32 et MCTS64 sont sous 50 % et
leurs intervalles de confiance incluent 50 %.

**G3B_GENERATOR_CANDIDATE = NO.** G2 reste le générateur autorisé. Aucun G4,
G3-C ou benchmark Minimax supplémentaire ne doit être lancé sur cette base.

**NEXT_ACTION =** traiter G2→G3 comme un signal faible/plateau et réaliser un
diagnostic unique du mécanisme avant toute nouvelle génération. Le diagnostic
doit partir du constat mesuré ici : MCTS change souvent l'action préférée, mais
la divergence moyenne reste modérée, surtout en fin de partie. Il devra tester
si l'amélioration produite par MCTS64 est suffisamment cohérente et
apprenable — notamment selon le ply et le nombre d'actions légales — et examiner
séparément la qualité/calibration de la cible Value. Il ne faut ni chercher une
seed chanceuse ni modifier simultanément plusieurs mécanismes.

## Artefacts

- `data/experiments/lot15_g3b_seed_20261515/report.json`
- `data/experiments/lot15_g3b_seed_20261515/policy_mcts_signal.json`
- `data/experiments/lot15_g3b_seed_20261515/arena_openings.json`
- `data/experiments/lot15_g3b_seed_20261515/sanity_fixed_positions.json`
- `data/experiments/lot15_g3b_seed_20261515/training/metrics.json`
- `data/experiments/lot15_g3b_seed_20261515/training/metrics.csv`
- `data/experiments/lot15_g3b_seed_20261515/training/best_validation_checkpoint.pt`
- `data/experiments/lot15_g3b_seed_20261515/training/last_checkpoint.pt`
