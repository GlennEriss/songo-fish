# Lot 7 — Évaluation stratégique contrôlée G0 vs G1

## 1. Question expérimentale

Le premier entraînement du SRN sur le pilote `D_RL` produit-il déjà un effet
stratégique mesurable lorsque le réseau est replacé dans MCTS ?

Le Lot 7 sépare volontairement deux objets :

1. les sorties brutes du réseau (`Policy`, `Value`) sur une batterie fixe ;
2. le comportement du joueur complet (`SRN + MCTS`) dans des parties réelles.

Ce lot ne promeut aucun checkpoint et ne construit aucun classement Elo.

## 2. Modèles comparés

Les trois réseaux partagent exactement la configuration suivante : 8 features
par nœud, 5 features globales, dimension cachée 32, 2 blocs relationnels,
pooling moyen et relations `NEXT/PREV`.

- `G0` : réseau initial reconstruit avec la seed `20260924`, avant toute mise à
  jour d'optimiseur ;
- `G1-best-validation` : checkpoint de l'époque 8 ;
- `G1-last` : checkpoint de l'époque 12.

La reconstruction de G0 a été vérifiée en rejouant l'évaluation de l'époque 0
sur le split de validation du Lot 6. Toutes les métriques ont été reproduites
exactement, avec un écart absolu maximal de `0.0`.

## 3. Batterie indépendante D_LAB

La batterie brute provient uniquement du split `test.jsonl` de D_LAB v001.
Elle n'a servi ni à l'entraînement, ni à la sélection d'un checkpoint.

La sélection est déterministe et indépendante des résultats des modèles : une
position est choisie par minimum SHA-256 seedé dans chaque strate non vide
`phase_proxy × nombre de coups légaux`. La phase est ici un proxy fondé sur
`move_number` : ouverture `<= 15`, milieu `16..60`, tardif `> 60`. Les 19
strates non vides du split test donnent 19 positions. Après tri des strates,
les positions canoniques D_LAB sont alternativement rematérialisées comme P1
et P2 afin d'exercer les deux identités physiques du SRN.

Les labels Minimax de D_LAB ne sont jamais lus par cette sélection et ne sont
jamais fournis au réseau.

## 4. Effet sur les sorties brutes

### 4.1 Policy

| Comparaison | Jensen-Shannon moyenne | JS max. | Argmax différents |
|---|---:|---:|---:|
| G0 / G1-best | 0,000278 | 0,001290 | 12/19 (63,2 %) |
| G0 / G1-last | 0,000249 | 0,000942 | 11/19 (57,9 %) |
| G1-best / G1-last | 0,0000436 | 0,000215 | 8/19 (42,1 %) |

L'entropie moyenne varie très peu : `1,30947` pour G0, `1,30894` pour
G1-best et `1,30894` pour G1-last. Les distributions restent donc très proches
et souvent presque plates. Cependant, de faibles déplacements autour d'actions
quasi ex æquo suffisent à changer fréquemment l'argmax.

### 4.2 Value

| Modèle | Moyenne | Écart-type | Minimum | Maximum |
|---|---:|---:|---:|---:|
| G0 | -0,04195 | 0,01144 | -0,06092 | -0,02651 |
| G1-best | 0,02338 | 0,75339 | -0,98043 | 0,93936 |
| G1-last | 0,05810 | 0,71584 | -0,98216 | 0,93016 |

Le changement absolu moyen de Value vaut `0,71218` entre G0 et G1-best et
`0,66483` entre G0 et G1-last. Le signe change respectivement sur 10/19 et
11/19 positions. L'entraînement a donc modifié la Value beaucoup plus
fortement que la forme globale de la Policy.

Cette observation est descriptive : une Value plus contrastée n'est pas
nécessairement mieux calibrée.

### 4.3 Trois exemples fixes

Les probabilités ci-dessous sont données dans l'ordre des actions locales
`0..6`, avec zéro exact sur les actions illégales.

**Position ouverture, actions légales 0/4/6**

- G0 : `P=[0,32084; 0; 0; 0; 0,33355; 0; 0,34561]`, `V=-0,05587` ;
- G1-best : `P=[0,34506; 0; 0; 0; 0,35677; 0; 0,29817]`, `V=0,32789` ;
- G1-last : `P=[0,34157; 0; 0; 0; 0,35346; 0; 0,30496]`, `V=0,27717`.

**Position ouverture, actions légales 0/1/3/4**

- G0 : argmax 0, `V=-0,03927` ;
- G1-best : argmax 0, `V=-0,39736` ;
- G1-last : argmax 4, `V=-0,03616`.

**Position ouverture, actions légales 1/2/3/4/6**

- G0 : argmax 6, `V=-0,05634` ;
- G1-best : argmax 4, `V=0,47707` ;
- G1-last : argmax 4, `V=0,45461`.

Le fichier `raw_network_outputs.json` conserve cinq exemples complets et les
19 sorties individuelles.

## 5. Protocole exact de l'arène

Huit positions initiales ont été générées avant les matches par des préfixes
légaux de longueurs `0, 1, 2, 3, 5, 8, 13, 21`. Pour chaque préfixe, un RNG
propre dérivé de la seed choisit uniformément parmi les coups légaux du moteur.
Ni G0, ni G1, ni le résultat d'un match n'intervient dans cette génération.
Les préfixes et états obtenus sont persistés dans `openings.json` et sont
rejoués par validation avant chaque partie.

Pour chaque position et chaque confrontation :

- partie 1 : A contrôle P1 et B contrôle P2 ;
- partie 2 : B contrôle P1 et A contrôle P2 ;
- même budget MCTS et même `c_puct=1,5` ;
- bruit de Dirichlet désactivé ;
- température nulle ; action finale = argmax des visites, plus petit indice en
  cas d'égalité finale ;
- la seed ne contrôle que les égalités exactes internes à PUCT ; elle dépend de
  la seed expérimentale, de l'ouverture et du numéro de coup, et est identique
  pour les deux inversions d'une paire ;
- maximum 400 demi-coups et seuil technique de répétition 3 ;
- aucune mise à jour de poids.

Les budgets testés sont 2, 8 et 32 simulations par coup. Cela représente 8
positions × 2 côtés × 3 confrontations × 3 budgets = **144 parties
principales**.

L'intervalle à 95 % est un bootstrap percentile à 10 000 réplications. Son
unité de rééchantillonnage est la position de départ : les deux inversions de
côté d'une même ouverture restent toujours groupées. Avec seulement 8 grappes,
ces intervalles sont volontairement larges et discrets.

## 6. Résultats principaux

Les W/N/D sont toujours donnés du point de vue du premier modèle nommé. Le
score vaut `(victoires + 0,5 × nuls) / parties terminales`.

### 6.1 G0 contre G1-best-validation

| Budget | G0 W/N/D | Score G0 | IC 95 % apparié | G0 en P1 | G0 en P2 |
|---:|---:|---:|---:|---:|---:|
| 2 | 9/1/6 | 0,59375 | [0,4375 ; 0,7500] | 4/0/4 | 5/1/2 |
| 8 | 5/0/11 | 0,31250 | [0,1250 ; 0,4375] | 0/0/8 | 5/0/3 |
| 32 | 6/0/10 | 0,37500 | [0,1875 ; 0,6250] | 2/0/6 | 4/0/4 |

À 8 simulations, l'expérience favorise G1-best sur ces huit ouvertures. Le
budget 2 donne toutefois une direction opposée et l'intervalle du budget 32
recouvre 0,5. L'effet n'est donc pas cohérent sur tous les budgets.

### 6.2 G0 contre G1-last

| Budget | G0 W/N/D | Score G0 | IC 95 % apparié | G0 en P1 | G0 en P2 |
|---:|---:|---:|---:|---:|---:|
| 2 | 5/0/11 | 0,31250 | [0,0625 ; 0,5625] | 3/0/5 | 2/0/6 |
| 8 | 3/0/13 | 0,18750 | [0,0000 ; 0,4375] | 1/0/7 | 2/0/6 |
| 32 | 10/0/6 | 0,62500 | [0,3750 ; 0,8125] | 4/0/4 | 6/0/2 |

G1-last est nettement favorisé au budget 8, mais la direction s'inverse au
budget 32. Cette interaction forte avec le budget interdit une conclusion
générale de supériorité.

### 6.3 G1-best-validation contre G1-last

| Budget | Best W/N/D | Score Best | IC 95 % apparié | Best en P1 | Best en P2 |
|---:|---:|---:|---:|---:|---:|
| 2 | 4/1/11 | 0,28125 | [0,1250 ; 0,4375] | 4/0/4 | 0/1/7 |
| 8 | 8/1/7 | 0,53125 | [0,3125 ; 0,7500] | 2/1/5 | 6/0/2 |
| 32 | 6/0/10 | 0,37500 | [0,1250 ; 0,6250] | 1/0/7 | 5/0/3 |

La meilleure loss de validation ne correspond pas, dans cette petite
expérience, à une supériorité stable en jeu. G1-last gagne clairement au budget
2, mais les deux budgets supérieurs restent indécis.

## 7. Longueur, terminaisons et diversité

| Confrontation | Budget | Moyenne | Médiane | Min. | Max. | Trajectoires distinctes |
|---|---:|---:|---:|---:|---:|---:|
| G0 / Best | 2 | 140,19 | 127,5 | 38 | 325 | 16/16 |
| G0 / Last | 2 | 144,25 | 143,0 | 64 | 265 | 16/16 |
| Best / Last | 2 | 156,81 | 95,5 | 64 | 351 | 16/16 |
| G0 / Best | 8 | 113,19 | 102,0 | 35 | 296 | 16/16 |
| G0 / Last | 8 | 128,12 | 109,0 | 71 | 262 | 16/16 |
| Best / Last | 8 | 81,06 | 71,0 | 25 | 180 | 16/16 |
| G0 / Best | 32 | 107,38 | 72,0 | 36 | 307 | 16/16 |
| G0 / Last | 32 | 86,19 | 82,0 | 14 | 153 | 16/16 |
| Best / Last | 32 | 116,00 | 104,0 | 13 | 355 | 16/16 |

Les 144 parties principales se sont toutes terminées réellement : 0 troncature
par répétition et 0 troncature à 400 demi-coups. Chaque confrontation contient
16 trajectoires distinctes sur 16 ; aucune partie n'est une répétition exacte
d'une autre observation comptée comme indépendante.

## 8. Contrôle RandomLegalAgent

Ce contrôle utilise le budget MCTS 8 et les mêmes huit paires d'ouvertures.

| Confrontation | W/N/D du SRN | Score | IC 95 % apparié |
|---|---:|---:|---:|
| G0 / Random | 7/0/9 | 0,43750 | [0,2500 ; 0,6250] |
| G1-best / Random | 9/1/6 | 0,59375 | [0,4375 ; 0,7500] |

Ces contrôles sont trop petits pour établir même une domination nette de
Random. Ils ne constituent en aucun cas une preuve de force stratégique.

Le benchmark Negamax, optionnel, n'a pas été exécuté : le temps a été consacré
à l'expérience principale et au contrôle Random.

## 9. Coût et intégrité

- parties principales : 144 ;
- parties Random : 32 ;
- total : 176 ;
- simulations MCTS : 226 920 ;
- évaluations réseau : 238 392 ;
- durée murale totale : 95,50 s sur CPU ;
- empreintes des paramètres avant/après : identiques pour les trois réseaux.

Aucun checkpoint n'a été renommé, promu ou modifié.

## 10. Interprétation expérimentale

Le premier apprentissage produit sans ambiguïté un **effet mesurable** : la
Value brute change fortement, l'argmax Policy change souvent malgré de faibles
distances distributionnelles, et les trajectoires de jeu diffèrent.

En revanche, cette expérience ne démontre pas que G1 est globalement meilleur
que G0. La conclusion relève de la catégorie **B** : une différence est
observée, mais son incertitude et surtout son interaction avec le budget MCTS
sont trop grandes pour conclure. Le résultat le plus favorable à G1 apparaît
au budget 8 ; au budget 32, G0 redevient supérieur à G1-last dans cet
échantillon. La sélection `best_validation` n'est pas non plus un substitut
fiable à une évaluation en jeu.

L'anomalie principale à investiguer est le contraste entre :

- une Policy encore presque uniforme et très peu déplacée en distance JS ;
- une Value devenue très dispersée, parfois proche de ±1 ;
- de grands changements de trajectoire MCTS.

Il est plausible que la Value apprise sur le petit pilote D_RL domine déjà la
recherche sans être suffisamment calibrée. Cette proposition est une
hypothèse expérimentale, pas une conclusion.

## 11. Recommandation pour le Lot 8

Le Lot 8 devrait être un lot de **diagnostic Policy/Value et de robustesse de
l'arène**, avant toute deuxième génération automatique :

1. construire des wrappers d'ablation combinant Policy G0 avec Value G1, puis
   Policy G1 avec Value G0 ;
2. comparer ces hybrides aux réseaux complets avec les mêmes ouvertures et
   budgets ;
3. analyser les distributions de Value, leur calibration terminale et leur
   saturation ;
4. augmenter le nombre de grappes d'ouvertures indépendantes pour resserrer les
   intervalles ;
5. répéter prioritairement les budgets 8 et 32 ;
6. ne promouvoir aucun modèle avant que l'effet observé soit cohérent et
   reproductible.

## 12. Artefacts de référence

- `packages/songo_ai/evaluation/srn_arena.py` : arène appariée ;
- `packages/songo_ai/evaluation/srn_benchmark.py` : batterie D_LAB et métriques
  réseau ;
- `apps/trainer/scripts/evaluate_srn_lot7.py` : protocole exécutable ;
- `packages/songo_ai/tests/test_srn_arena.py` : invariants du Lot 7 ;
- `data/experiments/lot7_g0_g1_seed_20260924/openings.json` : départs gelés ;
- `data/experiments/lot7_g0_g1_seed_20260924/raw_network_outputs.json` : sorties
  brutes ;
- `data/experiments/lot7_g0_g1_seed_20260924/arena_games.jsonl` : 176 parties ;
- `data/experiments/lot7_g0_g1_seed_20260924/arena_summaries.json` : agrégats ;
- `data/experiments/lot7_g0_g1_seed_20260924/report.json` : rapport machine.

