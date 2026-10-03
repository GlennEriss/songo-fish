# Lot 10 — Régénération contrôlée de D_RL à 8 simulations

## 1. Décision

Le gate de qualité n'est satisfait que par le corpus produit avec G0. Le
corpus produit avec G1-best améliore sa richesse marginale, mais ne rend pas
les cibles des états répétés plus cohérentes. Conformément au protocole, le
pipeline s'est arrêté avant tout entraînement G2.

La réponse à la question du lot est donc nuancée : **huit simulations
produisent un signal nettement meilleur avec G0, mais ne suffisent pas à
garantir un corpus stable avec G1-best**. Le passage aux générations
successives n'est pas encore justifié.

Aucun checkpoint G2 et aucune arène post-entraînement n'ont été produits.

## 2. Contrat de génération

Les deux expériences utilisent exactement les mêmes paramètres :

- 200 parties ;
- 8 simulations MCTS par position ;
- `c_puct=1,5` ;
- Dirichlet activé, `alpha=0,3`, `epsilon=0,25` ;
- température de cible `1,0` ;
- température d'action `1,0`, puis `0` à partir du ply 30 ;
- maximum 400 plies ;
- seuil technique de répétition 3 ;
- états physiques bruts, sans canonicalisation ;
- aucune augmentation miroir ;
- aucune annotation Minimax.

La seed d'une recherche est dérivée par :

```text
SHA256(base_seed : game_index : ply : purpose)
```

Elle ne dépend pas du modèle. G0 et G1-best reçoivent donc des seeds appariées.
Les identifiants de checkpoint restent distincts et les deux shards ne sont
jamais fusionnés.

## 3. Génération

| Mesure | D_RL_G0_8 | D_RL_G1_8 |
|---|---:|---:|
| Parties | 200 | 200 |
| Positions | 18 902 | 16 845 |
| Parties terminales | 200 | 200 |
| Troncatures répétition | 0 | 0 |
| Troncatures max-plies | 0 | 0 |
| Simulations MCTS | 151 216 | 134 760 |
| Durée génération | 67,07 s | 59,30 s |
| Longueur moyenne | 94,51 | 84,23 |

Toutes les sommes de `visit_counts` valent exactement 8. Les deux round-trips
JSONL sont exacts. Aucun exemple n'a été dédupliqué et tous les exemples ont
une vraie cible terminale.

## 4. Résultats P1/P2

### 4.1 Corpus G0

| Résultat | Valeur |
|---|---:|
| Victoires P1 | 107 |
| Victoires P2 | 91 |
| Nuls réels | 2 |
| Positions P1/P2 | 9 503 / 9 399 |
| z moyen P1/P2 | +0,0731 / -0,0653 |
| Longueur si P1 gagne | 92,93 |
| Longueur si P2 gagne | 94,89 |

Ce corpus est beaucoup mieux équilibré que le pilote du Lot 5.

### 4.2 Corpus G1-best

| Résultat | Valeur |
|---|---:|
| Victoires P1 | 56 |
| Victoires P2 | 135 |
| Nuls réels | 9 |
| Positions P1/P2 | 8 459 / 8 386 |
| z moyen P1/P2 | -0,3315 / +0,3373 |
| Longueur si P1 gagne | 89,89 |
| Longueur si P2 gagne | 79,01 |

Le corpus G1-best présente un biais physique important, de signe opposé au
Lot 5. Équilibrer le nombre brut de positions ne suffit donc toujours pas.

## 5. Qualité informationnelle Policy

| Corpus | One-hot | Support moyen | Entropie | Marge top1-top2 |
|---|---:|---:|---:|---:|
| Lot 5, 2 sims | 55,50 % | 1,445 | 0,308 | 0,555 |
| G0, 8 sims | **3,51 %** | **4,874** | **1,442** | 0,141 |
| G1-best, 8 sims | 23,64 % | 3,241 | 0,871 | 0,443 |

Les deux nouveaux corpus satisfont les trois critères de richesse marginale :

- taux one-hot au moins divisé par deux ;
- gain de support supérieur à une action ;
- entropie supérieure à 150 % du Lot 5.

L'amélioration est cependant beaucoup plus forte avec G0. La Value très
contrastée de G1-best concentre encore fortement les visites malgré le budget
supérieur.

## 6. États répétés et position initiale

| Mesure | Lot 5 | G0/8 | G1-best/8 |
|---|---:|---:|---:|
| États répétés | 8 | 104 | 106 |
| JS moyenne répétitions | 0,5520 | **0,0409** | 0,4754 |
| Accord argmax | 14,22 % | 15,83 % | 15,40 % |
| Variance composantes | 0,0671 | **0,00269** | 0,02849 |

Pour la position initiale :

| Mesure | Lot 5 | G0/8 | G1-best/8 |
|---|---:|---:|---:|
| Occurrences | 20 | 200 | 200 |
| Cibles distinctes | 16 | 37 | 84 |
| JS moyenne | 0,5423 | **0,0387** | **0,5449** |
| Accord argmax | 15,79 % | 15,78 % | 15,49 % |
| Variance composantes | 0,07054 | **0,00289** | **0,09026** |

L'accord d'argmax reste faible parce que plusieurs actions sont proches, mais
les distributions G0 sont numériquement beaucoup plus cohérentes. Pour
G1-best, la JS initiale est légèrement pire que celle du Lot 5 et la variance
augmente. L'exigence « même état, cible nettement plus cohérente » n'est donc
pas satisfaite.

## 7. Contrôle 8 contre 32 simulations

Chaque modèle est testé sur 24 états fixes, quatre seeds par état, avec le même
bruit racine apparié entre les budgets.

| Générateur | JS(π8,π32) | Accord argmax | Support 8/32 | Entropie 8/32 |
|---|---:|---:|---:|---:|
| G0 | **0,00311** | 83,33 % | 4,33 / 4,33 | 1,322 / 1,333 |
| G1-best | 0,06854 | 73,96 % | 3,80 / 4,55 | 1,091 / 1,324 |

Huit simulations constituent une approximation très proche de 32 pour G0 sur
cette batterie. La différence reste substantielle pour G1-best : le budget 32
visite davantage d'actions et produit une cible plus entropique.

## 8. Asymétrie historique conservée

| Mesure | G0/8 | G1-best/8 |
|---|---:|---:|
| États avec une action potentiellement non équivariante | 493 | 308 |
| Actions légales testées | 93 120 | 78 072 |
| Actions potentielles non équivariantes | 506 | 328 |
| Actions non équivariantes réellement jouées | 87 | 76 |
| Parties avec au moins une action réellement concernée | 70 | 67 |

Ces exemples ont été conservés. Ils appartiennent à l'environnement réellement
exécuté et n'ont été ni supprimés, ni transformés.

## 9. Gate de qualité

Le gate relatif au Lot 5 exige :

1. one-hot divisé au moins par deux ;
2. support moyen augmenté d'au moins une action ;
3. entropie augmentée d'au moins 50 % ;
4. JS de la position initiale divisée au moins par deux ;
5. sérialisation valide ;
6. présence de résultats terminaux exploitables.

Résultat :

| Critère | G0/8 | G1-best/8 |
|---|---:|---:|
| One-hot | succès | succès |
| Support | succès | succès |
| Entropie | succès | succès |
| Stabilité initiale | succès | **échec** |
| Sérialisation | succès | succès |
| Terminaux | succès | succès |
| Gate global | **SUCCÈS** | **ÉCHEC** |

Le gate G1-best n'échoue pas à cause d'un seuil marginal : sa JS initiale ne
s'améliore pas du tout (`0,5449` contre `0,5423`). L'arrêt est donc robuste à
une formulation moins stricte exigeant seulement une amélioration.

## 10. Entraînement et early stopping

L'early stopping configurable a été implémenté et testé avec conservation de
`best_validation_checkpoint` et `last_checkpoint`. L'initialisation depuis un
checkpoint utilise uniquement ses poids et crée un optimiseur neuf, en
enregistrant chemin, SHA-256 et époque source.

Cependant, **aucun entraînement du Lot 10 n'a été lancé**, car le gate commun
aux deux expériences a échoué. Il n'existe donc ni G2-from-G0-data ni
G2-from-G1-data.

## 11. Arène

L'arène était conditionnée à la création valide des deux G2. Elle n'a pas été
exécutée. Aucun résultat stratégique n'est inventé et aucun checkpoint n'est
qualifié de champion.

## 12. Anomalies et interprétation

Les deux shards sont techniquement sains. L'anomalie est scientifique : le
modèle générateur influence fortement la qualité des labels.

G1-best possède une Value saturée et physiquement biaisée, déjà identifiée aux
Lots 8 et 9A. Avec Dirichlet, cette Value peut modifier fortement les chemins
explorés après les premières visites. Augmenter le budget de 2 à 8 enrichit le
support, mais ne stabilise pas les distributions répétées. Le contrôle 32
montre qu'un budget supérieur continue à modifier sensiblement les cibles G1.

## 13. Recommandation

Le prochain lot devrait être un contrôle ciblé, sans G2 :

1. générer un pilote G1-best plus petit à 32 simulations ;
2. comparer G1-best complet à `P1/Vneutral` et éventuellement à une Value
   calibrée, afin d'isoler la source de l'instabilité ;
3. conserver G0/8 comme corpus de référence valide, sans l'utiliser seul pour
   contourner l'expérience comparative prévue ;
4. décider ensuite si le corpus G1 doit utiliser 32 simulations ou une Value
   mieux calibrée ;
5. ne commencer les deux entraînements G2 qu'après réussite du même gate.

## 14. Artefacts

- `data/d_rl/lot10_g0_8_seed_20260924.jsonl` ;
- `data/d_rl/lot10_g1_8_seed_20260924.jsonl` ;
- `data/experiments/lot10_drl8_seed_20260924/quality_report.json` ;
- `data/experiments/lot10_drl8_seed_20260924/report.json` ;
- `packages/songo_ai/evaluation/corpus_quality.py` ;
- `packages/songo_ai/model/srn_training.py` ;
- `apps/trainer/scripts/run_srn_lot10.py` ;
- `packages/songo_ai/tests/test_lot10_corpus_quality.py`.
