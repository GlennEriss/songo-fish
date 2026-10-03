# Lot 19 — Diverse Position Reanalysis

## 1. Objet et garanties

Le Lot 19 construit un corpus Policy autonome à partir de positions historiques, sans utiliser leurs annotations Teacher. G2-best reste inchangé ; aucun réseau n'est entraîné et aucun label Value n'est inventé.

Le flux appliqué est :

```text
D_TEACHER_UNIQUE − D_RL
  → positions seules (board[16], player_to_move)
  → stratification structurelle
  → G2 + MCTS sans Dirichlet
  → visit_counts + cible Policy
```

Le fichier final ne contient ni `best_action`, ni `action_values`, ni score, marge, profondeur ou PV Teacher, ni `value_target`, `z` ou résultat terminal fictif.

## 2. Sélection du pool diversifié

Le pool principal contient 20 000 positions absentes de tous les shards `D_RL` lisibles. Les positions terminales et celles sans action légale sont rejetées. L'identité reste strictement `(board[16], player_to_move)` : aucune canonicalisation, symétrie ou augmentation miroir n'est appliquée.

La sélection utilise uniquement des variables structurelles : joueur au trait, nombre d'actions légales, graines en jeu, magasins, différence des magasins, cases non vides par territoire, graines par territoire, distribution grossière des graines et provenance technique.

Les bins sont :

- graines en jeu : `≤10`, `11–20`, `21–35`, `36–50`, `>50` ;
- chaque magasin : `≤7`, `8–17`, `18–27`, `>27` ;
- cases non vides par territoire : `≤1`, `2–3`, `4–5`, `>5`.

Une première sélection reflétait trop fortement les anciens JSONL canoniques. Elle a été rejetée avant livraison. La sélection finale impose 10 000 positions P1 et 10 000 positions P2, puis effectue un round-robin reproductible dans les strates avec la seed `20261919`.

### Couverture finale

- joueur : P1 = 10 000, P2 = 10 000 ;
- `legal_count` 1 → 7 : 3 267, 2 669, 4 474, 2 177, 3 830, 1 917, 1 666 ;
- graines en jeu : 2–70, moyenne 30,15 ;
- magasins P1/P2 : 0–35, moyennes 19,95 / 19,90 ;
- différence des magasins : −35 → +35 ;
- cases non vides : toute la plage 0–7 sur chaque territoire.

Le corpus matriciel fournit 13 052 positions. Cette contribution de 65,26 % est explicitement documentée : elle vient du fait que les anciens JSONL ont perdu l'identité physique et représentent implicitement le joueur courant en P1, alors que le matriciel est la principale source de positions P2 récupérables. Il s'agit en outre du grand corpus indépendant identifié au Lot 18, et non d'une release emboîtée du 400k.

Par rapport aux corpus G1→G2 et G2→G3, l'intersection exacte est nulle par construction. Le pool couvre davantage les faibles nombres de coups légaux et des états plus avancés : environ 30,15 graines en jeu contre 33,89–34,21 dans `D_RL`, tout en conservant toutes les plages structurelles.

## 3. Calibration MCTS

Un sous-ensemble de 2 000 positions est stratifié selon joueur, branchement, graines, magasins, entropie G2 et marge G2.

Configuration commune :

```text
checkpoint       G2-best
c_puct           1.5
Dirichlet        OFF
target temp.     1.0
budgets          64 et 128
```

### Stabilité inter-seeds

Trois seeds ont été exécutées sur les 2 000 positions au budget 64. Les comptes de visites sont identiques dans 100 % des cas, l'accord argmax est de 100 % et la JS maximale vaut 0. Le MCTS existant est donc empiriquement déterministe sans bruit racine dans cette configuration ; la seed n'intervient que pour d'éventuelles égalités exactes, absentes sur cet échantillon.

### MCTS64 contre MCTS128

| Mesure | Résultat |
|---|---:|
| JS moyenne | 0,004982 |
| JS P95 | 0,020177 |
| Accord argmax | 78,40 % |
| Recouvrement top-2 | 85,03 % |
| Variation moyenne de probabilité top-1 | 0,04674 |

La faible JS moyenne masque 21,6 % de changements d'argmax. L'accord diminue avec le branchement et l'entropie : le budget 64 n'est donc pas considéré suffisamment stable pour construire la nouvelle référence.

### Contrôle ciblé MCTS256

Les 200 plus fortes instabilités 64/128 ont été réanalysées à 256 simulations :

| Comparaison | JS moyenne | Accord argmax | Top-2 |
|---|---:|---:|---:|
| 64 vs 256 | 0,03694 | 18,0 % | voir artefact |
| 128 vs 256 | 0,01264 | 68,0 % | voir artefact |

MCTS128 réduit la JS de 65,8 % et augmente l'accord argmax de 50 points sur ces cas ciblés. Il est donc matériellement meilleur, sans justifier un MCTS256 massif.

## 4. Corpus autonome produit

Le corpus final contient 20 000 `ReanalysisPolicyExample` :

```text
state
legal_mask
visit_counts
policy_target
search_metadata
source_position_metadata
```

`search_metadata` conserve G2, le budget 128, `c_puct`, l'absence de Dirichlet, la température, la root Value strictement diagnostique, ainsi que les mesures brutes de confiance : marge des visites, entropie et support. Ces mesures ne sont pas encore converties en poids d'apprentissage.

Le fichier final est [`lot19_diverse_20k_g2_mcts.jsonl`](../data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl), avec SHA256 :

```text
ddc42088c38264fc05dfe6b1faee62a4b7b1fa2613219d71fc6414351928d0cb
```

## 5. Qualité des cibles

| Mesure | Lot 19 / MCTS128 |
|---|---:|
| One-hot rate | 16,34 % |
| Support moyen | 3,65 |
| Support médian | 3 |
| Entropie moyenne | 1,033 |
| Accord argmax `P_G2` / réanalyse | 52,04 % |
| Recouvrement top-2 | 71,68 % |
| JS moyenne `P_G2` / réanalyse | 0,02294 |
| Gain moyen sur l'action préférée MCTS | +0,1112 |

La cible contient donc un signal Policy non trivial : MCTS change l'action dominante dans près de 48 % des positions et augmente sensiblement sa probabilité, sans s'effondrer en labels one-hot.

Comparaison historique :

| Corpus | Simulations | One-hot | Support moyen | Entropie moyenne |
|---|---:|---:|---:|---:|
| Lot 5 | 2 | 55,50 % | 1,45 | 0,308 |
| Lot 10 G0 | 8 | 3,51 % | 4,87 | 1,442 |
| Lot 10 G1 | 8 | 23,64 % | 3,24 | 0,871 |
| Lot 14 | 64 | 5,09 % | 4,41 | 1,274 |
| Lot 19 | 128 | 16,34 % | 3,65 | 1,033 |

La différence n'est pas interprétée comme une supériorité automatique : les positions et la politique génératrice diffèrent. Elle montre que les cibles Lot 19 sont informatives sans être uniformes ni presque toutes one-hot.

## 6. Cas critiques du Lot 17

Les 50 principales régressions et 50 principales améliorations ont été réanalysées avec G2/MCTS128 sans Dirichlet. Par rapport à l'ancienne cible MCTS64 bruitée :

- accord argmax : 80 % ;
- JS moyenne : 0,01931 ;
- JS P95 : 0,05632 ;
- les nouvelles cibles sans Dirichlet ont la propriété `SAME_RESULT` démontrée sur la calibration.

La suppression du bruit rend la cible reproductible, mais le changement d'argmax sur 20 % des cas critiques confirme qu'une partie du signal historique dépendait bien de la configuration de recherche.

## 7. Coût

Sur le 20K final, le temps de recherche cumulé mesuré est d'environ 900,18 s, soit 0,0450 s/position, 2 844 simulations/s et 2 554 évaluations réseau/s.

| Taille | Estimation MCTS128 |
|---:|---:|
| 20K | 0,25 h |
| 50K | 0,63 h |
| 100K | 1,25 h |
| 250K | 3,13 h |
| 941 599 | 11,77 h |

Ces chiffres sont des extrapolations du temps interne de recherche sur cette machine ; ils n'incluent pas nécessairement tout le chargement, la sélection et la sérialisation.

## 8. Gate et verdicts

Le gate vérifie : intégrité et légalité, provenance explicite, exclusion exacte de `D_RL`, diversité structurelle, stabilité inter-seeds et inter-budgets, one-hot rate, support, entropie et signal par rapport à `P_G2`.

```text
DIVERSE_POOL_VALID = YES
MCTS64_STABLE_ENOUGH = NO
MCTS128_MATERIALLY_BETTER = YES
REANALYSIS_BUDGET = 128
REANALYSIS_TARGETS_STABLE = YES
D_REANALYSIS_20K_READY = YES
RECOMMENDED_NEXT_CORPUS_SIZE = 20K
NEXT_TRAINING_EXPERIMENT = DUAL_SOURCE_POLICY:
  REGRET_WEIGHTED_D_RL + CONFIDENCE_WEIGHTED_D_REANALYSIS_20K
```

La taille recommandée reste 20K pour le prochain apprentissage : elle est suffisante pour tester causalement la correction Policy avant de payer le coût et le risque de confusion d'un corpus plus grand.

## 9. Prochain protocole d'apprentissage

Le prochain candidat doit conserver deux flux explicitement séparés :

1. `D_RL`, qui fournit Policy et la vraie Value terminale ; sa composante Policy peut tester la pondération par importance/regret issue du Lot 17 ;
2. `D_REANALYSIS_20K`, qui fournit uniquement Policy et une confiance de recherche autonome ; aucune loss Value ne doit être calculée sur ces exemples.

Une régularisation vers la Policy parente G2 et une ablation sans pondération devront permettre d'attribuer le gain. La formule exacte des poids appartient au prochain lot et n'est pas figée ici.

## 10. Livrables reproductibles

- script : [`run_srn_lot19.py`](../apps/trainer/scripts/run_srn_lot19.py) ;
- rapport consolidé : [`report.json`](../data/experiments/lot19_diverse_reanalysis/report.json) ;
- manifest : [`manifest.json`](../data/experiments/lot19_diverse_reanalysis/manifest.json) ;
- artefacts de calibration et qualité : [`lot19_diverse_reanalysis`](../data/experiments/lot19_diverse_reanalysis) ;
- corpus Policy-only : [`lot19_diverse_20k_g2_mcts.jsonl`](../data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl).
