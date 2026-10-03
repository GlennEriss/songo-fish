# Lot 16B — Audit du patrimoine de données réelles Songo

## Verdict exécutif

Le dépôt accessible ne contient pas 25 datasets de matchs réels. Il contient
**un seul dataset logique** de vrais coups, `match_moves_v1`, dupliqué sous deux
formats strictement équivalents (`JSON` et `JSONL`). Les éventuels exports
bruts ayant servi à sa construction ne sont pas présents dans le workspace.

Les répertoires 10k, 100k, 110k, 290k et 400k ne sont pas des matchs réels :
leurs manifests et schémas montrent qu'il s'agit de positions générées puis
annotées par teacher. Ils sont donc exclus de `D_REAL`, tout comme `D_RL`, les
arènes, checkpoints, caches et rapports expérimentaux.

## 1. Inventaire disponible

| Fichier | Format | Taille | Enregistrements | États uniques | Matchs | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `real_matches/match_moves_v1.json` | JSON | 1 085 346 o | 828 | 575 | 683 | `0b150a1a…af132f` |
| `real_matches/match_moves_v1.jsonl` | JSONL | 586 283 o | 828 | 575 | 683 | `17d9ad19…5aed18` |

Les deux fichiers partagent 828/828 enregistrements et 575/575 identités
physiques : ce sont deux sérialisations du même contenu, pas deux datasets.
La somme naïve de 1 656 lignes doit donc être ramenée à 828 enregistrements
logiques.

## 2. Schéma réel

Chaque enregistrement contient :

- `board_before[16]`, `player_position` ;
- `action_local`, `case_id`, `legal_mask`, `legal_case_ids` ;
- `match_id`, `ply`, timestamp ;
- `board_after`, capture, fin éventuelle et `winner_after` ;
- méthode et certitude de reconstruction.

Il ne contient pas un résultat terminal fiable pour chaque position, ni score
final systématique. `winner_after` est renseigné lorsque le coup stocké termine
la partie ; 386 enregistrements non terminaux ont logiquement une valeur nulle.
Le corpus ne suffit donc pas directement à superviser Value sur toutes ses
positions.

## 3. Intégrité et doublons internes

Les 828 enregistrements passent les contrôles suivants : plateau de dimension
16, compteurs entiers non négatifs, somme égale à 70, joueur valide, masque de
légalité cohérent avec le moteur et action jouée légale. Aucune donnée n'a été
corrigée.

- positions physiques uniques : **575** ;
- répétitions de position : **253**, soit **30,56 %** ;
- couples position/action uniques : **608** ;
- surplus de couples dû à des actions différentes sur un même état : **33** ;
- matchs uniques : **683** ;
- 615 matchs ont un seul coup stocké, 68 en ont 2 à 6 ;
- parmi 145 paires successives disponibles, 135 ont une continuité exacte
  `board_after → board_before`.

La reconstruction de trajectoires complètes n'est donc pas garantie : le
corpus est majoritairement un ensemble de coups isolés.

## 4. Union réelle et distributions

`D_REAL_UNION` désigne ici les 575 états physiques uniques, sans créer de
nouveau fichier d'entraînement et sans canonicaliser les joueurs.

| Distribution | Résultat |
|---|---|
| Joueur au trait | P1 496 (86,26 %), P2 79 (13,74 %) |
| Ply 0–30 | 82 (14,26 %) |
| Ply 31–90 | 373 (64,87 %) |
| Ply 91+ | 120 (20,87 %) |
| Actions les plus fréquentes | action 6 : 38,26 % ; action 5 : 15,30 % |
| Branching 1/2/3/4/5/6/7 | 56 / 47 / 73 / 125 / 130 / 116 / 28 |

Les résultats terminaux observés sur les 442 coups terminaux sont fortement
déséquilibrés : 410 victoires P1, 30 victoires P2 et 2 nulles. Cette
distribution, combinée au déséquilibre du joueur au trait, interdit de traiter
le corpus comme un échantillon représentatif sans étude complémentaire.

## 5. Chevauchement avec le self-play

L'identité comparée est strictement physique : `board[16] + player_to_move`.

| Corpus self-play | Positions | Uniques | Communes avec D_REAL | D_REAL absent | Couverture D_REAL |
|---|---:|---:|---:|---:|---:|
| G1→G2 | 42 847 | 41 046 | 52 | 523 | 9,04 % |
| G2→G3 | 42 876 | 41 073 | 52 | 523 | 9,04 % |

Les états communs ne représentent que 0,127 % des uniques G1→G2 et 0,127 %
des uniques G2→G3. Le corpus réel apporte donc **523 états exacts absents** de
chacun des corpus self-play actuels. Sa distribution diffère aussi nettement :
64,87 % de milieu de partie contre 40,15 % dans G2→G3, et seulement 4,87 % de
branching 7 contre 10,77 %.

Cette diversité est réelle mais ne prouve pas une qualité stratégique ou une
utilité d'entraînement.

## 6. Clarification des corpus 10k/100k/110k/290k/400k

Ces corpus sont analysés séparément comme patrimoine **synthétique**, jamais
inclus dans `D_REAL` :

| Corpus | Enregistrements | États uniques | Relation mesurée |
|---|---:|---:|---|
| 10k | 10 000 | 9 932 | entièrement présent dans 110k et 400k |
| 100k | 100 000 | 99 013 | entièrement présent dans 110k et 400k |
| 110k | 110 000 | 108 713 | fusion 10k+100k ; entièrement présent dans 400k |
| 290k | 290 000 | 285 858 | entièrement présent dans 400k |
| 400k | 392 118 | 392 118 | union dédupliquée 110k+290k |

Le 10k et le 100k partagent 232 états exacts. Le 110k et le 290k en partagent
2 453. Ces résultats confirment la lignée indiquée par les manifests et évitent
de compter les versions fusionnées comme des sources indépendantes.

## 7. Usages techniquement possibles

| Usage | Statut | Justification |
|---|---|---|
| A. Benchmark de positions | oui, avec déduplication | état physique et légalité disponibles |
| B. Imitation du coup humain | techniquement oui | action présente, mais elle n'est pas une vérité experte |
| C. Value depuis résultat réel | partiel/non direct | résultat seulement sur les coups terminaux stockés |
| D. Reconstruction de trajectoires | partielle | 68 matchs multi-coups, majorité de coups isolés |
| E. Départs de self-play | techniquement oui | 575 états valides, sous réserve du biais de collecte |
| F. Réanalyse MCTS | oui à petite échelle | états et masques légaux disponibles |

Aucun de ces usages n'a été exécuté pendant le Lot 16.

## 8. Performance et limites

L'audit utilise un parsing streaming pour JSONL et des ensembles d'identités
hachées pour les intersections. Le corpus réel et la vérification séparée de
la lignée synthétique ont été traités en 8,27 s dans la réexécution finale. Le
pic RSS rapporté par le système (environ 960 Mo)
inclut les trois réseaux et leurs diagnostics déjà chargés ; il ne représente
pas la mémoire propre du seul audit.

La limite principale est documentaire : les « environ 25 datasets » ou leurs
exports bruts ne sont pas présents. Le rapport ne peut donc ni inventer leur
inventaire, ni mesurer des chevauchements avec des fichiers absents.

## 9. Croisement prudent avec le Lot 16A

Le Lot 16A localise le blocage principal dans la Policy. `D_REAL` contient des
actions humaines et 523 états absents du self-play, mais seulement 575 états
uniques et de forts biais P1/action/terminal. Cela rend possible une future
analyse humain/G2/MCTS ou une petite réanalyse MCTS, pas une recommandation
d'entraînement immédiat.

La recommandation principale reste donc **POLICY_FOCUSED_EXPERIMENT** sur le
corpus immutable G2→G3. Une réanalyse des données réelles est une piste
secondaire, à envisager seulement après avoir compris pourquoi l'imitation
moyenne de π_MCTS64 dégrade la Policy en jeu.

## Artefacts

- `data/experiments/lot16_real_datasets/inventory.json`
- `data/experiments/lot16_real_datasets/schemas.json`
- `data/experiments/lot16_real_datasets/integrity.json`
- `data/experiments/lot16_real_datasets/overlaps.json`
- `data/experiments/lot16_real_datasets/overlap_matrix.csv`
- `data/experiments/lot16_real_datasets/union_statistics.json`
- `data/experiments/lot16_real_datasets/selfplay_comparison.json`
- `data/experiments/lot16_real_datasets/synthetic_lineage.json`
- `data/experiments/lot16_real_datasets/performance.json`
