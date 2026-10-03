# Lot 18 — Cartographie et audit de `D_TEACHER`

## 1. Objet et périmètre

Ce lot inventorie les corpus synthétiques historiques, mesure leur diversité réelle et évalue leur intérêt comme **source d'états** pour une future expérience Policy. Aucun entraînement, aucune réanalyse MCTS massive et aucune modification du moteur, du SRN ou des checkpoints n'ont été effectués.

`D_TEACHER` exclut explicitement `data/d_rl`, `data/real_matches`, les arènes, checkpoints, caches, rapports et fichiers temporaires. L'identité utilisée est strictement `(board[16], player_to_move)` : aucune symétrie P1/P2, aucun miroir et aucune augmentation.

La distinction structurante est la suivante :

```text
exemple historique = POSITION S + métadonnées + ANNOTATION TEACHER
```

Une position n'est pas une connaissance Minimax. Elle peut être réutilisée comme état tout en ignorant `best_action`, `action_values`, score, PV, profondeur et autres labels historiques.

## 2. Méthode reproductible

Le script [`run_srn_lot18.py`](../apps/trainer/scripts/run_srn_lot18.py) :

1. découvre les corpus à partir des manifests et des fichiers de données ;
2. traite les JSONL en streaming et les NPZ split par split ;
3. vérifie dimensions, entiers, conservation des 70 graines, joueur, masque légal et action Teacher ;
4. déduplique par identité physique exacte et conserve la multiplicité des provenances ;
5. agrège les conflits d'annotation dans SQLite, sans recopier le corpus ;
6. compare l'union unique à `D_RL` et `D_REAL` ;
7. exécute uniquement une inférence G2/G3-B sur 10 000 positions sélectionnées par `SHA256(seed || position)`.

L'audit final a traité 1 786 396 enregistrements en environ 75 secondes. Le pic mémoire observé par le système est d'environ 1,24 Go sur macOS. Aucun fichier géant `D_TEACHER_UNIQUE` n'a été créé.

## 3. Inventaire

L'inventaire contient 46 fichiers physiques, dont 34 fichiers de positions et 12 manifests, regroupés en **12 corpus logiques** :

| Corpus logique | Brut | Unique | Observation |
|---|---:|---:|---|
| `dataset_full_matrix_merged_all_colabs` | 883 422 | 552 982 | parties synthétiques multi-Colab, annotations Minimax `insane` |
| `dataset_v001_10k` | 10 000 | 9 932 | historique Teacher |
| `dataset_v002_100k` | 100 000 | 99 013 | historique Teacher |
| `dataset_v003_110k` | 110 000 | 108 713 | fusion 10k + 100k |
| `dataset_v004_290k` | 290 000 | 285 858 | historique Teacher |
| `dataset_v005_400k` | 392 118 | 392 118 | fusion dédupliquée 110k + 290k |
| `dataset_curated_endgames` | 6 | 6 | positions de fins documentées |
| `datasets/dataset_s2026_50` | 50 | 50 | petit corpus historique |
| `pilot_200` | 200 | 200 | pilote enrichi |
| `pilot_200_v2` | 200 | 200 | mêmes états, annotation différente |
| `pilot_fastengine_check` | 200 | 200 | mêmes états |
| `pilot_mp_test` | 200 | 200 | mêmes états |

Les hashes SHA256, tailles, chemins, manifests et schémas exacts figurent dans `inventory.json`, `logical_corpora.json` et `schemas.json`.

### Limite d'identité historique

Les JSONL internes stockent un plateau déjà ramené au point de vue du joueur courant, implicitement placé en P1. L'identité physique originale du joueur n'y est plus reconstructible. Cette limite concerne 392 362 positions uniques. L'audit ne les a pas re-canonicalisées : il compare la représentation effectivement stockée. Le corpus matriciel NPZ, lui, permet de retrouver le joueur par confrontation du masque avec le moteur.

## 4. Intégrité et annotations

Toutes les lignes auditées satisfont les invariants disponibles dans leur format : plateau de 16 valeurs entières non négatives, total de 70 graines, joueur valide, masque conforme au moteur et action annotée légale. Le champ historique `is_exact` signifie seulement que la meilleure PV rejouée atteint un terminal ; il ne constitue pas une preuve minimax formelle.

Les annotations sont hétérogènes. Les grands JSONL v001–v005 n'annotent en pratique qu'une action par ligne, alors que le corpus matriciel contient une distribution de politique complète et les pilotes peuvent contenir plusieurs `action_values`. La marge Teacher reste donc une statistique historique et ne doit être assimilée ni au regret autonome du Lot 17, ni à une vérité stratégique, ni à un poids de loss.

Sur l'union des répétitions inter-corpus :

- 444 626 positions ont plusieurs annotations ;
- 25 779 présentent au moins une variation d'annotation ;
- 7 846 présentent un `best_action`/`policy_index` différent ;
- 366 sont associées à une profondeur différente ;
- 3 937 sont associées à un tier différent ;
- 392 308 répétitions impliquent des valeurs d'actions incomplètes.

Ces conflits interdisent de traiter silencieusement l'annotation Teacher comme une cible absolue. Ils n'empêchent pas d'utiliser l'état seul.

## 5. Déduplication et relations entre versions

La somme naïve donne **1 786 396** positions, contre **941 599** identités uniques : 844 797 occurrences sont redondantes, soit un taux global de duplication de **47,29 %**.

Les noms ne décrivent pas une simple suite cumulative :

- `10k ⊄ 100k` : seulement 232 positions communes ;
- `10k ⊂ 110k` : 9 932 / 9 932 ;
- `100k ⊂ 110k` : 99 013 / 99 013 ;
- `110k ⊄ 290k` : 2 453 positions communes ;
- `110k ⊂ 400k` : 108 713 / 108 713 ;
- `290k ⊂ 400k` : 285 858 / 285 858 ;
- les quatre corpus pilotes contiennent exactement les mêmes 200 positions ;
- le grand corpus matriciel est largement indépendant du `400k` : seulement 3 740 intersections, soit 0,68 % du matriciel et 0,95 % du `400k`.

Le verdict est donc `TEACHER_CORPORA_LARGELY_NESTED = PARTIAL` : certaines releases sont des fusions exactes, mais le patrimoine complet contient aussi un bloc matriciel massif presque indépendant.

## 6. Diversité des états

L'union unique couvre les deux joueurs (667 567 positions P1 et 274 032 P2) et tout l'éventail de 1 à 7 actions légales. Elle comporte en moyenne 28,63 graines encore en jeu, 8,17 cases non vides et des magasins couvrant 0 à 35. Ces distributions montrent une diversité structurelle large ; elles ne constituent pas une taxonomie de phases stratégiques.

### Comparaison avec `D_RL`

| Référence | Unique | Intersection avec `D_TEACHER` | Couverture de `D_TEACHER` |
|---|---:|---:|---:|
| G1→G2 | 41 046 | 951 | 0,101 % |
| G2→G3 | 41 073 | 958 | 0,102 % |
| union `D_RL` | 81 668 | 1 762 | 0,187 % |

Ainsi, **939 837** positions Teacher uniques sont absentes du self-play actuel. La comparaison est exacte dans les coordonnées stockées, sous la limite de perspective signalée plus haut.

### Comparaison avec `D_REAL`

Sur les 575 positions physiques uniques de `data/real_matches`, 53 sont communes, 522 sont propres à `D_REAL` et **941 546** sont propres à `D_TEACHER`. `D_REAL` demeure un corpus séparé et n'a pas été mélangé à l'audit Teacher.

## 7. Diagnostic Policy G2 / G3-B

Sur l'échantillon reproductible de 10 000 positions, G2 présente une forte incertitude locale : environ 90 % des positions ont une marge top1–top2 inférieure à 0,10. G2 et G3-B conservent des distributions globalement proches, mais leur argmax n'est identique que sur environ 70 % des positions ; le recouvrement top-2 est voisin de 82 % et la divergence JS moyenne reste très faible (environ 0,00028).

Le motif du Lot 17 apparaît donc aussi hors de `D_RL` : de petites variations de probabilité suffisent fréquemment à renverser le classement lorsque la marge est faible. Ce diagnostic ne compare pas les actions au Teacher et ne mesure aucun « regret Teacher ».

## 8. Capacité de réutilisation

Les **941 599** positions valides sont réutilisables sans leurs labels Teacher comme candidats à une réanalyse autonome. Cette affirmation ne signifie pas qu'elles doivent toutes entrer dans un entraînement. Le flux futur recommandé est :

```text
déduplication et sélection stratifiée
    → suppression logique des labels Teacher
    → réanalyse G2 + MCTS autonome
    → mesure de stabilité des cibles
    → expérience Policy contrôlée
```

Trois scénarios restent conceptuellement possibles : A) réanalyse autonome, recommandée ; B) distillation Teacher, techniquement possible mais non autonome ; C) benchmark seul. Aucun n'est exécuté dans ce lot.

Un futur sous-ensemble devrait être construit sur la nouveauté vis-à-vis de `D_RL`, le nombre d'actions légales, les magasins, les graines en jeu et le joueur, jamais sur `best_action Teacher` considéré comme vérité.

## 9. Verdicts obligatoires

```text
D_TEACHER_FOUND = YES
D_TEACHER_POSITION_DIVERSITY = HIGH
D_TEACHER_ADDS_NEW_STATES_VS_DRL = YES
TEACHER_CORPORA_LARGELY_NESTED = PARTIAL
POSITIONS_REUSABLE_WITHOUT_TEACHER_LABELS = YES
AUTONOMOUS_REANALYSIS_FEASIBLE = YES
```

Réponse stratégique : **YES**. Les corpus historiques apportent suffisamment de positions nouvelles et diversifiées pour constituer une source d'états pertinente : 939 837 des 941 599 positions uniques sont absentes du self-play actuel. Cette conclusion porte sur les **positions**, pas sur la fiabilité des labels Minimax.

## 10. Expérience suivante unique

`NEXT_EXPERIMENT = DIVERSE_POSITION_REANALYSIS_FIRST`

Le prochain lot doit sélectionner un sous-ensemble dédupliqué et stratifié de positions nouvelles, ignorer toutes les annotations Teacher, produire pour chaque position plusieurs réanalyses G2+MCTS avec seeds contrôlées, puis ne conserver pour l'expérience Policy que des cibles dont la stabilité est mesurée. Ce choix précède la loss pondérée par regret : le Lot 17 a montré que le bruit MCTS existe, et pondérer fortement des cibles instables risquerait d'amplifier ce bruit.

## 11. Réponses aux six questions finales

1. **Combien de corpus ?** 12 corpus logiques, représentés par 46 fichiers physiques.
2. **Combien de positions uniques ?** 941 599, pour 1 786 396 occurrences brutes.
3. **Sont-ils emboîtés ?** Partiellement : 110k fusionne 10k et 100k, 400k fusionne 110k et 290k, mais le corpus matriciel apporte 549 242 positions qui ne figurent pas dans 400k.
4. **Combien sont nouvelles face au self-play ?** 939 837, soit 99,813 % de l'union Teacher.
5. **Peut-on ignorer Minimax ?** Oui : les 941 599 états sont séparables des annotations, avec une limite de perspective documentée sur 392 362 états historiques.
6. **Quelle expérience maintenant ?** Une réanalyse autonome diversifiée et contrôlée d'abord, avant tout entraînement Policy pondéré.

## 12. Artefacts

Les résultats machine lisibles sont regroupés sous [`data/experiments/lot18_d_teacher`](../data/experiments/lot18_d_teacher), notamment l'inventaire, les schémas, l'intégrité, les doublons, la matrice complète de chevauchement, les comparaisons `D_RL`/`D_REAL`, leurs distributions structurelles, les statistiques Teacher, les conflits, le diagnostic G2/G3-B et le rapport consolidé.
