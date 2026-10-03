# Lot 25 — Large-Scale Autonomous Data Expansion

## Résultat

Le Lot 25 construit et fige `D_SCALE_V1` sans entraîner ni promouvoir de G3. G2-best est l'unique générateur. Aucun label Teacher, aucun Minimax, aucune canonicalisation et aucune augmentation miroir ne sont utilisés.

| Couche | Volume | Rôle futur |
|---|---:|---|
| D_SELFPLAY_LARGE | 5 000 parties, 441 563 positions raw | Policy et Value terminale réelle |
| D_REANALYSIS_LARGE | 100 000 positions uniques | Policy MCTS128 uniquement |
| D_STRATEGIC_SAMPLE | 2 000 positions Qdiag256 | signal de ranking stratégique uniquement |

Le corpus contient 543 563 exemples logiques, 515 233 états physiques uniques et 500 851 états absents des anciens D_RL et D_REANALYSIS20K. Son empreinte de manifest est `76015ccd6cd0e49fd506eb7c4d0c795e23e93803e2654401efabe55f0898b1ee`.

## Self-play autonome

La génération emploie G2-best, MCTS64, `c_puct=1.5`, bruit Dirichlet `(alpha=0.3, epsilon=0.25)`, température 1 jusqu'au ply 30 puis 0. Les 5 000 parties donnent 2 616 victoires P1, 2 197 victoires P2, 182 nulles et 5 troncatures. La longueur moyenne est 88,31 coups.

Parmi les 441 563 positions, 415 399 sont uniques, soit 5,93 % de répétitions conservées volontairement. Les cibles ont un support moyen de 4,41 actions, une entropie moyenne de 1,274 et seulement 5,05 % de one-hot. Les sept actions locales sont toutes substantiellement représentées.

## Réanalyse autonome

Le réservoir Teacher a uniquement fourni les états physiques. La sélection intervient après déduplication des 941 599 positions et stratifie joueur, graines en jeu, magasins, nombre de coups légaux, cases non vides et provenance. Sur les 100 000 positions retenues, 99 938 sont absentes des anciens D_RL.

Chaque cible est produite par G2 + MCTS128, Dirichlet désactivé. Le corpus est strictement Policy-only : aucun `z`, root value ou score Teacher n'est transformé en cible Value. Le support moyen vaut 3,95, l'entropie moyenne 1,130 et le taux one-hot 11,95 %.

Le contrôle MCTS128/256 sur 1 000 positions donne 77,5 % d'accord argmax, 85,2 % d'accord top-2 et une divergence JS moyenne de 0,00461. La distribution complète est proche malgré des changements de top-1; ces cas sont des candidats naturels à une réanalyse adaptative future.

Le contrôle ciblé Qdiag256/512 sur 200 positions donne 74,0 % d'accord top-1 et un écart absolu moyen de 0,01235 sur la valeur de l'action préférée par Qdiag256. Qdiag demeure donc un signal autonome utile mais bruité, et non un oracle exact.

## Diversité et saturation

| Taille | Bins structurels couverts | États nouveaux vs anciens entraînements |
|---:|---:|---:|
| 20 000 | 1 918 | 17 708 |
| 50 000 | 2 130 | 44 204 |
| 100 000 | 2 245 | 88 512 |

La nouveauté continue de croître presque linéairement, tandis que la couverture des bins commence à ralentir. Pour cette première version, 100k offre donc le meilleur compromis vérifié. Passer immédiatement à 250k coûterait environ 3,52 h de recherche contre 1,41 h pour 100k, sans preuve expérimentale suffisante d'un gain proportionnel.

Le corpus couvre exactement 75 des positions du petit benchmark D_REAL. Les décisions humaines ne sont pas utilisées comme vérité stratégique.

## Coût et reproductibilité

- Self-play : 11 417,8 secondes cumulées, environ 1 576 parties/heure.
- Réanalyse : 5 071,1 secondes cumulées, environ 19,72 positions/seconde.
- Taille disque des trois couches : 576 202 528 octets.
- 151 shards ou fichiers de données, tous associés à une provenance et une empreinte.
- Seed de sélection et de génération : `20262525`.
- Fingerprint de configuration : `36dceb48592edf28dc28110c21ff1bf3542f3965c7bd56200332f31d9239f01e`.

## Séparation des sources pour l'entraînement futur

Les trois couches ne devront pas être concaténées uniformément. D_SELFPLAY_LARGE autorise Policy et Value; D_REANALYSIS_LARGE autorise uniquement la Policy MCTS128; D_STRATEGIC_SAMPLE autorise uniquement les objectifs stratégiques pairwise. Les ratios d'échantillonnage devront être calibrés pendant le futur entraînement G3.

## Verdicts

```text
SELFPLAY_SCALE_VALID = YES
REANALYSIS_SCALE_VALID = YES
TEACHER_LABELS_FULLY_EXCLUDED = YES
AUTONOMOUS_TARGET_QUALITY_ACCEPTABLE = YES
STATE_COVERAGE_MATERIALLY_EXPANDED = YES
STRUCTURAL_DIVERSITY_IMPROVED = YES
REANALYSIS_128_STABLE_ENOUGH_AT_SCALE = YES
STRATEGIC_SAMPLE_VALID = YES
DATA_SCALE_BOTTLENECK_CONFIRMED = INCONCLUSIVE
D_SCALE_V1_READY = YES
RECOMMENDED_REANALYSIS_SCALE = 100K
NEXT_ACTION = LARGE_SCALE_G3_TRAINING
```

## Réponses finales

1. 5 000 nouvelles parties autonomes ont été générées.
2. Le self-play apporte 415 399 positions uniques; D_SCALE_V1 contient 500 851 états nouveaux face aux anciens corpus d'entraînement.
3. 100 000 positions historiques ont été réanalysées sans annotation Teacher.
4. MCTS128/256 atteint 77,5 % d'accord argmax, 85,2 % top-2 et JS moyenne 0,00461.
5. Les niveaux 20k/50k/100k apportent respectivement 17 708, 44 204 et 88 512 états nouveaux, avec 1 918, 2 130 et 2 245 bins structurels.
6. Oui, self-play et réanalyse couvrent des distributions distinctes et complémentaires; la matrice exacte est conservée dans `overlap_matrix.json`.
7. Le volume recommandé est 100k : il matérialise une forte expansion à coût contrôlé; 250k reste une extension ultérieure, pas une nécessité démontrée.
8. Oui, D_SCALE_V1 est prêt et immuable pour les expériences suivantes.
9. Oui, le projet peut passer à un entraînement G3 à grande échelle, sans présumer que davantage de données garantira une amélioration.

Réponse à la question scientifique finale : **YES**. Le projet dispose désormais d'une base autonome assez grande, diverse et contrôlée pour tester sérieusement une nouvelle génération G3.
