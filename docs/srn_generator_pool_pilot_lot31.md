# Lot 31 — Pilote du pool de générateurs

## 1. Objet et périmètre

Le Lot 31 teste, sans entraînement, si un pool figé composé de **G2** et de **G3_VALUE_REWORK** produit à budget égal une matière autonome plus diverse que le self-play G2 seul. Aucun gradient, checkpoint G4, label Teacher/Minimax, changement du moteur, de MCTS ou du SRN n'a été introduit.

G2 reste le champion officiel. G3_VALUE_REWORK combine la Policy de G3_STRATEGIC et la Value de V28_A ; il n'intervient ici que comme générateur probatoire. Les chemins, empreintes SHA-256, empreintes Policy/Value et l'empreinte d'architecture sont consignés dans `model_identity.json`.

## 2. Protocole figé

Deux bras de 400 parties ont été produits :

- CONTROL : 400 G2–G2 ;
- POOL : 100 G2–G2, 100 G3–G3 et 200 cross-play, équilibrés en 100 G2(P1)–G3(P2) et 100 G3(P1)–G2(P2).

L'ordonnancement est déterministe. Les deux bras utilisent 64 simulations MCTS par position, `c_puct=1.5`, bruit Dirichlet `(alpha=0.3, epsilon=0.25)`, température d'action 1 avant le pli 30 puis 0, limite de 400 plis et détection de répétition à 3. Les états physiques ne sont ni canonisés ni augmentés.

L'égalité expérimentale signifie ici : même nombre de parties et même budget par décision. Le total de simulations peut différer puisque la longueur des parties est un résultat de l'expérience.

## 3. Règles de décision préenregistrées

Avant l'analyse, les seuils suivants ont été enregistrés dans `configuration.json` : gain normalisé matériel ≥ 2 %, gain relatif de désaccord stratégique ≥ 5 %, part d'états propres au cross-play ≥ 2 %, source pathologique au-delà de 5 % de troncatures ou de +2 points face au contrôle, coût acceptable si le ratio de temps par simulation reste ≤ 2, et forte divergence Policy à JS ≥ 0,001.

La diversité utile exige simultanément un gain matériel de couverture, une nouveauté historique ou stratégique mesurable, un apport cross-play non trivial, l'absence de pathologie et un coût acceptable. Les seuils ne sont pas réajustés après observation.

## 4. Intégrité et provenance

Les 800 parties prévues ont été générées. Chaque position conserve le bras, la génération, l'identité de partie, les modèles et rôles P1/P2, leurs empreintes, le joueur au trait, le budget MCTS, la graine, le masque légal, les visites, le résultat terminal et la provenance source.

CONTROL contient 399 parties terminales et une troncature par répétition ; les 161 positions de cette trajectoire n'ont aucun faux `z`. POOL contient 400 parties terminales et aucune troncature. Dans tous les autres cas, `z` provient exclusivement du résultat réel.

## 5. Couverture principale

| Mesure | CONTROL | POOL |
|---|---:|---:|
| Parties | 400 | 400 |
| Positions brutes | 36 925 | 36 157 |
| États physiques uniques | 35 511 | 34 782 |
| Taux de duplication | 3,829 % | 3,803 % |
| États uniques / 10 000 positions | 9 617,06 | 9 619,71 |
| Nouveaux états vs historique | 34 276 | 33 667 |
| Nouveaux / 10 000 positions | 9 282,60 | 9 311,34 |

POOL produit 729 états uniques bruts de moins, en cohérence avec ses 768 positions brutes de moins. Après normalisation, son avantage de couverture n'est que de **+0,0276 %**, très inférieur au seuil matériel de 2 %. Son gain normalisé de nouveauté historique est de **+0,310 %**, lui aussi inférieur au seuil.

Les deux bras se recouvrent peu : 359 états communs, 34 423 états propres à POOL et 35 152 propres à CONTROL. Cela montre une forte stochasticité des trajectoires, mais pas à elle seule une meilleure qualité du pool.

## 6. Ouvertures, milieux et fins de partie

La phase d'ouverture est définie par `ply < 20`, la fin par `seeds_in_play <= 20`, et le reste comme milieu. Ce découpage repose uniquement sur des variables brutes.

| Phase | CONTROL brut / unique | POOL brut / unique |
|---|---:|---:|
| Ouverture | 8 000 / 6 624 | 8 000 / 6 636 |
| Milieu | 14 104 / 14 104 | 13 917 / 13 917 |
| Fin | 14 821 / 14 783 | 14 240 / 14 229 |

Chaque bras produit 400 signatures de trajectoire et 400 préfixes de dix coups distincts. La longueur moyenne est de 92,31 plis pour CONTROL et 90,39 pour POOL ; les médianes sont respectivement 73,5 et 78. Les distributions structurelles par graines en jeu, écart des magasins, profondeur et nombre de coups légaux couvrent les mêmes familles de bins. La nouveauté de POOL ne provient donc pas d'une seule région triviale manifeste.

CONTROL termine par 384 victoires, 15 nulles et une troncature. POOL termine par 389 victoires et 11 nulles. Les magasins finaux moyens sont proches : 36,15/33,85 pour CONTROL et 35,31/34,69 pour POOL.

## 7. Actions et signal stratégique

L'entropie des actions jouées est presque identique : 1,94367 pour CONTROL et 1,94346 pour POOL. Les fréquences des sept actions restent proches et le nombre moyen de coups légaux est de 4,40 contre 4,45. Il n'existe donc pas de dérive grossière vers une action ou des positions artificiellement simples.

Sur tous les états uniques, le désaccord top-1 entre les Policy G2 et G3 vaut 9,225 % dans CONTROL et 9,485 % dans POOL, soit **+2,81 % relatif**. Ce signal est inférieur au seuil préenregistré de 5 %. La JS moyenne diminue légèrement (0,0000992 vers 0,0000950) et la couverture des états à JS ≥ 0,001 passe de 0,177 % à 0,152 %. Le pool complet n'apporte donc pas un gain stratégique matériel démontré.

Cette analyse mesure des proxies — nouveauté, désaccord, entropie, complexité légale — et ne prétend pas prédire directement la force d'un futur modèle.

## 8. Contribution des sources du POOL

| Source | Brut | Unique | Propre aux autres sources | Nouveau vs historique | Désaccord top-1 |
|---|---:|---:|---:|---:|---:|
| G2–G2 | 8 945 | 8 675 | 8 559 | 8 325 | 8,634 % |
| G3–G3 | 9 501 | 9 230 | 9 104 | 8 917 | 9,642 % |
| Cross-play | 17 711 | 17 097 | 16 955 | 16 427 | 9,709 % |

Dans l'ordre d'ablation G2–G2, puis G3–G3, puis cross-play, G3–G3 ajoute 9 152 états marginaux (99,15 % de ses états) et le cross-play 16 955 (99,17 %). La meilleure contribution marginale absolue vient donc du **cross-play**. Celui-ci fait passer POOL_NO_CROSS de 17 827 à 34 782 états uniques.

Le cross-play obtient également davantage de désaccord top-1 que le G2–G2 interne au POOL (9,709 % contre 8,634 %). Il atteint donc des régions à la fois exclusives et légèrement plus conflictuelles entre Policy. Toutefois, sa JS moyenne et son taux de forte divergence ne dépassent pas ceux des sources homogènes : la différence stratégique reste modeste et ne suffit pas à qualifier tout le POOL de supérieur.

G3–G3 ne présente aucune troncature, apporte une grande nouveauté marginale et un désaccord top-1 supérieur au G2–G2 du même bras. Cela justifie `G3_SELFPLAY_ADDS_USEFUL_DIVERSITY=YES` au sens des proxies du pilote, mais pas une admission définitive du générateur.

## 9. Coût et replay simulé

CONTROL a consommé 2 363 200 simulations, soit 5 908 par partie, en 889,94 s. POOL en a consommé 2 314 048, soit 5 785,12 par partie, en 1 296,47 s. Le modèle hybride rend le temps par simulation **1,488 fois** plus élevé, sous la limite de 2. L'efficacité est pratiquement identique : 15 026,66 contre 15 030,80 états uniques par million de simulations.

Le replay simulé a demandé 1 200 exemples et obtenu exactement 400 exemples G2–G2, 400 G3–G3 et 400 cross-play. La provenance et les cibles Value sont conservées ; aucun gradient n'a été calculé. Le mécanisme est donc techniquement prêt pour une expérience ultérieure, sans autoriser ici un entraînement.

## 10. Interprétation et limites

Le pilote est valide, non pathologique et le coût est acceptable. Il démontre clairement que le cross-play produit des états que les deux self-play homogènes n'atteignent pas dans cet échantillon. En revanche, il ne démontre pas que remplacer une partie du G2–G2 par le pool augmente matériellement la couverture normalisée, la nouveauté historique ou la diversité stratégique globale selon les seuils fixés.

Le très faible chevauchement entre deux réalisations de 400 parties indique qu'une partie importante des états « propres » reflète aussi la stochasticité normale de MCTS. Une réplication ou une extension est nécessaire pour séparer plus fermement l'effet du mélange de générateurs de la variance d'échantillonnage. Le statut de G3 comme générateur reste donc **INCONCLUSIVE**, et non RETAIN ou REMOVE.

## 11. Verdicts obligatoires

```text
PILOT_VALID = YES
EQUAL_GENERATION_BUDGET = YES
POOL_INCREASES_UNIQUE_STATE_COVERAGE = NO
POOL_INCREASES_HISTORICAL_NOVELTY = NO
POOL_INCREASES_STRATEGIC_DIVERSITY = NO
CROSS_PLAY_ADDS_UNIQUE_STATES = YES
CROSS_PLAY_ADDS_STRATEGIC_DIVERSITY = YES
G3_SELFPLAY_ADDS_USEFUL_DIVERSITY = YES
POOL_DATA_PATHOLOGICAL = NO
COMPUTE_COST_ACCEPTABLE = YES
SOURCE_BALANCING_VALID = YES
REPLAY_STRATIFICATION_VALID = YES
POOL_DATASET_READY_FOR_TRAINING = YES
G3_VALUE_REWORK_GENERATOR_STATUS = INCONCLUSIVE
OFFICIAL_CHAMPION = G2
G3_PROMOTED = NO
NEXT_ACTION = GENERATOR_POOL_PILOT_EXTENSION
```

## 12. Réponses synthétiques aux seize questions

1. CONTROL produit 36 925 positions brutes ; POOL 36 157.
2. CONTROL contient 35 511 états physiques uniques ; POOL 34 782.
3. Normalisé au volume, POOL couvre très légèrement plus : 9 619,71 contre 9 617,06 par 10 000, mais le gain de 0,0276 % n'est pas matériel.
4. 34 423 états POOL sont absents de CONTROL.
5. 33 667 états POOL et 34 276 états CONTROL sont nouveaux face aux 634 192 états historiques indexés.
6. G3–G3 apporte 9 152 états marginaux après G2–G2, dont 8 917 nouveaux face à l'historique.
7. Le cross-play apporte 16 955 états marginaux après les deux self-play homogènes, dont 16 427 nouveaux face à l'historique.
8. Oui : 99,17 % des états cross-play sont absents des deux sources homogènes dans ce pilote.
9. Le cross-play augmente légèrement le désaccord top-1, mais le POOL global ne franchit pas le seuil stratégique ; le signal reste insuffisant pour exclure un effet de variance/bruit.
10. Le cross-play apporte la meilleure nouveauté marginale absolue.
11. Oui : le ratio de temps par simulation vaut 1,488, sous le plafond 2.
12. Oui : le replay stratifié retourne exactement 400 exemples par source et conserve la provenance.
13. Oui, les données sont techniquement prêtes ; cela n'autorise pas encore un entraînement.
14. `G3_VALUE_REWORK_GENERATOR_STATUS = INCONCLUSIVE`.
15. `OFFICIAL_CHAMPION = G2` et `G3_PROMOTED = NO`.
16. `NEXT_ACTION = GENERATOR_POOL_PILOT_EXTENSION`.

## 13. Artefacts de référence

Les résultats détaillés et auditables se trouvent dans `data/experiments/lot31_generator_pool/`. `report.json` agrège les résultats, tandis que les fichiers spécialisés conservent la couverture, la nouveauté historique, les trajectoires, les actions, le désaccord stratégique, l'ablation cross-play, les contributions, le coût, le replay et la décision. Les shards JSONL et leurs manifests demeurent séparés sous `control/` et `pool/`.
