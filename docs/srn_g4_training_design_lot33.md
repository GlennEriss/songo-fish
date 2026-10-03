# Lot 33 — Conception de l'entraînement G4

## 1. Décision de conception

Le Lot 33 définit la première génération du paradigme `CHAMPION_GATED_DIVERSE_GENERATOR_POOL`. Il ne génère pas le corpus massif et n'entraîne aucun modèle.

Le design retenu repose sur cinq décisions :

1. génération opérationnelle cross-play enrichie à 10/10/80 ;
2. replay équilibré par génération et stratifié par source ;
3. initialisation Policy G3_STRATEGIC et Value V28_A ;
4. objectifs neuronaux autonomes minimaux, sans nouvelle loss stratégique ;
5. sélection séparée des composants suivie d'une évaluation multi-adversaires.

Les statuts officiels ne changent pas : G2 reste champion, G3 n'est pas promu et G3_VALUE_REWORK demeure générateur complémentaire retenu.

## 2. Corpus généré D_G4_GENERATED

Le futur corpus principal comportera **8 000 nouvelles parties** :

| Source | Part | Parties |
|---|---:|---:|
| G2–G2 | 10 % | 800 |
| G3–G3 | 10 % | 800 |
| G2–G3 | 80 % | 6 400 |

Le cross-play est équilibré : 3 200 parties avec G2 en P1 et 3 200 avec G3 en P1. Le ratio 10/10/80 est une baseline opérationnelle issue du Lot 32, pas un optimum revendiqué.

Huit mille parties représentent environ 727 752 positions et 46,6 millions de simulations MCTS selon les mesures locales du Lot 32. Ce volume dépasse les 5 000 parties du Lot 25 sans augmenter arbitrairement d'un ordre de grandeur ; il fournit 6 400 trajectoires cross-play, soit quatre fois le bras cross-play enrichi du Lot 32.

## 3. Composition du corpus d'entraînement

Le replay logique utilise les proportions suivantes :

| Couche | Ratio d'échantillonnage | Rôle |
|---|---:|---|
| Nouvelle génération G4 | 70 % | Policy et Value terminale |
| Historique RL autonome | 20 % | Préservation et lutte contre l'oubli |
| Réanalyse autonome | 10 % | Policy uniquement |

L'historique est conservé parce que les Lots 20–24 ont montré qu'une amélioration locale peut détruire des relations déjà correctes. Il est plafonné à 20 % et équilibré entre générations : son volume physique ne peut donc pas écraser le nouveau signal cross-play.

La réanalyse porte sur 40 000 états physiques. G2 et G3_VALUE_REWORK produisent chacun une cible MCTS128 sans bruit sur les mêmes états, soit 80 000 exemples Policy-only. Cette symétrie conserve les désaccords sans supposer qu'un générateur a raison. Les annotations Teacher restent interdites ; D_TEACHER ne peut fournir que des identités d'états.

D_REAL reste une batterie externe de 575 états. Qdiag reste un diagnostic ciblé, jamais le carburant principal.

## 4. Contrat source-aware

Chaque exemple doit conserver la génération, la partie, le type de source, les modèles et rôles P1/P2, le joueur au trait, l'état brut, le masque légal, les comptes de visites bruts, la configuration MCTS, la graine et les empreintes des checkpoints.

Les types retenus sont `G2_SELFPLAY`, `G3_SELFPLAY`, `G2_G3_CROSSPLAY`, `HISTORICAL_SELFPLAY` et `AUTONOMOUS_REANALYSIS`. `STRATEGIC_QDIAG` n'est pas une source d'entraînement baseline.

Le replay `GENERATION_BALANCED_SOURCE_STRATIFIED` alloue à chaque batch de 256 environ 179 exemples nouveaux, 51 historiques et 26 réanalysés. Dans la couche nouvelle, 18/18/143 emplacements approchent 10/10/80 ; un reste tournant déterministe garantit les ratios exacts à long terme. L'historique est d'abord équilibré entre générations, puis échantillonné.

Les positions de désaccord ne sont pas sur-échantillonnées. Le +8,04 % du Lot 32 prouve une diversité, pas la vérité de G2 ou G3.

## 5. Cibles et objectifs

### Policy

La cible principale reste la distribution MCTS reconstruite depuis les comptes de visites bruts :

`L_policy = -Σ_a π_MCTS(a|s) log pθ(a|s)`.

Les actions illégales sont masquées. La cible n'est ni le coup joué ni un best move one-hot. Aucune loss nouvelle de ranking n'est ajoutée : comme la variable expérimentale est déjà la distribution des données, conserver une CE autonome simple permet une attribution causale claire.

### Value

La Value minimise :

`L_value = (vθ(s) - z_terminal)²`.

Seuls les vrais résultats terminaux, dans la perspective du joueur au trait, sont autorisés. Les parties tronquées peuvent conserver leur cible Policy mais n'alimentent jamais la Value. La réanalyse, les scores Teacher/Minimax, les root values et tout pseudo-z sont exclus.

## 6. Initialisation et entraînement contrôlé

G4 est initialisé avec :

- `G4_INITIAL_POLICY = G3_STRATEGIC` ;
- `G4_INITIAL_VALUE = V28_A`.

Cette combinaison est celle qui a fourni le meilleur candidat récent sous recherche. Repartir de G2 ou de zéro abandonnerait sans justification le progrès Policy du Lot 27 et la stabilité Value du Lot 28.

CONTROL_G4 et POOL_G4 utilisent le même SRN, cette même initialisation, AdamW, un learning rate constant de 0,0003, weight decay 0,0001, batch 256, clipping à 1 et exactement 32 000 mises à jour. Un checkpoint est enregistré toutes les 4 000 mises à jour. Aucun early stopping ne réduit le budget d'un seul bras.

- **CONTROL_G4** : 8 000 nouvelles parties G2–G2, 70 % de nouvelles données G2-only, les mêmes 20 % historiques et 10 % de réanalyse G2-only.
- **POOL_G4** : 8 000 parties 10/10/80, 70 % de nouvelles données cross-play enrichies, les mêmes 20 % historiques et 10 % de réanalyse duale G2/G3.

La seule variable conceptuelle est le paradigme de génération des données. Architecture, initialisation, optimiseur, mises à jour, splits, cadence de checkpoints et évaluation restent fixes.

## 7. Splits et absence de fuite

Les splits sont réalisés par `game_id` : 85 % TRAIN, 10 % VALIDATION et 5 % STRATEGIC_TEST. Toutes les positions d'une partie cross-play restent ensemble. Les états réanalysés sont séparés par identité physique.

VALIDATION sélectionne les checkpoints. STRATEGIC_TEST mesure préservation, correction, calibration et comportement sur 2 000 désaccords G2/G3, 2 000 états historiques, 575 états D_REAL et 2 000 états structurellement divers. ARENA est exclusivement réservée à la force de jeu et à la promotion. Les ouvertures d'arène ne servent jamais à choisir un checkpoint.

## 8. Sélection Policy et Value

La sélection est découplée : au maximum deux composants Policy et deux composants Value, soit quatre combinaisons `P1V1`, `P1V2`, `P2V1`, `P2V2`.

Les gates offline éliminent seulement les corruptions et catastrophes : NaN, masse illégale, régression CE ou MSE supérieure à 10 %, préservation stratégique sous 85 %, ou écart-type Value sous 0,05. Ils ne prouvent jamais la force de jeu.

Les Policy restantes sont comparées sur CE, préservation, désaccord et comportement MCTS64. Les Value sont comparées sur MSE, signe, calibration et stabilité sous recherche. Une petite arène MCTS64 élimine seulement les combinaisons manifestement mauvaises ; elle ne choisit pas finement l'epoch gagnant.

Ce protocole évite Lot20 bis : aucune amélioration offline ne suffit. Il évite aussi Lot23 bis : la préservation des relations déjà correctes est un gate explicite.

## 9. Batterie d'adversaires

La batterie principale contient exactement :

1. G2, gate du champion ;
2. G3_VALUE_REWORK, gate de robustesse face au style complémentaire.

Toutes les arènes sont appariées et inversent les côtés. Les scores P1 et P2 sont rapportés séparément. Ajouter une grande league historique n'est pas justifié à ce stade.

Les budgets principaux sont MCTS64 et MCTS128. MCTS256 est une confirmation conditionnelle si le score agrégé est à moins de trois points de 50 %, si 64 et 128 diffèrent de plus de cinq points, ou si un signal de collapse apparaît.

## 10. Search robustness

La monotonie stricte n'est pas requise. Un candidat est robuste si aucun écart négatif supérieur à sept points entre budgets n'est reproduit sur un second ensemble apparié, si les intervalles restent compatibles avec la tendance globale et si aucun collapse récurrent n'apparaît.

Cette règle traite une fluctuation ponctuelle comme de l'incertitude à confirmer, et non comme une disqualification automatique.

## 11. Règle de promotion

La promotion exige simultanément :

- score agrégé MCTS64/128 contre G2 ≥ 52 % et probabilité bootstrap appariée `P(score>50 %) ≥ 95 %` ;
- score contre G3_VALUE_REWORK ≥ 48 %, avec borne basse IC95 ≥ 42 % ;
- écart absolu P1/P2 ≤ 10 points ;
- absence de baisse reproduite supérieure à sept points entre budgets ;
- préservation stratégique ≥ 85 % et aucune pathologie.

La borne basse de chaque budget individuel n'a pas besoin de dépasser 50 %. La règle combine progression champion, robustesse populationnelle, équilibre des côtés et recherche.

## 12. Admission comme générateur

Un candidat non promu peut devenir `GENERATOR_ONLY` s'il obtient au moins 42 % agrégé contre le champion, apporte soit ≥5 % de diversité stratégique soit ≥2 % de données marginales lors d'un pilote dédié, ne présente aucune pathologie et reste sous un ratio de coût ×2.

Cette décision est distincte de la promotion. Les statuts possibles sont `CHAMPION_AND_GENERATOR`, `GENERATOR_ONLY`, `REJECTED` et `INCONCLUSIVE`.

## 13. Succès du nouveau paradigme

`POOL_PARADIGM_ADDS_VALUE` sera positif si POOL_G4 dépasse CONTROL_G4 d'au moins trois points sur le score multi-adversaires agrégé avec `P(Δ>0) ≥ 95 %`, ou si POOL_G4 franchit la promotion alors que CONTROL_G4 échoue, sous réserve d'absence de collapse populationnel, de robustesse sous recherche et d'au moins 85 % de préservation.

Cette conclusion est séparée de `G4_IS_STRONGER`. Le nouveau corpus peut apporter de la valeur sans produire immédiatement un champion.

## 14. Budget prévisionnel

Les extrapolations locales du Lot 32 donnent :

| Travail futur | Positions/cibles attendues | Simulations | Durée locale estimée | Stockage JSONL |
|---|---:|---:|---:|---:|
| POOL_G4, 8 000 parties | 727 752 | 46,58 M | 7,71 h | 1,20 Gio |
| CONTROL_G4, 8 000 parties | 706 744 | 45,23 M | 4,69 h | 1,06 Gio |
| Réanalyse duale | 80 000 cibles | 10,24 M | 1,13 h | à mesurer |

La génération et la réanalyse totalisent environ 13,53 heures sur la machine observée. Il s'agit d'une extrapolation mesurée, pas d'une promesse portable. La durée d'entraînement devra être établie par un préflight ; elle n'est pas inventée dans ce lot.

## 15. Verdicts

```text
G4_DESIGN_VALID = YES
CROSSPLAY_ENRICHED_GENERATION_SELECTED = YES
SOURCE_AWARE_TRAINING_REQUIRED = YES
GENERATION_BALANCED_REPLAY_REQUIRED = YES
HISTORICAL_DATA_RETAINED = YES
AUTONOMOUS_REANALYSIS_RETAINED = YES
STRATEGIC_DISAGREEMENT_OVERSAMPLING = NO
POLICY_VALUE_SELECTION_DECOUPLED = YES
MULTI_OPPONENT_EVALUATION_REQUIRED = YES
STRICT_SEARCH_MONOTONICITY_REQUIRED = NO
G4_CONTROL_REQUIRED = YES

G4_NEW_GENERATION_GAMES = 8000
G4_GENERATION_RATIO = G2_G2 10%, G3_G3 10%, G2_G3 80%
G4_TRAINING_SOURCE_RATIO = NEW_GENERATION 70%, HISTORICAL_RL 20%, AUTONOMOUS_REANALYSIS 10%

G4_INITIAL_POLICY = G3_STRATEGIC
G4_INITIAL_VALUE = V28_A
G4_POLICY_OBJECTIVE = MASKED_SOFT_TARGET_CE_TO_PI_MCTS
G4_VALUE_OBJECTIVE = MSE_TO_TRUE_TERMINAL_Z_ONLY
G4_REPLAY_STRATEGY = GENERATION_BALANCED_SOURCE_STRATIFIED

MAX_POLICY_CANDIDATES = 2
MAX_VALUE_CANDIDATES = 2
MAX_POLICY_VALUE_COMBINATIONS = 4
G4_EVALUATION_OPPONENTS = [G2, G3_VALUE_REWORK]
PRIMARY_SEARCH_BUDGETS = [64, 128]
MCTS256 = CONDITIONAL_CONFIRMATION

OFFICIAL_CHAMPION = G2
G3_PROMOTED = NO
G3_VALUE_REWORK_GENERATOR_STATUS = RETAIN
NEXT_ACTION = G4_DATA_GENERATION_AND_CONTROLLED_TRAINING
```

## 16. Réponses aux vingt questions finales

1. Le nouveau corpus expérimental doit produire 8 000 parties par bras d'entraînement.
2. POOL_G4 utilise 10 % G2–G2, 10 % G3–G3 et 80 % G2–G3.
3. Le replay utilise 70 % de nouvelle génération, 20 % d'historique autonome et 10 % de réanalyse autonome.
4. L'historique est conservé pour limiter l'oubli et mesurer la préservation, mais plafonné à 20 %.
5. Le replay équilibre d'abord les générations et impose ensuite ses quotas de sources ; le volume brut historique ne décide pas de sa fréquence.
6. G4 est initialisé avec la Policy G3_STRATEGIC.
7. G4 est initialisé avec la Value V28_A.
8. La Policy minimise la CE masquée vers la distribution de visites MCTS complète.
9. La Value minimise la MSE vers le seul résultat terminal réel.
10. Oui. Deux Policy et deux Value au maximum sont sélectionnées séparément, puis quatre combinaisons au maximum sont évaluées.
11. Une amélioration offline ne permet jamais l'acceptation ; chaque candidat passe un gate MCTS puis une évaluation multi-adversaires.
12. CONTROL_G4 reçoit la même initialisation et le même budget, mais 8 000 nouvelles parties G2-only et une réanalyse G2-only.
13. POOL_G4 reçoit la même initialisation et le même budget, mais 8 000 parties 10/10/80 et une réanalyse duale G2/G3.
14. Les adversaires sont G2 et G3_VALUE_REWORK.
15. Search robustness signifie absence de collapse supérieur à sept points reproduit entre budgets, avec analyse des IC et de la tendance globale ; aucune monotonie stricte.
16. La promotion exige progression agrégée contre G2, robustesse contre G3, équilibre des côtés, robustesse recherche, préservation et absence de pathologie.
17. L'admission générateur exige une force minimale, une diversité ou valeur marginale mesurée, un coût ≤2× et aucune pathologie.
18. Le paradigme apporte de la valeur si POOL_G4 dépasse causalement CONTROL_G4 selon la règle préenregistrée, sans collapse ni destruction stratégique.
19. `OFFICIAL_CHAMPION = G2` ; `G3_PROMOTED = NO`.
20. `NEXT_ACTION = G4_DATA_GENERATION_AND_CONTROLLED_TRAINING`.
