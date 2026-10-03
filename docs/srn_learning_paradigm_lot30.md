# Lot 30 — Paradigme d’apprentissage autonome à long terme

## Décision

Le paradigme recommandé est :

```text
RECOMMENDED_LEARNING_PARADIGM = CHAMPION_GATED_DIVERSE_GENERATOR_POOL
```

Il sépare trois décisions qui étaient jusque-là confondues :

1. le **champion officiel**, référence de force et de déploiement ;
2. les **générateurs autorisés**, sources contrôlées de nouvelles trajectoires ;
3. les **candidats**, modèles entraînés qui doivent encore être évalués.

La décision du Lot 29 reste intacte :

```text
OFFICIAL_CHAMPION = G2
G3_PROMOTED = NO
```

G3_VALUE_REWORK est néanmoins admis comme générateur probatoire, parce qu’il produit un signal positif et reproductible sans pathologie majeure. Cette admission concerne uniquement la production de données diversifiées.

## Pourquoi le single-champion est seulement partiellement adéquat

La boucle `champion → self-play homogène → challenger → duel → promotion/rejet` a permis d’obtenir et confirmer G2. Elle apporte simplicité, stabilité, reproductibilité et maîtrise du coût. Elle reste donc utile comme mécanisme de contrôle.

Elle a toutefois atteint ses limites comme moteur exclusif d’évolution :

- le self-play reste centré sur la distribution stratégique de G2 ;
- les candidats apprennent de nouveaux signaux mais peuvent perdre des relations déjà correctes ;
- de petites divergences Policy produisent de grandes conséquences sous MCTS ;
- un candidat globalement intéressant peut être entièrement rejeté pour une faiblesse locale ;
- chaque génération a nécessité une nouvelle intervention manuelle : ranking, correct-and-preserve, recalibration ou réparation Value ;
- le progrès Policy et le progrès Value ne sont pas toujours synchrones.

Les Lots 14–29 indiquent donc un problème double : les sorties Policy/Value restent difficiles à optimiser, mais la manière d’organiser l’évolution amplifie ce problème en traitant toute progression comme un bloc binaire à promouvoir ou jeter.

## Comparaison des paradigmes

| Paradigme | Diversité | Stabilité | Autonomie | Coût relatif | Complexité | Oubli | Non-transitivité | Adéquation observée |
|---|---|---|---|---:|---|---|---|---|
| Single champion | faible | locale élevée | partielle | 1× | faible | faible protection | mauvaise | insuffisant seul |
| Champion + challenger | moyenne | moyenne | moyenne | 1,1–1,3× | faible-moyenne | limitée | limitée | amélioration utile, tunnel à deux modèles |
| Opponent pool | moyenne-haute | élevée | élevée | 1,2–1,5× | moyenne | bonne | bonne | très adaptée |
| Population self-play | haute | moyenne | élevée | 1,5–2,5× | élevée | bonne | bonne | utile mais trop large actuellement |
| League complète | très haute | potentiellement élevée | élevée | >2,5× | très élevée | très bonne | très bonne | disproportionnée |
| Mélange générationnel | moyenne-haute | élevée | élevée | 1,2–1,6× | moyenne | bonne | bonne | très adaptée |
| Distillation de population | haute à l’entraînement | moyenne-haute | moyenne | 1,5–2× | élevée | moyenne | moyenne | option ultérieure |
| Pool borné contrôlé par champion | haute mais bornée | élevée | élevée | cible 1,3–1,7× | moyenne | bonne | bonne | meilleur compromis |

La recommandation n’est donc pas une league complexe. Elle combine le minimum nécessaire : champion stable, petit pool de générateurs, adversaires historiques, cross-play et sélection populationnelle.

## Boucle recommandée

```text
OFFICIAL CHAMPION + AUTHORIZED GENERATOR POOL
                    ↓
      SELF-PLAY HOMOGÈNE + CROSS-PLAY
                    ↓
      D_RL_GENERATION_N AVEC PROVENANCE
                    ↓
   REPLAY STRATIFIÉ PAR SOURCE ET GÉNÉRATION
                    ↓
        ENTRAÎNEMENT SOURCE-AWARE
                    ↓
    POLICY/VALUE CANDIDATES PARTIELLEMENT SÉPARÉS
                    ↓
       PETIT ENSEMBLE D’AGENTS CANDIDATS
                    ↓
       ÉVALUATION MULTI-ADVERSAIRES
                    ↓
  PROMOTION CHAMPION / ADMISSION GENERATOR / REJET
                    ↓
       MISE À JOUR ET RETRAIT DU POOL
                    ↓
              GÉNÉRATION N+1
```

L’inférence produit reste inchangée : un agent compact compatible SRN, accompagné de MCTS. Le pool n’est utilisé que pour l’entraînement et l’évaluation ; il n’est pas embarqué dans Unity.

## Generator pool

### Admission

L’admission est plus faible que la promotion, mais elle n’est pas automatique. Un modèle doit satisfaire les portes suivantes :

- identité immuable et reproductible ;
- absence de pathologie moteur, recherche ou côté ;
- non-infériorité crédible sur plusieurs budgets ou complémentarité démontrée ;
- différence stratégique mesurable ;
- diversité utile : nouvelles trajectoires sans comportement faible ou aléatoire ;
- contribution d’abord bornée dans un pilote probatoire.

Les seuils numériques définitifs ne sont pas déduits après coup du cas G3. Ils devront être calibrés prospectivement à partir du pilote.

### Retrait

Un générateur est retiré s’il est durablement faible face à la population, redondant sans couverture nouvelle, obsolète, pathologique ou dominé sans diversité utile. Le pool reste volontairement petit : champion, générateurs récents complémentaires et quelques ancres historiques.

### Cas G3_VALUE_REWORK

G3_VALUE_REWORK satisfait les portes probatoires disponibles : résultats positifs à 64/128/256, combiné indépendant supérieur à 50 %, absence de biais de côté, identité vérifiée et comportement Policy distinct déjà mesuré. Il est donc admis comme `PROBATIONARY_DIVERSITY_GENERATOR`.

Cette décision ne signifie jamais `G3_PROMOTED = YES`.

## Diversité stratégique utile

La diversité sera mesurée selon deux axes séparés :

- **force** : matrice de résultats contre champion, historique et autres générateurs ;
- **comportement** : JS Policy, désaccord d’actions, états visités, recouvrement des trajectoires, ouvertures, motifs terminaux et comportement d’arbre.

`USEFUL_STRATEGIC_DIVERSITY` signifie qu’un modèle ajoute des états ou trajectoires rares et exploitables tout en restant suffisamment compétent et valide. Une divergence causée uniquement par la faiblesse n’est pas utile.

## Matchmaking et cross-play

Le matchmaking recommandé est `CHAMPION_ANCHORED_DIVERSITY_AND_UNCERTAINTY_MATCHMAKING`.

Il doit garantir des quotas minimaux pour :

- self-play du champion ;
- self-play des générateurs ;
- cross-play champion–générateur avec inversion des côtés ;
- cross-play contre des adversaires historiques récents.

Au-delà de ces quotas, les matches sont orientés vers les confrontations incertaines, les provenances sous-représentées et les trajectoires nouvelles. Un éventuel Elo sert au résumé et au matchmaking, jamais comme vérité de promotion.

Le cross-play est essentiel : `G2 vs G3_VALUE_REWORK` peut atteindre des états absents de `G2 vs G2` et `G3 vs G3`. Il expose aussi les non-transitivités éventuelles.

## Contrat de données

Chaque partie et chaque trajectoire doivent conserver au minimum :

```text
generation_id
game_id
p1_model_id
p2_model_id
p1_role
p2_role
p1_fingerprint
p2_fingerprint
mcts_budget
seed
result
generator_role
provenance
state
legal_mask
visit_counts
player_to_move
terminal_z
```

La Policy continue d’apprendre les distributions de visites MCTS. La Value utilise uniquement les vrais résultats terminaux `z`. Les positions Teacher peuvent contribuer à une batterie de couverture ou à une reanalysis sans leurs labels ; Minimax reste un benchmark externe.

## Replay et apprentissage source-aware

La stratégie proposée est `GENERATION_STRATIFIED_SLIDING_WINDOW_WITH_HISTORICAL_ANCHORS` :

- fenêtre glissante des générations récentes ;
- strates distinctes champion, générateur, cross-play et historique ;
- équilibrage par source avant sélection par diversité à l’intérieur de la source ;
- petites ancres historiques immuables ;
- validation et arènes contre plusieurs générations.

Le volume brut d’un générateur ne doit jamais déterminer son poids d’apprentissage. Cette organisation limite l’oubli catastrophique et évite qu’une nouvelle génération meilleure contre G_n devienne plus faible contre G_n−1.

## Sélection Policy/Value

Les Lots 27–28 montrent que Policy et Value doivent être sélectionnées **partiellement séparément**. La future boucle peut maintenir un petit nombre de candidats Policy et Value, assembler un cross-product borné et pré-enregistré, puis évaluer les agents complets sous MCTS.

Cette séparation reste contrôlée : pas de league illimitée de composants. La promotion et le déploiement portent toujours sur un agent complet et reproductible.

## Promotion future

La promotion doit devenir multi-adversaires. Un candidat futur devra démontrer :

- un avantage crédible contre le champion ;
- aucune régression importante contre les historiques récents ;
- un score populationnel acceptable ;
- une réplication indépendante ;
- aucune pathologie de côté ou de recherche.

La monotonie stricte 64→128→256 n’est pas requise. La robustesse de recherche sera jugée par les tailles d’effet, les IC appariés, les réplications et la tendance globale. Un effondrement correspond à une régression matérielle reproduite, pas à une seule fluctuation ponctuelle. Cette règle est prospective et ne modifie pas le Lot 29.

## Pilote recommandé

Le prochain lot doit exécuter `GENERATOR_POOL_PILOT_V1`, sans entraîner de nouveau modèle.

Deux bras de même coût :

- **A — G2-only :** 400 parties G2 vs G2 ;
- **B — Pool :** 100 G2 vs G2, 100 G3 vs G3 et 200 parties cross-play G2 vs G3 avec côtés équilibrés.

Budget proposé : MCTS64. Les seeds, quotas et manifests doivent être figés avant génération. Le volume est suffisant pour comparer couverture et nouveauté sans lancer une nouvelle génération massive.

Le pilote mesure : états uniques, nouveaux états face à D_SCALE_V1, recouvrement, diversité des trajectoires/actions, distribution terminale, longueurs, états propres au cross-play, désaccords stratégiques et pathologies par source. Il n’entraîne ni G3 bis ni G4.

## Risques et mitigations

| Risque | Mitigation |
|---|---|
| dérive de population | quotas d’ancrage champion et holdouts indépendants |
| contamination par modèle faible | admission probatoire, plafonds par source, retrait |
| explosion du coût | pool borné, pas de league complète |
| stratégies cycliques | cross-play et adversaires historiques |
| adversaires obsolètes | retraite selon contribution de nouveauté |
| déséquilibre du dataset | sampler source-balanced |
| oubli catastrophique | ancres historiques et promotion multi-adversaires |

## Verdicts

```text
CURRENT_SINGLE_CHAMPION_PARADIGM_ADEQUATE = PARTIAL
GENERATOR_POOL_RECOMMENDED = YES
POPULATION_TRAINING_RECOMMENDED = YES
CROSS_PLAY_RECOMMENDED = YES
HISTORICAL_OPPONENTS_RECOMMENDED = YES
POLICY_VALUE_SELECTION_SHOULD_BE_PARTIALLY_DECOUPLED = YES
PROMOTION_SHOULD_BE_MULTI_OPPONENT = YES
STRICT_SEARCH_MONOTONICITY_REQUIRED = NO

RECOMMENDED_LEARNING_PARADIGM = CHAMPION_GATED_DIVERSE_GENERATOR_POOL
OFFICIAL_CHAMPION = G2
G3_PROMOTED = NO
G3_VALUE_REWORK_GENERATOR_ADMISSION = YES
NEXT_ACTION = GENERATOR_POOL_PILOT
```

## Réponses finales

1. Le single-champion a atteint ses limites comme source unique : il stabilise G2 mais produit stagnation, distribution étroite et cycles manuels de réparation.
2. Le problème provient à la fois de l’apprentissage du modèle et surtout de l’organisation binaire de l’évolution, qui jette les progrès partiels.
3. Oui, CHAMPION et GENERATOR doivent être séparés.
4. Oui, un candidat non promu peut contribuer au self-play après admission probatoire.
5. Oui, le cross-play est nécessaire pour révéler des états et interactions absents du self-play homogène.
6. Oui, quelques adversaires historiques doivent être conservés pour la non-transitivité et l’oubli.
7. La contamination est évitée par quotas, plafonds de source, probation, métriques de pathologie et retrait.
8. L’oubli est limité par replay historique stratifié et évaluation multi-adversaires.
9. Policy et Value doivent être sélectionnées partiellement séparément, puis recombinées et testées comme agents complets.
10. La promotion future doit mesurer la robustesse face au champion et à la population, avec réplication indépendante et robustesse de recherche statistique.
11. Le paradigme recommandé est `CHAMPION_GATED_DIVERSE_GENERATOR_POOL`.
12. Oui, G3_VALUE_REWORK est admis comme générateur probatoire sans devenir champion.
13. Le prochain test est un pilote égal-budget de 400 parties G2-only contre 400 parties issues du pool mixte.
14. `NEXT_ACTION = GENERATOR_POOL_PILOT`.
