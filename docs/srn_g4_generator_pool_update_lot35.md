# Lot 35 — mise à jour du pool de générateurs G4

## Résultat du gate d’identité

Le Lot 35 s’arrête correctement au premier gate : l’identité exacte du champion G4 n’est pas résolue par les artefacts existants.

CONTROL_G4R et POOL_G4R satisfont tous deux l’intégralité de la règle de promotion du Lot 33. Le rapport du Lot 34R enregistre seulement `OFFICIAL_CHAMPION = G4`, sans relier cet alias à l’un des deux finalistes. La règle de promotion ne contient aucun tie-break. Enfin, la confrontation directe n’est pas concluante : POOL obtient 47,93 % à MCTS64 et 51,18 % à MCTS128, avec des intervalles contenant 50 %.

Choisir CONTROL sur sa performance légèrement supérieure contre G2, ou POOL sur sa préservation et sa performance contre G3, serait une décision post-hoc interdite par le contrat.

## Identités candidates

Les checkpoints, empreintes Policy, Value, architecture, données et protocole sont enregistrés dans [`g4_champion_identity.json`](../data/experiments/lot35_generator_pool/g4_champion_identity.json) et [`model_registry.json`](../data/experiments/lot35_generator_pool/model_registry.json).

- CONTROL_G4R : Policy update 8 000, Value update 12 000.
- POOL_G4R : Policy update 6 000, Value update 12 000.

Ils conservent provisoirement le rôle `PROBATIONARY_GENERATOR`. Aucun des deux ne reçoit artificiellement le rôle `CHAMPION`.

## Procédure pré-enregistrée de résolution

1. Exécuter 1 024 parties appariées et inversées CONTROL–POOL à MCTS128, seed indépendant `20263501`.
2. Sélectionner un modèle seulement si le bootstrap apparié donne au moins 95 % de probabilité que son score dépasse 50 %.
3. Si le résultat reste indécis, répéter 1 024 parties à MCTS256 avec le seed `20263502` et la même règle.
4. Si le résultat reste encore indécis, conserver l’identité non résolue et demander une décision explicite de gouvernance ; ne jamais fabriquer un gagnant à partir de métriques isolées.

## Conséquences

La génération diagnostique du Lot 35 n’a pas été exécutée. Sans champion unique et sans fingerprints officiels, les notions de self-play G4, de G4_ALTERNATE et de contribution marginale par rapport au champion ne sont pas définies sans ambiguïté.

G2 et G3_VALUE_REWORK restent provisoirement `HISTORICAL_OPPONENT`. Aucun modèle n’est retiré et aucun pool actif n’est figé.

Les seuils du futur diagnostic sont déjà pré-enregistrés, mais ne seront appliqués qu’après résolution de l’identité : contribution d’états minimale 2 %, désaccord stratégique minimal 5 %, redondance élevée à partir de 95 %, coût relatif maximal 1,5 et maximum de trois générateurs actifs.

## Contrat G5 conservé

```text
G5_INITIAL_POLICY = G4_CHAMPION_POLICY (fingerprint à résoudre)
G5_INITIAL_VALUE = G4_CHAMPION_VALUE (fingerprint à résoudre)
G5_POLICY_OBJECTIVE = CORRECT_AND_PRESERVE
G5_VALUE_OBJECTIVE = MSE_TO_TRUE_TERMINAL_Z_ONLY
POLICY_VALUE_SELECTION_DECOUPLED = YES
G5_TRAINING_PERFORMED = NO
G5_MASSIVE_DATA_GENERATION_PERFORMED = NO
```

## Verdicts

```text
G4_CHAMPION_IDENTITY_RESOLVED = NO
G4_CHAMPION_SOURCE = UNRESOLVED
G2_ACTIVE_GENERATOR = INCONCLUSIVE
G2_HISTORICAL_OPPONENT = YES
G3_ACTIVE_GENERATOR = INCONCLUSIVE
G3_HISTORICAL_OPPONENT = YES
G3_PROMOTED = NO
G4_ALTERNATE_ACTIVE_GENERATOR = INCONCLUSIVE
ACTIVE_GENERATOR_POOL = []
HISTORICAL_OPPONENT_POOL = [G2, G3_VALUE_REWORK]
RETIRED_MODELS = []
MAX_ACTIVE_GENERATORS = 3
OFFICIAL_CHAMPION = G4_ALIAS_IDENTITY_UNRESOLVED
NEXT_ACTION = G4_CHAMPION_IDENTITY_RESOLUTION
```

Cette interruption préserve la validité scientifique du registre et empêche le futur G5 de dépendre d’un alias de champion non reproductible.
