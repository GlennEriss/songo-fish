# Lot 35R — résolution de l’identité du champion G4

## Conclusion

L’identité unique du champion G4 n’a pas pu être résolue. CONTROL_G4R et POOL_G4R restent statistiquement indiscernables après les deux phases pré-enregistrées, soit 2 048 nouvelles parties appariées. Aucun match supplémentaire et aucun tie-break post-hoc ne sont autorisés.

Les fingerprints Policy, Value et architecture des deux candidats ont été figés avant les arènes. Aucun modèle, moteur, composant SRN ou paramètre MCTS n’a été modifié.

## Phase 1 — MCTS128

La phase 1 comprend 1 024 parties, soit 512 paires avec inversion des côtés et seeds indépendants du Lot 34R.

- score POOL : 48,62 % ;
- score CONTROL : 51,38 % ;
- IC95 apparié du score POOL : 46,04–51,17 % ;
- `P(POOL > CONTROL) = 13,48 %` ;
- `P(CONTROL > POOL) = 85,73 %`.

Le seuil de 95 % n’est atteint dans aucune direction. La phase 1 est donc `INCONCLUSIVE` et déclenche obligatoirement la phase 2.

## Phase 2 — MCTS256

La phase 2 utilise 1 024 nouvelles parties, 512 paires inversées et un second manifeste de seeds indépendant.

- score POOL : 50,69 % ;
- score CONTROL : 49,31 % ;
- IC95 apparié du score POOL : 47,95–53,18 % ;
- `P(POOL > CONTROL) = 65,18 %` ;
- `P(CONTROL > POOL) = 33,46 %`.

Le seuil de 95 % n’est toujours atteint dans aucune direction. La variation de direction entre MCTS128 et MCTS256 renforce le constat d’équivalence pratique plutôt qu’une hiérarchie fiable.

## Registre

Le registre n’est pas modifié vers un faux champion unique :

```text
G4_CHAMPION_IDENTITY_RESOLVED = NO
G4_CHAMPION_SOURCE = UNRESOLVED
G4_ALTERNATE_SOURCE = UNRESOLVED
OFFICIAL_CHAMPION = G4_ALIAS_IDENTITY_UNRESOLVED
```

CONTROL_G4R et POOL_G4R conservent leurs identités et fingerprints exacts. Aucun pruning n’a été effectué. Le pool de générateurs et la conception exécutable de G5 restent bloqués jusqu’à la définition d’un registre capable de représenter deux co-champions G4.

## Garanties

```text
PHASE1_VALID = YES
PHASE1_GAMES = 1024
PHASE1_BUDGET = MCTS128
PHASE1_DECISION = INCONCLUSIVE

PHASE2_EXECUTED = YES
PHASE2_VALID = YES
PHASE2_GAMES = 1024
PHASE2_BUDGET = MCTS256
PHASE2_DECISION = INCONCLUSIVE

G5_TRAINING_PERFORMED = NO
G5_MASSIVE_DATA_GENERATION_PERFORMED = NO
NEXT_ACTION = G4_COCHAMPION_REGISTRY_DESIGN
```

Le rapport complet est disponible dans [`data/experiments/lot35r_champion_identity/report.json`](../data/experiments/lot35r_champion_identity/report.json).
