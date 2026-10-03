# Lot 34R — retry contrôlé avec préservation stratégique

## Résultat

Le Lot 34R est valide et complet. Il réutilise exactement les 16 000 parties du Lot 34, sans nouvelle génération, sans modification du moteur, de MCTS ou de l’architecture SRN.

L’analyse forensic montre que, dans CONTROL comme dans POOL, update 0 était le dernier point à au moins 85 % de préservation et que le premier checkpoint sauvegardé, à 4 000 updates, était déjà sous le seuil. Aucun checkpoint non trivial du Lot 34 n’était récupérable. Le diagnostic principal est donc `POLICY_OBJECTIVE_DRIFT`.

## Intervention

Une seule intervention commune a été pré-enregistrée avant le retry : `CORRECT_AND_PRESERVE`, reprise du Lot 24 avec λ correction = 0,1, λ préservation = 1,0 et ρ = 0,5. Il n’y a eu aucun sweep. La Value reste entraînée exclusivement par MSE sur les vrais résultats terminaux ; les parties tronquées ne fournissent aucun label Value.

La règle commune arrête un bras après deux contrôles consécutifs sous 85 %. Les deux entraînements se sont arrêtés à 12 000 updates.

| Update | CONTROL préservation | CONTROL correction | POOL préservation | POOL correction |
|---:|---:|---:|---:|---:|
| 2 000 | 94,26 % | 6,48 % | 92,70 % | 6,83 % |
| 4 000 | 91,21 % | 9,19 % | 91,48 % | 9,48 % |
| 6 000 | 88,69 % | 11,83 % | 88,47 % | 12,82 % |
| 8 000 | 85,98 % | 13,39 % | 86,80 % | 14,02 % |
| 10 000 | 84,56 % | 15,36 % | 84,02 % | 17,04 % |
| 12 000 | 80,58 % | 18,69 % | 83,12 % | 20,46 % |

Les deux bras ont donc réellement appris tout en conservant une région sûre non triviale.

## Finalistes

- CONTROL_G4R : Policy update 8 000, Value update 12 000 ; préservation 85,98 %.
- POOL_G4R : Policy update 6 000, Value update 12 000 ; préservation 88,47 %.

Le short gate donne 91,41 % pour CONTROL et 92,19 % pour POOL contre G2. Ces résultats servent uniquement au choix des combinaisons Policy×Value, pas à la promotion.

## Arènes principales

| Comparaison — score du premier modèle | MCTS64 | MCTS128 |
|---|---:|---:|
| POOL vs CONTROL | 47,93 % | 51,18 % |
| CONTROL vs G2 | 88,45 % | 86,59 % |
| POOL vs G2 | 87,50 % | 86,82 % |
| CONTROL vs G3_VALUE_REWORK | 86,37 % | 86,99 % |
| POOL vs G3_VALUE_REWORK | 88,85 % | 87,87 % |

Chaque cellule repose sur 512 parties appariées et inversées. Les deux G4R dominent nettement G2 et G3 et restent stables lorsque le budget passe de 64 à 128. En revanche, POOL ne domine pas CONTROL : la moyenne directe vaut 49,56 %, avec des intervalles contenant 50 % aux deux budgets.

## Interprétation

Le retry démontre que l’amélioration provient de la correction de la dynamique d’apprentissage Policy et de la sélection stratégique des checkpoints. Il ne démontre pas que le corpus cross-play 10/10/80 est supérieur au corpus G2-only. `POOL_DATA_PARADIGM_EFFECT = NEUTRAL`.

Les deux candidats satisfont séparément la règle de promotion du Lot 33 : progression contre G2, robustesse populationnelle, équilibre des côtés, robustesse multi-budget et préservation stratégique. G4 est donc promu, tandis que G3 n’est pas promu rétroactivement.

## Verdicts

```text
LOT34_DATA_REUSED_EXACTLY = YES
NEW_GENERATION_PERFORMED = NO
CHECKPOINT_FORENSICS_VALID = YES
SAFE_CHECKPOINT_EXISTS_CONTROL = NO
SAFE_CHECKPOINT_EXISTS_POOL = NO
PRIMARY_LOT34_FAILURE_CAUSE = POLICY_OBJECTIVE_DRIFT
RETRY_INTERVENTION_REQUIRED = YES
RETRY_INTERVENTION = CORRECT_AND_PRESERVE
CONTROL_STRATEGIC_GATE = PASS
POOL_STRATEGIC_GATE = PASS
NONTRIVIAL_POLICY_LEARNING_CONTROL = YES
NONTRIVIAL_POLICY_LEARNING_POOL = YES
MAIN_ARENAS_ALLOWED = YES
POOL_G4R_BEATS_CONTROL_G4R = NO
POOL_DATA_PARADIGM_EFFECT = NEUTRAL
POOL_G4R_ROBUST_VS_G2 = YES
POOL_G4R_ROBUST_VS_G3 = YES
SEARCH_ROBUSTNESS = HEALTHY
POPULATION_ROBUSTNESS = YES
G4_CHAMPION_PROMOTION = YES
G4_GENERATOR_ADMISSION = YES
OFFICIAL_CHAMPION = G4
G3_PROMOTED = NO
NEXT_ACTION = G4_GENERATOR_POOL_UPDATE
```

Le rapport machine-readable complet est disponible dans [`data/experiments/lot34r_g4_retry/report.json`](../data/experiments/lot34r_g4_retry/report.json).
