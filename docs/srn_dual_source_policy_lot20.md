# Lot 20 — Correction expérimentale du bottleneck Policy

## 1. Objet et protocole

Le Lot 20 teste si un apprentissage dual-source peut corriger la petite queue d'erreurs Policy à haut regret observée au Lot 17, sans contaminer la tête Value. L'architecture SRN, le moteur, MCTS et G2 restent inchangés.

Deux contrôles partent exactement de G2 :

- **C20-BASE** : entraînement historique sur `D_RL` seulement, avec Policy et vraie Value terminale ;
- **C20-DUAL** : Policy pondérée sur `D_RL`, Policy pondérée sur `D_REANALYSIS_20K`, et Value exclusivement sur `D_RL`.

Le corpus de réanalyse du Lot 19 est strictement Policy-only. Aucun `z`, résultat terminal, label Teacher ou score Minimax n'est créé pour ces positions.

## 2. Provenance et reproductibilité

| Entrée | SHA256 |
|---|---|
| G2 | `eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753` |
| D_RL | `2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57` |
| D_REANALYSIS_20K | `ddc42088c38264fc05dfe6b1faee62a4b7b1fa2613219d71fc6414351928d0cb` |

Les deux candidats ont été initialisés avec une copie exacte des paramètres Policy et Value de G2. Les splits contiennent 35 111 / 7 765 exemples `D_RL` et 15 987 / 4 013 exemples de réanalyse, avec la seed `20262020`.

La configuration commune utilise AdamW, un learning rate de 0,003, un batch nominal de 256, un weight decay de 0,0001 et un clipping de gradient à 1. C20-BASE conserve le protocole historique. C20-DUAL emploie des demi-batches de 128 exemples par source et le même nombre de mises à jour par epoch.

## 3. Pondérations

### D_RL : importance par regret

Le poids brut est `1 + alpha × clip(regret / q95, 0, 1)`, puis borné et normalisé à une moyenne de 1. Une position sans diagnostic conserve un poids neutre avant normalisation. La calibration courte a comparé `alpha ∈ {0, 1, 2}` sur un sous-ensemble tenu à l'écart. `alpha = 1` est retenu : regret moyen 0,07382, P95 0,36593 et 40 erreurs au-dessus de 0,25, contre 0,07633, 0,36998 et 42 pour `alpha = 0`.

La distribution finale des poids `D_RL` a une moyenne de 1, un minimum de 0,9867 et un maximum de 1,9867.

### D_REANALYSIS : confiance de recherche

Le score combine à parts égales la marge des visites et `1 - entropie normalisée`. Les poids sont bornés dans `[0,5 ; 1,5]` et normalisés à une moyenne de 1. Sur les 20 000 positions : minimum 0,7793, médiane 0,8540 et maximum 1,5.

## 4. Contrat de loss

```text
L = L_policy_D_RL_pondérée
  + L_policy_D_REANALYSIS_pondérée
  + L_value_D_RL
```

Les trois coefficients valent 1 dans cette première expérience. Le graphe calculé pour la réanalyse peut produire une sortie Value au forward, mais aucune loss ne la consomme : son gradient Value est nul par construction. Le micro-overfit confirme que les deux losses Policy et la loss Value `D_RL` sont apprenables.

## 5. Résultats hors ligne

### Policy

| Modèle | D_RL CE | D_RL top-1 | D_RE CE | D_RE top-1 |
|---|---:|---:|---:|---:|
| G2 | 1,39338 | 32,97 % | 1,14134 | 51,13 % |
| C20-BASE | 1,39338 | 32,97 % | 1,14134 | 51,13 % |
| C20-DUAL | **1,39143** | **34,15 %** | **1,13931** | **52,53 %** |

C20-BASE s'arrête à l'epoch 0 : le réentraînement historique ne bat pas G2 sur sa validation. C20-DUAL retient l'epoch 1 et apprend les deux sources sans forgetting Policy visible hors ligne.

### Queue de regret indépendante

Sur 370 positions de validation diagnostiquées :

| Modèle | regret moyen | médiane | P95 | P99 | erreurs > 0,25 |
|---|---:|---:|---:|---:|---:|
| G2 / C20-BASE | 0,08652 | 0,02612 | 0,36024 | 0,72009 | 34 |
| C20-DUAL | **0,08025** | **0,01706** | **0,35673** | **0,69004** | **33** |

La queue diminue, mais modestement. C20-DUAL produit 61 améliorations et 52 régressions ; son delta moyen de regret vaut −0,00627. Ce résultat valide le signal diagnostique, pas encore la force de jeu.

### Value

| Modèle | MSE | sign accuracy |
|---|---:|---:|
| G2 / C20-BASE | 0,71624 | 55,25 % |
| C20-DUAL | 0,71723 | 58,93 % |

La hausse de MSE est de 0,14 % et l'exactitude de signe progresse : la Value est considérée préservée.

## 6. Arènes

Chaque duel utilise 128 ouvertures appariées, deux couleurs par ouverture, soit 256 parties par budget. Les intervalles sont obtenus par bootstrap apparié sur les ouvertures.

| Duel | Budget | Score du premier agent | IC bootstrap 95 % | Bilan W-D-L |
|---|---:|---:|---:|---:|
| C20-BASE vs G2 | 64 | 50,00 % | [50,00 ; 50,00] % | 119-18-119 |
| C20-BASE vs G2 | 128 | 50,00 % | [50,00 ; 50,00] % | 123-8-123, 2 tronquées |
| C20-DUAL vs G2 | 64 | 34,58 % | [29,49 ; 39,84] % | 83-9-161, 3 tronquées |
| C20-DUAL vs G2 | 128 | 28,13 % | [23,44 ; 32,81] % | 65-14-177 |
| C20-DUAL vs C20-BASE | 128 | 28,13 % | [23,63 ; 32,81] % | 65-14-177 |

C20-BASE est exactement G2, ce qui explique la symétrie parfaite. C20-DUAL est significativement plus faible aux deux budgets. L'amélioration des cibles hors ligne ne se convertit donc pas en meilleure politique de jeu dans MCTS. Aucun benchmark Minimax n'a été lancé, conformément au gate.

## 7. Réponses aux questions de recherche

1. **Les deux sources sont-elles apprises sans contaminer Value ?** Oui. Les CE et top-1 progressent sur les deux sources, et `D_REANALYSIS` ne contribue jamais à la loss Value.
2. **La queue d'erreurs à haut regret diminue-t-elle ?** Oui, mais faiblement : moyenne, P95, P99 et nombre d'erreurs > 0,25 diminuent.
3. **Les positions diversifiées du Lot 19 sont-elles apprises ?** Oui hors ligne : CE 1,14134 → 1,13931 et top-1 51,13 % → 52,53 %.
4. **Les connaissances Policy et Value de G2 sont-elles préservées ?** Elles le sont selon les validations hors ligne, sans forgetting catastrophique ; elles ne le sont toutefois pas sous la forme d'une force de jeu préservée.
5. **Le modèle joue-t-il significativement mieux que G2 ?** Non. Il joue significativement moins bien aux budgets 64 et 128.
6. **C20-DUAL peut-il devenir candidat G3 ?** Non. G2 reste le générateur officiel.
7. **Quelle est l'unique prochaine étape ?** `POLICY_WEIGHTING_RECALIBRATION`.

L'hypothèse prioritaire est que l'équilibre `lambda_RL = lambda_RE = 1`, combiné à une sélection au seul score hors ligne, déplace trop rapidement la politique malgré de petites améliorations moyennes. Le prochain lot doit recalibrer l'intensité de la source réanalysée et/ou la contrainte envers la Policy parente, sans modifier l'architecture ni multiplier les seeds après ce résultat négatif.

## 8. Verdicts

```text
DUAL_SOURCE_TRAINING_VALID = YES
REGRET_WEIGHTING_VALID = YES
CONFIDENCE_WEIGHTING_VALID = YES
HIGH_REGRET_TAIL_REDUCED = YES
DIVERSE_POLICY_SIGNAL_LEARNED = YES
CATASTROPHIC_POLICY_FORGETTING = NO
VALUE_PRESERVED = YES
C20_BASE_PROGRESS_OVER_G2 = NO
C20_DUAL_PROGRESS_OVER_G2 = NO
C20_DUAL_PROGRESS_OVER_BASE = NO
G3_CANDIDATE = NONE
NEXT_ACTION = POLICY_WEIGHTING_RECALIBRATION
```

## 9. Livrables

- script : [`run_srn_lot20.py`](../apps/trainer/scripts/run_srn_lot20.py) ;
- fonctions de loss : [`dual_source_training.py`](../packages/songo_ai/model/dual_source_training.py) ;
- rapport consolidé : [`report.json`](../data/experiments/lot20_dual_source_policy/report.json) ;
- configuration et provenance : [`configuration.json`](../data/experiments/lot20_dual_source_policy/configuration.json) ;
- évaluations et arènes : [`lot20_dual_source_policy`](../data/experiments/lot20_dual_source_policy) ;
- checkpoints expérimentaux : [`checkpoints`](../data/experiments/lot20_dual_source_policy/checkpoints).
