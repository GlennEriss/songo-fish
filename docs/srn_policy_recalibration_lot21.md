# Lot 21 — Policy recalibration et régularisation parentale

## 1. Question et protocole

Le Lot 21 cherche une zone de mise à jour sûre autour de G2 : apprendre une partie du nouveau signal MCTS128 du Lot 19 sans détruire la structure Policy qui permet à G2 de guider efficacement MCTS.

Les quatre candidats partent exactement de G2, utilisent la même seed, les mêmes splits et les mêmes pondérations regret/confiance que le Lot 20 :

| Candidat | `lambda_RE` | `beta` parent |
|---|---:|---:|
| C21-A | 0,10 | 0 |
| C21-B | 0,25 | 0 |
| C21-C | 0,25 | 0,10 |
| C21-D | 0,25 | 0,25 |

La loss est :

```text
L = L_policy_RL
  + lambda_RE * L_policy_REANALYSIS
  + L_value_RL
  + beta/2 * (KL_RL + KL_REANALYSIS)
```

avec `KL = KL(P_G2 || P_candidate)`, calculé uniquement sur les actions légales. Les logits G2 sont détachés : G2 ne reçoit aucun gradient. La source réanalysée reste Policy-only et ne produit aucun label ni gradient Value.

## 2. Données et reproductibilité

| Entrée | SHA256 |
|---|---|
| G2 | `eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753` |
| D_RL | `2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57` |
| D_REANALYSIS_20K | `ddc42088c38264fc05dfe6b1faee62a4b7b1fa2613219d71fc6414351928d0cb` |

Les splits du Lot 20 sont conservés : 35 111 / 7 765 exemples `D_RL` et 15 987 / 4 013 exemples de réanalyse. La seed d'optimisation est `20262020`. L'égalité exacte des sorties Policy et Value avec G2 a été vérifiée à l'epoch 0 pour les quatre candidats.

## 3. Calibration de la régularisation parentale

Une sonde de 20 mises à jour avec `lambda_RE=0,25` a mesuré le rapport entre le gradient parental et les gradients Policy RL+RE :

| beta | contribution loss parent | norme gradient parent | ratio parent / Policy |
|---:|---:|---:|---:|
| 0,05 | 0,000053 | 0,001150 | 2,44 % |
| 0,10 | 0,000106 | 0,002300 | 4,88 % |
| 0,25 | 0,000266 | 0,005750 | 12,20 % |

Les niveaux faible et moyen sont donc fixés avant les arènes à 0,10 et 0,25. Cette calibration montre aussi que la loss Value possède une norme de gradient nettement supérieure aux termes Policy sur le batch sondé ; elle n'a toutefois pas été modifiée afin de conserver le protocole historique demandé.

## 4. Sélection des checkpoints

Le score fixé avant entraînement combine les CE RL et réanalyse normalisées par leur valeur epoch 0, la MSE Value normalisée et les deux divergences parentales. L'epoch 0 est un candidat réel : un entraînement qui n'améliore pas ce score conserve G2 au lieu de retenir artificiellement un epoch dégradé.

| Candidat | meilleur epoch | interprétation |
|---|---:|---|
| C21-A | 0 | aucune mise à jour sûre retenue |
| C21-B | 4 | apprentissage non nul retenu |
| C21-C | 0 | l'ancrage faible ne produit pas de compromis validé |
| C21-D | 0 | l'ancrage moyen ne produit pas de compromis validé |

C21-A, C21-C et C21-D sont donc identiques à G2 dans leurs checkpoints finaux. Ils sont marqués `OVER_REGULARIZED=YES` au sens opérationnel du lot : aucun nouveau signal n'est conservé. Pour C21-A, cette absence vient du faible signal d'apprentissage et de la règle de sélection, non d'un terme parent explicite.

## 5. Résultats hors ligne

| Modèle | CE D_RL | top-1 D_RL | CE D_RE | top-1 D_RE | MSE Value |
|---|---:|---:|---:|---:|---:|
| G2 | 1,39338 | 32,97 % | 1,14134 | 51,13 % | 0,71624 |
| C21-A | 1,39338 | 32,97 % | 1,14134 | 51,13 % | 0,71624 |
| C21-B | **1,39124** | **34,26 %** | **1,13898** | 50,93 % | **0,71488** |
| C21-C | 1,39338 | 32,97 % | 1,14134 | 51,13 % | 0,71624 |
| C21-D | 1,39338 | 32,97 % | 1,14134 | 51,13 % | 0,71624 |

C21-B apprend la distribution de réanalyse au sens de la CE et augmente en moyenne de 0,00336 la probabilité de l'action préférée par MCTS128. La baisse de top-1 D_RE rappelle que CE et classement discret ne mesurent pas la même chose.

## 6. Dérive Policy

Pour C21-B :

| Batterie | JS moyenne | KL moyenne `G2 || C21-B` | désaccord argmax | désaccord top-2 |
|---|---:|---:|---:|---:|
| D_RL validation | 0,000738 | 0,002977 | 26,93 % | 33,10 % |
| D_RE validation | 0,000617 | 0,002482 | 28,98 % | 31,65 % |
| Lot 17 | 0,000781 | 0,003155 | 34,59 % | 33,24 % |

La divergence moyenne est numériquement petite, mais le classement de l'action dominante change sur 27 à 35 % des positions. Cette observation explique pourquoi une faible JS moyenne ne constitue pas à elle seule une zone sûre. C20-DUAL avait même une JS D_RE moyenne plus faible, 0,000462, tout en étant très faible en arène : la magnitude moyenne de la dérive est insuffisante pour prédire la qualité du guidage MCTS.

## 7. Queue de regret

| Modèle | moyenne | médiane | P95 | P99 | erreurs > 0,25 | coût total des pires 5 % |
|---|---:|---:|---:|---:|---:|---:|
| G2 | 0,08652 | 0,02612 | 0,36024 | 0,72009 | 34 | 10,9402 |
| C21-B | **0,07745** | **0,01695** | **0,34791** | **0,69004** | **30** | **10,4989** |

`HIGH_REGRET_TAIL_VS_G2=IMPROVED` pour C21-B. Les trois autres candidats sont identiques à G2 et obtiennent `SIMILAR`.

## 8. Gate et arène courte

Seul C21-B passe le gate hors ligne : apprentissage D_RE non nul, Value préservée, divergence finie et queue de regret améliorée.

L'arène courte MCTS128 utilise 32 ouvertures appariées, soit 64 parties :

| Duel | W-D-L | score | IC95 apparié | P1 | P2 |
|---|---:|---:|---:|---:|---:|
| C21-B vs G2 | 22-1-41 | 35,16 % | [25,00 ; 46,09] % | 12-1-19 | 10-0-22 |

Le score est catastrophique pour un screening. C21-B n'est donc pas finaliste et aucune grande arène n'est lancée. Conformément au protocole, cette arène courte ne sert pas à prouver une différence fine ; elle suffit ici à rejeter une configuration manifestement faible. `SEARCH_SCALING_HEALTHY` reste `INCONCLUSIVE`, puisqu'aucun candidat n'a franchi le screening pour la comparaison 64/128.

## 9. Courbe de mise à jour sûre

| Point | apprentissage D_RE | JS D_RE | queue de regret | force MCTS128 |
|---|---|---:|---|---:|
| G2 / epochs 0 | aucun nouveau signal | 0 | référence | 50 % par définition |
| C21-B | oui | 0,000617 | améliorée | 35,16 % |
| C20-DUAL | oui | 0,000462 | légèrement améliorée | 28,13 % |

Le lot n'identifie aucune zone **non nulle** où apprentissage du nouveau signal et recherche saine coexistent. La seule zone empiriquement sûre observée reste la dérive nulle, c'est-à-dire G2 lui-même ; ce n'est pas une solution d'apprentissage.

## 10. Réponses finales

1. **Réduire `lambda_RE` permet-il de récupérer la force de G2 ?** Non. `lambda_RE=0,10` ne conserve aucune mise à jour et `lambda_RE=0,25` produit seulement 35,16 % contre G2.
2. **L'ancrage vers G2 apporte-t-il quelque chose de plus ?** Non dans les niveaux calibrés. Les checkpoints sélectionnés de C21-C/D restent à l'epoch 0 et n'apprennent rien de nouveau.
3. **Quelle dérive semble compatible avec une recherche saine ?** Seule la dérive nulle est validée. Une JS moyenne de 0,000617, accompagnée d'environ 29 % de changements d'argmax sur D_RE, est déjà incompatible avec le screening MCTS128.
4. **Les nouvelles connaissances MCTS128 sont-elles apprises dans cette zone ?** Elles sont apprises par C21-B, mais hors de toute zone de jeu sûre identifiée. Elles ne le sont pas par les checkpoints restés sur G2.
5. **La queue à haut regret reste-t-elle maîtrisée ?** Oui ; C21-B l'améliore selon toutes les mesures principales.
6. **Un candidat dépasse-t-il statistiquement G2 ?** Non. Aucun candidat n'atteint même l'arène principale.
7. **Faut-il confirmer un G3 ou abandonner la simple recalibration ?** Aucun G3 ne doit être confirmé. La simple recalibration des coefficients doit être abandonnée au profit d'une révision de l'objectif Policy.

### Réponse scientifique

**NO.** Dans le plan factoriel testé, aucune zone de mise à jour Policy non nulle n'apprend le signal MCTS128 tout en conservant une recherche MCTS efficace.

## 11. Verdicts

```text
POLICY_RECALIBRATION_VALID = YES
LOWER_REANALYSIS_WEIGHT_HELPS = NO
PARENT_REGULARIZATION_HELPS = NO
SAFE_POLICY_UPDATE_REGION_FOUND = NO
HIGH_REGRET_TAIL_REDUCED = YES
VALUE_PRESERVED = YES
SEARCH_SCALING_HEALTHY = INCONCLUSIVE
SIMPLE_POLICY_RECALIBRATION_INSUFFICIENT = YES
G3_CANDIDATE = NONE
NEXT_ACTION = REVISIT_POLICY_OBJECTIVE
```

G2 reste le générateur officiel. Aucun benchmark Minimax, nouveau self-play, G4 ou seed supplémentaire n'a été lancé.

## 12. Livrables

- script : [`run_srn_lot21.py`](../apps/trainer/scripts/run_srn_lot21.py) ;
- rapport : [`report.json`](../data/experiments/lot21_policy_recalibration/report.json) ;
- comparaison : [`candidate_comparison.json`](../data/experiments/lot21_policy_recalibration/candidate_comparison.json) ;
- calibration : [`beta_calibration.json`](../data/experiments/lot21_policy_recalibration/beta_calibration.json) ;
- dérive : [`policy_drift.json`](../data/experiments/lot21_policy_recalibration/policy_drift.json) ;
- regret : [`regret_tail.json`](../data/experiments/lot21_policy_recalibration/regret_tail.json) ;
- gradients : [`gradient_contributions.json`](../data/experiments/lot21_policy_recalibration/gradient_contributions.json) ;
- arène courte : [`short_arena.json`](../data/experiments/lot21_policy_recalibration/short_arena.json) ;
- arène principale vide après gate : [`main_arena.json`](../data/experiments/lot21_policy_recalibration/main_arena.json).
