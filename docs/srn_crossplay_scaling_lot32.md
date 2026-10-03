# Lot 32 — Expérience de montée en échelle du cross-play

## 1. Question expérimentale

Le Lot 32 teste si l'augmentation de la part de cross-play G2↔G3 produit, à budget comparable, une diversité autonome matériellement supérieure à G2 seul. Il prolonge le pilote du Lot 31 avec un facteur cinq : 2 000 parties par bras, soit 6 000 parties au total.

Le lot est exclusivement consacré à la génération et à la mesure. Aucun entraînement, fine-tuning, nouveau checkpoint, label Teacher/Minimax ou changement du SRN, de MCTS ou du moteur n'a été réalisé. G2 demeure champion officiel ; G3_VALUE_REWORK reste composé de la Policy G3_STRATEGIC et de la Value V28_A.

## 2. Protocole préenregistré

Trois compositions ont été fixées avant génération :

| Bras | G2–G2 | G3–G3 | Cross-play équilibré | Total |
|---|---:|---:|---:|---:|
| CONTROL | 2 000 | 0 | 0 | 2 000 |
| ORIGINAL_POOL | 500 | 500 | 1 000 (500/500 orientations) | 2 000 |
| CROSSPLAY_ENRICHED | 200 | 200 | 1 600 (800/800 orientations) | 2 000 |

Tous les bras utilisent 64 simulations par décision, `c_puct=1.5`, le même bruit Dirichlet, la même température, une limite de 400 plis et la même règle de répétition. `LOT32_SEED_SET_V1` est indépendant du Lot 31. Les checkpoints, leurs SHA-256 et les empreintes Policy/Value/architecture ont été enregistrés avant génération.

Les seuils du Lot 31 ont été conservés : couverture ou nouveauté historique normalisée ≥ 2 %, diversité stratégique relative ≥ 5 %, coût par simulation ≤ 2× CONTROL. Le seuil de forte divergence JS reste 0,001. Aucun seuil n'a été déplacé après observation.

## 3. Intégrité de l'expérience

Les 6 000 parties prévues ont été produites. CONTROL et ORIGINAL_POOL ont 2 000 parties terminales chacun. CROSSPLAY_ENRICHED a 1 999 parties terminales et une troncature, soit 0,05 %, sans faux label Value.

Le contrôle exhaustif des 539 616 positions ne trouve aucun coup illégal, aucune transition incohérente, aucune violation de conservation des 70 graines, aucun masque invalide, aucun identifiant de partie dupliqué et aucun défaut de provenance. Les trois configurations de recherche sont identiques et les checkpoints sont restés immuables.

## 4. Couverture des états

| Mesure | CONTROL | ORIGINAL_POOL | CROSSPLAY_ENRICHED |
|---|---:|---:|---:|
| Positions brutes | 176 686 | 180 992 | 181 938 |
| États uniques | 167 656 | 171 899 | 172 983 |
| Duplication | 5,111 % | 5,024 % | 4,922 % |
| Uniques / 10 000 positions | 9 488,92 | 9 497,60 | 9 507,80 |

Face à CONTROL, le gain normalisé vaut **+0,091 %** pour ORIGINAL_POOL et **+0,199 %** pour CROSSPLAY_ENRICHED. Les deux résultats restent très inférieurs au seuil matériel de 2 %. Le résultat négatif de couverture du Lot 31 se reproduit donc à plus grande échelle.

Le faible chevauchement entre bras persiste : CONTROL et CROSSPLAY_ENRICHED ne partagent que 2 132 états physiques. Ce constat confirme qu'une grande quantité d'états exclusifs peut provenir naturellement de trajectoires stochastiques longues ; elle ne constitue pas, seule, un gain matériel.

## 5. Nouveauté historique

Le référentiel historique a été figé à 634 192 états pour les trois bras.

| Mesure | CONTROL | ORIGINAL_POOL | CROSSPLAY_ENRICHED |
|---|---:|---:|---:|
| Nouveaux vs historique | 163 081 | 167 526 | 168 534 |
| Nouveaux / 10 000 positions | 9 229,99 | 9 255,99 | 9 263,27 |
| Gain relatif vs CONTROL | — | +0,282 % | +0,361 % |

La nouveauté historique augmente de façon statistiquement cohérente pour CROSSPLAY_ENRICHED dans le bootstrap par parties, mais son amplitude reste loin des 2 % requis. L'IC95 du proxy bootstrap pour la différence relative CONTROL→CROSSPLAY_ENRICHED est `[+0,031 %, +0,624 %]`. Il s'agit d'un petit effet, pas d'un gain matériel.

## 6. Diversité stratégique

L'échantillon déterministe contient 50 000 états uniques par bras. Le désaccord top-1 entre les Policy G2 et G3 vaut :

- CONTROL : 9,654 % ;
- ORIGINAL_POOL : 9,584 %, soit −0,73 % relatif ;
- CROSSPLAY_ENRICHED : 10,430 %, soit **+8,04 % relatif**.

CROSSPLAY_ENRICHED franchit donc le seuil stratégique préenregistré de 5 %. Le signal ne vient pas d'une hausse générale de l'entropie des actions : celle-ci demeure presque identique entre les bras. Le désaccord apparaît davantage dans les profondeurs intermédiaires et tardives, notamment les bins 6 à 9, plutôt que seulement à l'ouverture.

La JS moyenne reste très faible et le taux de JS ≥ 0,001 n'augmente pas. Le résultat doit donc être interprété précisément : le pool enrichi visite davantage d'états où le premier choix des deux Policy diffère, sans produire une séparation massive de leurs distributions complètes. Aucun Minimax n'est utilisé pour décider laquelle a raison.

## 7. Cross-play-only et saturation

ORIGINAL_POOL produit 85 253 états exclusivement issus du cross-play, soit 9 440,77 par 10 000 positions cross-play. CROSSPLAY_ENRICHED en produit 136 663, soit 9 473,64 par 10 000. Le taux reste donc élevé à l'échelle ×5.

Dans CROSSPLAY_ENRICHED, les tranches successives de 200 parties ajoutent entre 16 494 et 18 489 nouveaux états. La dernière tranche conserve 91,5 % de l'apport de la première, très au-dessus du critère de saturation rapide fixé à 50 %. La nouveauté cross-play persiste donc et ne s'effondre pas rapidement.

Cette persistance explique pourquoi les 16 955 états cross-play-only du Lot 31 annonçaient bien un phénomène durable. Elle ne suffisait toutefois pas, à elle seule, à prédire le gain stratégique observé ici.

## 8. Valeur marginale des sources

Dans CROSSPLAY_ENRICHED :

| Source | Positions | Uniques | Propres aux autres sources | Nouveaux historiques / 10k |
|---|---:|---:|---:|---:|
| G2–G2 | 18 072 | 17 461 | 17 058 | 9 302,79 |
| G3–G3 | 19 610 | 19 007 | 18 553 | 9 303,42 |
| Cross-play | 144 256 | 137 358 | 136 663 | 9 261,45 |

G3–G3 apporte une contribution propre non triviale malgré son volume réduit. Le cross-play fournit de loin le plus grand apport marginal absolu, mais son rendement historique par position n'est pas supérieur aux sources homogènes. Sa valeur spécifique réside surtout dans la composition stratégique différente révélée par le désaccord Policy.

## 9. Visitation structurelle et trajectoires

Les distances de variation totale face à CONTROL restent faibles. Pour CROSSPLAY_ENRICHED, elles vont de 1,13 % sur le nombre de coups légaux à 2,29 % sur la profondeur, sous le seuil descriptif préenregistré de 5 %. Le cross-play ne déplace donc pas massivement les distributions brutes de graines en jeu, magasins, légalité ou profondeur.

Chaque bras produit 2 000 trajectoires exactes distinctes. À quatre demi-coups, les nombres de préfixes uniques sont 1 252, 1 244 et 1 246 ; à douze demi-coups, ils atteignent 2 000 dans tous les bras. La divergence ne se limite donc pas à multiplier artificiellement les ouvertures.

Les longueurs moyennes sont 88,34, 90,50 et 90,97 plis. Les médianes sont respectivement 72,5, 73 et 72. Le gain stratégique ne s'explique donc pas simplement par des parties beaucoup plus longues.

## 10. Coût computationnel

| Bras | Simulations | Simulations/partie | Temps | Uniques/million simulations | Ratio temps/simulation |
|---|---:|---:|---:|---:|---:|
| CONTROL | 11 307 904 | 5 653,95 | 4 222,67 s | 14 826,44 | 1,000× |
| ORIGINAL_POOL | 11 583 488 | 5 791,74 | 5 884,53 s | 14 840,00 | 1,360× |
| CROSSPLAY_ENRICHED | 11 644 032 | 5 822,02 | 6 941,68 s | 14 855,94 | 1,596× |

Le surcoût du modèle hybride est réel mais reste sous la limite de 2×. Le nombre total de simulations varie uniquement avec la longueur des trajectoires ; chaque décision conserve exactement le même budget.

## 11. Bootstrap et replay virtuel

Les incertitudes ont été estimées par 10 000 réplications au niveau partie, jamais au niveau position. Le proxy de couverture intra-partie ne montre aucun avantage robuste de couverture pour les pools : les IC95 des différences relatives incluent zéro. Le petit gain de nouveauté historique de CROSSPLAY_ENRICHED est positif, mais reste sous le seuil matériel.

Les replays virtuels de 1 200 exemples préservent la provenance. Pour les deux pools, ils produisent exactement 400 exemples G2–G2, 400 G3–G3 et 400 cross-play. Toutes les cibles Value respectent la règle terminale. Aucun gradient n'a été calculé.

## 12. Décision

ORIGINAL_POOL reproduit à grande échelle le résultat neutre du Lot 31. CROSSPLAY_ENRICHED ne gagne pas matériellement en couverture ni en nouveauté historique, mais dépasse le seuil stratégique avec +8,04 % de désaccord top-1, tout en conservant une contribution cross-play clairement identifiable, non pathologique, persistante et à coût acceptable.

Conformément au critère qui exige un gain matériel sur au moins une dimension majeure — et non sur toutes — G3_VALUE_REWORK est retenu comme générateur. Cette décision ne constitue pas une promotion de joueur. La composition recommandée pour la prochaine phase de conception est la famille **CROSSPLAY_ENRICHED**, sans prétendre que le ratio 80 % est optimal.

## 13. Verdicts obligatoires

```text
EXPERIMENT_VALID = YES
EQUAL_SEARCH_CONFIGURATION = YES
ORIGINAL_POOL_SCALE_EFFECT = NEUTRAL
CROSSPLAY_ENRICHED_EFFECT = POSITIVE
CROSSPLAY_WEIGHT_EFFECT = POSITIVE
CROSSPLAY_NOVELTY_PERSISTS_AT_SCALE = YES
CROSSPLAY_NOVELTY_SATURATES_RAPIDLY = NO
MATERIAL_COVERAGE_GAIN = NO
MATERIAL_HISTORICAL_NOVELTY_GAIN = NO
MATERIAL_STRATEGIC_DIVERSITY_GAIN = YES
CROSSPLAY_SHIFTS_STATE_VISITATION = NO
G3_SELFPLAY_ADDS_MARGINAL_VALUE = YES
CROSSPLAY_ADDS_MARGINAL_VALUE = YES
DATA_PATHOLOGY = NO
COMPUTE_COST_ACCEPTABLE = YES
REPLAY_STRATIFICATION_VALID = YES
G3_VALUE_REWORK_GENERATOR_STATUS = RETAIN
RECOMMENDED_GENERATION_MODE = CROSSPLAY_ENRICHED
OFFICIAL_CHAMPION = G2
G3_PROMOTED = NO
NEXT_ACTION = POOL_GENERATED_G4_TRAINING_DESIGN
```

## 14. Réponses aux dix-sept questions finales

1. Chaque bras a produit 2 000 parties. CONTROL contient 176 686 positions, ORIGINAL_POOL 180 992 et CROSSPLAY_ENRICHED 181 938.
2. Les couvertures normalisées sont 9 488,92, 9 497,60 et 9 507,80 états uniques par 10 000 positions.
3. Oui. Le résultat négatif de couverture du Lot 31 se reproduit : aucun pool n'atteint +2 %.
4. Non. Le meilleur gain de nouveauté historique vaut seulement +0,361 %.
5. Oui pour CROSSPLAY_ENRICHED : +8,04 % de désaccord top-1 relatif. ORIGINAL_POOL reste neutre.
6. Oui. Le phénomène cross-play-only persiste avec 136 663 états propres dans le bras enrichi.
7. La nouveauté continue de croître ; la dernière tranche conserve 91,5 % de l'apport initial et ne sature pas rapidement.
8. Pas massivement selon les variables structurelles brutes : toutes les distances de visitation restent sous 5 %.
9. Oui. G3–G3 apporte 18 553 états propres dans CROSSPLAY_ENRICHED.
10. Oui. G2–G3 apporte 136 663 états propres et le signal stratégique matériel du bras enrichi.
11. CROSSPLAY_ENRICHED est le plus intéressant ; ORIGINAL_POOL reproduit un effet neutre.
12. Le coût relatif par simulation vaut 1,360× pour ORIGINAL_POOL et 1,596× pour CROSSPLAY_ENRICHED.
13. Non. Une seule troncature sur 6 000 parties et aucune anomalie de règles ou de données.
14. `G3_VALUE_REWORK_GENERATOR_STATUS = RETAIN`.
15. `RECOMMENDED_GENERATION_MODE = CROSSPLAY_ENRICHED`.
16. `OFFICIAL_CHAMPION = G2` et `G3_PROMOTED = NO`.
17. `NEXT_ACTION = POOL_GENERATED_G4_TRAINING_DESIGN`.

## 15. Portée de la recommandation

Le prochain lot doit concevoir — et non lancer implicitement — l'entraînement G4 alimenté par un replay cross-play enrichi. Il devra notamment figer le mélange avec l'historique, les règles de stratification, les critères de promotion et les ablations permettant d'attribuer un éventuel progrès au nouveau corpus. Le Lot 32 n'autorise encore aucun entraînement.
