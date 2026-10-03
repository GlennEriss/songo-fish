# Lot 13 — Confirmation indépendante de G2

## Verdict

**CONFIRMED_G2 = YES**  
**G2_GENERATOR_AUTHORIZED = YES**

La progression de G2-best-validation sur G1-best se reproduit sur une seconde
expérience, avec une nouvelle seed, deux fois plus d'ouvertures qu'au Lot 12 et
une borne basse de l'intervalle de confiance supérieure à 50 % au budget
principal MCTS64. Aucun entraînement, corpus D_RL ou modèle G3 n'a été créé.

## 1–5. Identités et protocole

1. **G1-best.** Lot 6, époque 8, SHA-256
   `09784ce81c7d21d005ccd3fced588c6ec59f65a973ef2a5b71ec27489d10bc36`.
2. **G2-best-validation.** Lot 12, époque 4, SHA-256
   `eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753`.
3. **Nouvelle seed.** `20261313`, distincte de la seed Lot 12 `20261200`.
4. **Ouvertures.** 128 états uniques, produits par préfixes légaux avec rejet
   déterministe des doublons et terminaux. La sélection ne lit aucune sortie de
   G1 ou G2.
5. **Parties.** 256 parties appariées par budget, donc 512 au total.

La recherche utilise `c_puct=1.5`, Dirichlet OFF, température 0 et action
argmax des visites. Le bootstrap emploie 20 000 réplications et conserve
l'ouverture appariée comme unité de rééchantillonnage.

## 6–11. Résultats globaux

| Budget | G2 W/D/L | Terminaux | Score terminal G2 | IC 95 % apparié |
|---:|---:|---:|---:|---:|
| 32 | 184 / 5 / 63 | 252/256 | 74,01 % | [68,36 %, 78,91 %] |
| 64 | 181 / 6 / 68 | 255/256 | 72,16 % | [66,80 %, 77,54 %] |

Les troncatures restent hors du dénominateur terminal et ne sont jamais
converties en nulles. Les résultats appariés complets conservent, pour chaque
ouverture, le score de G2 comme P1, son score comme P2 et la moyenne de la
paire.

## 12–13. Résultats par côté au budget principal MCTS64

| Côté de G2 | W/D/L | Terminaux | Score terminal |
|---|---:|---:|---:|
| P1 | 93 / 4 / 31 | 128 | 74,22 % |
| P2 | 88 / 2 / 37 | 127 | 70,08 % |

La progression n'est donc pas portée par un seul côté physique. Au contrôle
MCTS32, G2 gagne également depuis les deux côtés : 84/4/40 comme P1 et
100/1/23 comme P2 sur 124 terminaux.

## 14–16. Appariement, longueurs et terminaux

14. Les 128 ouvertures possèdent exactement deux parties à chaque budget. Le
    bootstrap est effectué sur les 128 paires et non sur les parties isolées.
15. **MCTS32 :** longueur moyenne 77,48, médiane 64,5, minimum 2, maximum 309.
    **MCTS64 :** moyenne 89,83, médiane 73,5, minimum 2, maximum 400.
16. **MCTS32 :** 252 vrais terminaux, 4 répétitions techniques, 0 max-plies.
    **MCTS64 :** 255 vrais terminaux, 0 répétition et 1 max-plies. Cette unique
    partie à 400 plies représente 0,39 % de la condition et n'est pas une
    anomalie majeure.

À MCTS64, les victoires G2 durent en moyenne 90,76 plies, celles de G1 80,35
et les nulles 117,67.

## 17. Comparaison descriptive avec le Lot 12

| Budget | Score Lot 12 | Score Lot 13 | IC95 Lot 12 | IC95 Lot 13 |
|---:|---:|---:|---:|---:|
| 32 | 78,97 % | 74,01 % | [72,27 %, 85,94 %] | [68,36 %, 78,91 %] |
| 64 | 71,09 % | 72,16 % | [63,28 %, 78,13 %] | [66,80 %, 77,54 %] |

Les scores n'ont pas à être identiques. Le résultat important est que le sens
et l'ampleur générale de l'avantage persistent, avec un intervalle plus étroit
au budget principal grâce aux 128 ouvertures.

## 18–19. Autorisation générationnelle

Les quatre critères MCTS64 passent :

- score G2 supérieur à 50 % ;
- borne basse de l'IC 95 % supérieure à 50 % ;
- score G2 supérieur à 50 % comme P1 et comme P2 ;
- aucune anomalie expérimentale majeure.

En conséquence :

- **CONFIRMED_G2 = YES** ;
- **G2_GENERATOR_AUTHORIZED = YES**.

Cette autorisation désigne précisément le checkpoint G2-best-validation époque
4 identifié ci-dessus. Elle n'autorise aucune substitution par G2-last.

## 20–23. Tests, durée et anomalies

20. Les tests ajoutés couvrent l'identité SHA-256 des checkpoints, le verdict,
    la dépendance aux deux côtés et la génération reproductible de 128
    ouvertures uniques. Les tests historiques couvrent déjà l'appariement et le
    bootstrap par ouverture.
21. La suite complète passe avec **386 tests**.
22. Temps expérimental : 1 160,40 s, dont 383,68 s à MCTS32 et 772,15 s à
    MCTS64.
23. L'unique incident préalable fut un doublon détecté lors de la première
    construction des ouvertures. L'expérience s'est arrêtée avant toute partie.
    L'échantillonneur a été renforcé par rejet déterministe, testé, puis
    l'expérience complète a été relancée avec la même seed. Dans l'expérience
    finale : quatre répétitions MCTS32 et un max-plies MCTS64, tous traités
    comme troncatures techniques.

Les empreintes des deux modèles sont strictement identiques avant et après les
512 parties.

## 24. Recommandation finale

Le prochain lot peut passer directement à la génération G2→G3 demandée :

```text
G2-best-validation epoch 4
    → self-play MCTS64
    → corpus D_RL G2→G3 séparé
    → entraînement G3 initialisé depuis G2-best
    → arène G3 vs G2
```

Conserver les mêmes principes : corpus immutable et non fusionné, split par
partie, gate avant entraînement, aucune promotion automatique, et évaluation
appariée avec une seed encore nouvelle.

## Artefacts

- `data/experiments/lot13_confirm_g2_seed_20261313/report.json`
- `data/experiments/lot13_confirm_g2_seed_20261313/configuration.json`
- `data/experiments/lot13_confirm_g2_seed_20261313/openings.json`
- `data/experiments/lot13_confirm_g2_seed_20261313/games.jsonl`
- `data/experiments/lot13_confirm_g2_seed_20261313/paired_results.json`
- `data/experiments/lot13_confirm_g2_seed_20261313/summaries.json`
