# Lot 16A — Localisation du blocage G2→G3

## Verdict exécutif

- **POLICY_BOTTLENECK = YES**
- **VALUE_BOTTLENECK = NO**
- **POLICY_VALUE_INTERACTION_PROBLEM = NO**
- **PRIMARY_G2_TO_G3_BOTTLENECK = POLICY**

Les deux réplications G3 apprennent légèrement mieux la cible Policy MCTS64
selon les métriques offline, mais leur Policy seule dégrade significativement
la force de jeu lorsque la Value reste celle de G2. À l'inverse, remplacer
uniquement la Value de G2 par celle de G3-A ou G3-B ne produit aucune
dégradation statistiquement détectable. L'interaction des têtes n'est pas la
cause principale : dans G3-B, la nouvelle Value atténue plutôt le défaut de la
nouvelle Policy.

## 1. Contrôles techniques

Les sept wrappers requis ont été construits sans créer de poids : H00, H10,
H01, H11, H20, H02 et H22. Sur une batterie fixe de 32 positions, chaque
wrapper restitue bit à bit la Policy et la Value de sa source déclarée.

Les empreintes des paramètres G2, G3-A et G3-B avant et après les diagnostics
sont identiques. Aucun `optimizer.step`, checkpoint hybride, entraînement,
self-play ou nouveau label n'a été créé.

## 2. Analyse Policy offline

Analyse exhaustive des 42 876 positions du corpus immutable G2→G3 :

| Modèle | JS | KL(π‖P) | Accord argmax | Action MCTS dans top-2 | Recouvrement top-2 | Entropie | Δ proba action MCTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| G2 | 0,026492 | 0,109422 | 32,39 % | 57,14 % | 56,28 % | 1,38450 | +0,15111 |
| G3-A | 0,025946 | 0,107120 | 34,46 % | 58,70 % | 57,14 % | 1,38023 | +0,14760 |
| G3-B | 0,026062 | 0,107622 | 34,03 % | 58,16 % | 56,81 % | 1,38302 | +0,14927 |

G3-A et G3-B se rapprochent légèrement de π_MCTS64 : KL baisse de 2,10 % et
1,65 %, et l'accord argmax gagne 2,07 et 1,64 points. Le gain est cependant
faible. Surtout, ces métriques moyennes ne mesurent ni l'importance stratégique
des positions où le classement change, ni la cohérence temporelle des cibles.

## 3. Analyse Value offline

42 664 positions portent un résultat terminal ; 212 positions tronquées sont
correctement exclues.

| Modèle | MSE | MAE | Sign accuracy | Prédiction moyenne | Écart-type |
|---|---:|---:|---:|---:|---:|
| G2 | 0,675290 | 0,693985 | 67,58 % | +0,02179 | 0,52711 |
| G3-A | 0,674921 | 0,709569 | 68,12 % | −0,01637 | 0,48820 |
| G3-B | 0,671896 | 0,705385 | 68,15 % | +0,02549 | 0,49535 |

Les MSE et précisions de signe de G3 sont légèrement meilleures, tandis que
les MAE sont moins bonnes et les sorties moins dispersées. La segmentation par
joueur ne révèle pas de rupture. Pour les trois modèles, le début de partie est
le segment le plus difficile (MSE environ 0,81), devant le milieu (environ
0,55). G3-A favorise davantage la détection des pertes mais dégrade celle des
victoires ; G3-B reste plus proche de G2. Ces compromis offline ne suffisent
pas à désigner un meilleur évaluateur, d'où les ablations en jeu.

## 4. Arènes hybrides MCTS64

Protocole commun : seed 20261616, 128 ouvertures uniques, inversion P1/P2,
256 parties par duel, `c_puct=1.5`, Dirichlet désactivé, température nulle,
argmax des visites et 20 000 réplications bootstrap appariées.

| Duel | Composante testée | W/D/L | Score | IC95 apparié | Troncatures |
|---|---|---:|---:|---:|---:|
| A : H20 vs H00 | P_G3B + V_G2 | 108/11/137 | 44,34 % | [39,26 %, 49,41 %] | 0 |
| B : H02 vs H00 | P_G2 + V_G3B | 124/9/121 | 50,59 % | [45,31 %, 56,64 %] | 2 |
| C : H22 vs H00 | G3-B complet | 120/11/124 | 49,22 % | [43,75 %, 54,30 %] | 1 |
| D : H10 vs H00 | P_G3A + V_G2 | 105/10/139 | 43,31 % | [37,70 %, 49,02 %] | 2 |
| E : H01 vs H00 | P_G2 + V_G3A | 132/5/117 | 52,95 % | [47,07 %, 58,01 %] | 2 |

Les duels Policy A et D ont tous deux leur borne supérieure sous 50 %. Le
phénomène est donc commun aux deux réplications. Les duels Value B et E ont
leurs intervalles autour de 50 % : aucune détérioration causale de la Value
n'est détectée.

### Résultats par côté

| Duel | Challenger P1 | Challenger P2 |
|---|---:|---:|
| A | 42/4/82 | 66/7/55 |
| B | 60/7/59 + 2 tronc. | 64/2/62 |
| C | 53/7/67 + 1 tronc. | 67/4/57 |
| D | 46/4/76 + 2 tronc. | 59/6/63 |
| E | 62/2/63 + 1 tronc. | 70/3/54 + 1 tronc. |

Les Policies G3 souffrent particulièrement comme P1. Leur score agrégé reste
cependant défavorable même après inversion des côtés.

## 5. Interprétation

Le blocage n'est pas une incapacité à réduire la loss Policy : G3 apprend bien
un peu mieux π_MCTS64. Le blocage se situe dans la **qualité effective de la
Policy apprise**. Trois faits convergent :

1. les gains offline Policy sont petits ;
2. remplacer uniquement P_G2 par P_G3-A ou P_G3-B réduit significativement la
   force MCTS64 ;
3. remplacer uniquement V_G2 par V_G3-A ou V_G3-B ne la réduit pas.

`POLICY_VALUE_INTERACTION_PROBLEM` vaut `NO` au sens causal demandé : la
combinaison complète G3-B remonte de 44,34 % (Policy seule) à 49,22 %, donc la
Value G3-B compense une partie du défaut. Cela n'implique pas que l'interaction
soit optimale, seulement qu'elle n'est pas la source principale observée.

## 6. Décision

**NEXT_EXPERIMENT = POLICY_FOCUSED_EXPERIMENT.**

L'expérience suivante doit analyser la qualité décisionnelle des changements
de classement induits par l'apprentissage de π_MCTS64, plutôt que réentraîner
avec une autre seed. Elle devra conserver G2 et le corpus fixe, identifier les
positions où G3 change l'argmax de G2, puis mesurer si ces changements sont
stables sous répétition/recherche et corrélés au résultat ou au regret de
recherche. La justification quantitative est directe : KL s'améliore de moins
de 2,1 %, tandis que les Policies isolées chutent à 44,34 % et 43,31 % en
arène. Aucun entraînement n'est lancé dans ce lot.

## Artefacts

- `data/experiments/lot16_policy_value/report.json`
- `data/experiments/lot16_policy_value/hybrid_identity.json`
- `data/experiments/lot16_policy_value/policy_offline.json`
- `data/experiments/lot16_policy_value/value_offline.json`
- `data/experiments/lot16_policy_value/arena_openings.json`
- cinq résumés `duel_*_summary.json` et cinq journaux `duel_*_games.jsonl`.
