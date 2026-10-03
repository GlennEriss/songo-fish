# Lot 11 — Diagnostic MCTS final avant G2

## Verdict

**READY_FOR_G2 : YES**

Le budget retenu est **64 simulations MCTS par position**. Aucun entraînement
G2, aucune promotion de modèle et aucune modification du moteur ou de
l'architecture SRN n'ont été réalisés dans ce lot.

Le diagnostic utilise le checkpoint G1-best du Lot 6 (meilleure époque : 8),
une batterie fixe de 128 positions comprenant explicitement l'état initial
S0, et 8 graines par position. Ces valeurs correspondent aux bornes basses
prédéfinies (128–256 positions et au moins 8 graines) afin de borner le coût
CPU tout en conservant une couverture structurelle D_LAB indépendante des
sorties du modèle.

## Courbe budget / qualité / coût

Les états forcés, qui ne possèdent qu'une action légale, sont exclus des
statistiques inter-seeds. Le bruit Dirichlet est actif dans l'étude principale.

| Simulations | JS inter-seeds moyen | Accord argmax | Support moyen | Entropie moyenne | One-hot |
|---:|---:|---:|---:|---:|---:|
| 8  | 0,1790 | 0,3096 | 3,287 | 0,927 | 17,07 % |
| 16 | 0,1214 | 0,3342 | 3,689 | 1,099 | 3,15 % |
| 32 | 0,0530 | 0,3733 | 4,196 | 1,269 | 0 % |
| 64 | 0,0214 | 0,4298 | 4,452 | 1,312 | 0 % |

Le budget 16 ne satisfait pas les critères relatifs de stabilité. Le budget
32 satisfait les critères de qualité, mais l'amélioration vers 64 n'est pas
encore un plateau : la JS baisse encore de 0,0316 et l'accord argmax gagne
5,65 points. Le budget 64 est donc le plus petit budget admissible dans la
grille étudiée.

Le contrôle sans bruit Dirichlet donne une JS inter-seeds exactement nulle à
tous les budgets. L'instabilité mesurée avec bruit provient donc de
l'interaction exploration/budget, et non d'une non-déterminisme caché du
moteur, du réseau ou de MCTS.

## Ablations causales

Les variantes d'une même ablation utilisent les mêmes graines. Aucun modèle
n'est entraîné et aucun checkpoint n'est produit.

À Policy G1 fixe et budget 64 :

| Couple Policy / Value | JS moyenne | Accord argmax |
|---|---:|---:|
| P1V1 | 0,0214 | 0,4112 |
| P1V0 | 0,0276 | 0,2941 |
| P1Vneutral | 0,0208 | 0,2947 |

La Value G1 améliore surtout la cohérence de l'argmax. La remplacer par G0
dégrade simultanément la JS et l'accord. Une Value neutre conserve une JS
faible mais perd nettement en accord argmax : une faible JS ne suffit donc pas
à elle seule à conclure sur la qualité de décision.

L'échelle de Value confirme ce comportement : `alpha=0`, `0,5`, `1` donnent
respectivement des accords argmax de 0,2832, 0,3853 et 0,4355. La Value apprise
influence donc réellement le classement final des actions, même lorsque les
distributions restent globalement proches.

Pour `c_puct`, 0,75 est moins stable (JS 0,0396) que 1,5 (0,0156) et 3,0
(0,0150). `c_puct=1,5` est conservé : le gain marginal de JS à 3,0 s'accompagne
d'une baisse de l'accord argmax et ne justifie pas un changement de la
configuration de référence.

## Traces et Value des feuilles

Les premières simulations de S0 et des trois positions les plus instables ont
été tracées jusqu'aux feuilles. Chaque trace conserve l'action racine, le
prior, les visites, Q après backup, l'état feuille, son caractère terminal et
la Value. La convention est explicite : la Value feuille est exprimée du point
de vue du joueur au trait dans la feuille ; Q racine est exprimé du point de
vue du joueur au trait à la racine.

Sur les 1 024 mêmes feuilles non terminales :

- G0 reste proche de zéro : moyenne −0,0432, écart-type 0,0114 ;
- G1 est très polarisé : moyenne 0,2514, écart-type 0,7432 ;
- 90,53 % des valeurs G1 ont une valeur absolue supérieure à 0,5 ;
- 49,02 % dépassent 0,8 et 8,40 % dépassent 0,95.

Cette polarisation explique pourquoi la Value peut verrouiller tôt certains
classements à faible budget. Elle reste un risque à surveiller dans G2, mais
le passage à 64 simulations réduit fortement la sensibilité inter-seeds et le
pilote satisfait tous les critères de corpus.

## Pilote indépendant à 64 simulations

Le pilote contient 50 parties indépendantes et 4 123 positions : 47 victoires
terminales et 3 nulles, sans troncature ni exemple non labellisé. Le round-trip
JSONL est exact et les empreintes des paramètres avant/après diagnostic sont
identiques.

Comparé au corpus G1/8 du Lot 10 :

| Indicateur | Lot 10 G1/8 | Pilote G1/64 |
|---|---:|---:|
| One-hot | 23,64 % | 3,57 % |
| Support moyen | 3,241 | 4,578 |
| Entropie moyenne | 0,871 | 1,338 |
| JS de S0 | 0,5449 | 0,0090 |

Les cinq contrôles du gate passent : réduction du one-hot, gain de support,
gain d'entropie, stabilisation de S0, sérialisation et labels terminaux valides.

## Configuration autorisée pour le prochain lot

- checkpoint initial : G1-best, époque 8 ;
- 500 parties de self-play ;
- 64 simulations MCTS ;
- `c_puct=1.5` ;
- bruit Dirichlet : `alpha=0.3`, `epsilon=0.25` ;
- température cible : 1,0 ;
- température d'action : 1,0 jusqu'au ply 30, puis 0 ;
- limite de répétition : 3 ;
- limite de partie : 400 plies ;
- corpus estimé : environ 41 230 positions ;
- coût linéaire estimé à partir du pilote : environ 975 secondes sur la
  machine de mesure, hors entraînement.

Cette autorisation concerne uniquement la génération du corpus et
l'entraînement G2 du prochain lot. Elle ne vaut pas promotion automatique : G2
devra encore être comparé à G1-best par une arène indépendante.

## Artefacts

- `data/experiments/lot11_g1_budget_seed_20260924/report.json` : rapport de
  référence et gate final ;
- `budget_curve.json` : courbe complète, décision et détail de S0 ;
- `ablations.json` : contrôles Dirichlet, Policy/Value, alpha et `c_puct` ;
- `simulation_traces.json` : traces précoces et distributions de feuilles ;
- `battery.json` : provenance des 128 positions ;
- `data/d_rl/lot11_g1_selected_seed_20260924.jsonl` : pilote indépendant.

La suite de tests complète compte **382 tests réussis**.
