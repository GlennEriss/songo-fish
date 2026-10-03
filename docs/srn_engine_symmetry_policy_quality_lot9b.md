# Lot 9B — Symétrie réelle du moteur et qualité des cibles MCTS

## 1. Question et décision

La transformation P1/P2 pertinente est l'échange des camps **avec conservation
de l'ordre local**. Elle conserve toujours la légalité sur les batteries
testées, mais elle n'est pas une symétrie exacte de toutes les transitions du
moteur historique.

La situation relève donc de la catégorie **B** : symétrie exacte sauf une
famille rare, reproductible et précisément identifiable. Cela n'autorise ni
une canonicalisation globale, ni une augmentation miroir générale tant que le
comportement historique du tour complet n'est pas arbitré.

En parallèle, l'essentiel de la variabilité Policy observée dans le pilote
D_RL vient bien de la combinaison `Dirichlet + seulement deux simulations`.
Avec deux simulations, l'exploration devient directement le label.

Ce lot n'a modifié ni le moteur, ni le SRN, ni D_RL. Il n'a effectué aucun
entraînement et n'a produit aucun checkpoint.

## 2. Transformations testées

### 2.1 Échange avec ordre local conservé

Pour :

```text
S = (pits 0..6, pits 7..13, store 14, store 15, player_to_move)
```

la transformation `T_keep` est :

```text
new pits 0..6  = old pits 7..13
new pits 7..13 = old pits 0..6
new store 14   = old store 15
new store 15   = old store 14
new player     = swap(old player)
T_action(a)    = a
T_winner(P1)   = P2
T_winner(P2)   = P1
T_winner(DRAW) = DRAW
```

Cette transformation est une involution : `T_keep(T_keep(S)) = S`.

### 2.2 Échange avec ordre local inversé

La seconde candidate inverse chaque camp :

```text
new pits 0..6  = reverse(old pits 7..13)
new pits 7..13 = reverse(old pits 0..6)
T_action(a)    = 6-a
```

Elle est également involutive, mais elle n'est pas compatible avec la
direction de semis du moteur actuel.

## 3. Réconciliation avec le Lot 9A

Le Lot 9A avait validé le statut terminal et les masques légaux sur 350 paires,
puis mesuré les sorties du réseau. Il n'avait pas testé chaque successeur.

Cette distinction résout le paradoxe :

- `legal(T(S)) = T_action(legal(S))` peut être vrai ;
- alors que `T(F(S,a)) = F(T(S),T_action(a))` échoue après le coup.

Sur les trois grandes batteries du Lot 9B, `T_keep` produit **zéro échec de
masque légal**. Les échecs apparaissent exclusivement dans les transitions.

## 4. Reproduction exacte des 2 005 transitions du Lot 1A

| Transformation | Testées | Succès | Échecs | Taux d'échec |
|---|---:|---:|---:|---:|
| Ordre local conservé | 2 005 | 1 993 | 12 | 0,5985 % |
| Ordre local inversé | 2 005 | 34 | 1 971 | 98,3042 % |

Les douze échecs de `T_keep` ont tous les propriétés suivantes :

- la case jouée contient 14 graines ;
- la dernière graine du tour complet est déposée dans le magasin ;
- `_sow()` retourne l'indice précédent au lieu de l'arrivée physique ;
- le cas matérialisé comme P1 déclenche une capture que son miroir P2 ne
  déclenche pas, ou inversement ;
- les cases et les magasins divergent après la transition.

L'audit historique retrouve en plus les conséquences dérivées déjà décrites :
12 divergences de capture et de masque suivant, dont 3 divergences de capacité
de transmission.

`T_reverse` est définitivement rejetée : elle échoue sur les semis ordinaires,
la légalité de 99 cas, les captures, certains terminaux et certains gagnants.

## 5. Fréquence réelle de l'asymétrie moteur

Les mesures suivantes testent **toutes les actions légales** de chaque état.
Une trajectoire est marquée si au moins un état possède au moins une action
potentiellement non équivariante, même si cette action n'a pas été jouée.

### 5.1 Transformation retenue `T_keep`

| Source | États | États affectés | Actions légales | Échecs | Trajectoires affectées |
|---|---:|---:|---:|---:|---:|
| D_LAB complet | 10 000 | 187 (1,87 %) | 43 518 | 196 (0,450 %) | 133/401 (33,17 %) |
| D_RL Lot 5 | 2 593 | 39 (1,50 %) | 12 522 | 41 (0,327 %) | 14/20 (70 %) |
| 100 trajectoires nouvelles | 10 737 | 256 (2,38 %) | 51 618 | 264 (0,511 %) | 84/100 (84 %) |

Les 501 échecs cumulés des trois sources appartiennent tous à la famille
`final_store_deposit`. Aucune autre famille n'a été observée.

Les taux par trajectoire sont élevés parce qu'une longue trajectoire offre de
nombreuses occasions de rencontrer une action potentiellement fautive. Ils ne
doivent pas être confondus avec le taux par transition, qui reste inférieur à
0,52 %.

### 5.2 Actions réellement jouées dans D_RL

Sur les 2 593 transitions effectivement jouées :

- 4 sont non équivariantes, soit `0,1543 %` ;
- elles appartiennent à 4 parties sur 20, soit `20 %` des trajectoires.

L'asymétrie moteur est donc rare par action, mais elle n'est ni théorique ni
absente du corpus RL.

## 6. Terminal, gagnant et cible Value

La cible est définie par :

```text
z(S) = +1 si winner(S) == player_to_move(S)
       0 si DRAW
      -1 sinon
```

Sous une transition équivariante, `T` échange à la fois le gagnant physique et
le joueur physique au trait. L'égalité entre les deux est donc conservée :

```text
z(T(S)) = z(S)
```

La cible ne change pas de signe. Les terminaux équivariants testés ne
présentent aucun échec de cette relation. En revanche, aucune conclusion ne
peut être imposée à une trajectoire dont une transition a déjà divergé.

## 7. Mapping structurel du masque et de la Policy

Pour `T_keep` :

```text
M_T[a]  = M[a]
pi_T[a] = pi[a]
```

Pour `T_reverse`, qui est rejetée :

```text
M_T[6-a]  = M[a]
pi_T[6-a] = pi[a]
```

Ce mapping définit seulement le transport des coordonnées d'action. Il ne
prétend pas que deux recherches MCTS bruitées doivent produire la même cible.

## 8. Origine du biais Value P1/P2 dans D_RL

Le pilote contient 12 victoires P1 et 8 victoires P2.

| Gagnant | Parties | Longueur moyenne | Minimum | Maximum |
|---|---:|---:|---:|---:|
| P1 | 12 | 148,08 | 65 | 265 |
| P2 | 8 | 102,00 | 32 | 158 |

Les positions sont presque équilibrées : 1 302 tours P1 et 1 291 tours P2.
Cependant :

| Perspective physique | z moyen observé | z si chaque partie avait le même poids | Effet longueur |
|---|---:|---:|---:|
| P1 au trait / ply pair | +0,37327 | +0,20000 | +0,17327 |
| P2 au trait / ply impair | -0,36793 | -0,20000 | -0,16793 |

Le déséquilibre a donc deux causes mesurées :

1. les 12 victoires P1 contre 8 produisent déjà un signal `±0,20` ;
2. les parties gagnées par P1 sont beaucoup plus longues et amplifient ce
   signal jusqu'à environ `±0,37` lors d'un échantillonnage par position.

Comme le joueur alterne depuis P1, l'identité physique et la parité du ply sont
exactement confondues dans ce corpus. Le modèle peut réduire sa loss en
exploitant ce raccourci sans comprendre la position.

## 9. Audit des vingt positions initiales

L'audit a reconstruit le modèle exact du Lot 5 : SRN aléatoire, dimension 16,
deux blocs et seed `20260924`. Les seeds de recherche et d'action ont été
recalculées avec les fonctions déterministes du self-play.

- les vingt visit counts sont reproduits exactement : 20/20 ;
- les vingt actions jouées sont reproduites exactement : 20/20 ;
- les priors réseau bruts sont identiques dans les vingt cas ;
- avec Dirichlet, les vingt recherches donnent 16 cibles différentes ;
- avec les mêmes vingt seeds mais Dirichlet désactivé, elles donnent une seule
  cible ;
- cinq répétitions avec une seed fixe donnent exactement le même résultat.

Les priors bruts sont presque uniformes, entre `0,14018` et `0,14415`. La
variabilité ne vient donc ni d'un changement de réseau, ni du dataset, ni d'un
non-déterminisme non contrôlé. Elle vient du bruit de Dirichlet, dont l'effet
est discrétisé et amplifié par seulement deux visites. Aucun effet résiduel de
tie-break n'est observé sans Dirichlet sur ce cas.

Le fichier de Lot 5 ne stockait pas les priors ni le vecteur Dirichlet. Ils ont
été reconstruits, et non prétendus présents dans D_RL. Le rapport JSON conserve
les vingt vecteurs et toutes les seeds reconstruites.

## 10. Stabilité MCTS sans Dirichlet

La batterie fixe comprend 16 positions D_LAB, dont 14 avec plusieurs actions
légales. Chaque combinaison est répétée huit fois avec des seeds différentes.

Sans Dirichlet, les cibles sont identiques entre les huit seeds à tous les
budgets : JS moyenne `0`, accord argmax `100 %`, une seule cible par état. Les
tie-breaks seedés n'ont donc créé aucune variabilité observable sur cette
batterie.

Sur les 14 positions non forcées :

| Budget | One-hot | Actions visitées moy. | Entropie moyenne |
|---:|---:|---:|---:|
| 2 | 0 % | 2,00 | 0,6931 |
| 8 | 0 % | 4,14 | 1,3335 |
| 32 | 0 % | 4,14 | 1,3506 |
| 64 | 0 % | 4,14 | 1,3286 |

Deux visites produisent mécaniquement une cible de support maximal 2, même
lorsque cette cible est parfaitement reproductible.

## 11. Stabilité MCTS avec Dirichlet

Résultats sur les 14 positions non forcées :

| Budget | One-hot | Actions visitées | Entropie | JS inter-seeds | Accord argmax |
|---:|---:|---:|---:|---:|---:|
| 2 | **50,0 %** | 1,50 | 0,3466 | **0,4275** | 31,63 % |
| 8 | 0 % | 4,12 | 1,3100 | 0,0252 | 41,84 % |
| 32 | 0 % | 4,14 | 1,2647 | 0,0422 | 27,55 % |
| 64 | 0 % | 4,14 | 1,2501 | 0,0447 | 38,01 % |

Le saut qualitatif principal se situe entre 2 et 8 simulations : disparition
des one-hot non forcés, support moyen multiplié par 2,7 et JS inter-seeds
divisée par environ 17.

Les budgets 32 et 64 ne rendent toutefois pas l'argmax monotoniquement plus
stable avec ce réseau initial presque indifférent. Le bruit racine reste
volontairement visible. Le Lot 9B ne fixe donc pas 8 comme valeur définitive ;
il établit seulement que 2 est insuffisant et que 8 est le premier candidat
raisonnable à comparer à 32 et 64 dans une génération contrôlée.

## 12. Exploration et label

Le chemin causal est ici mesuré directement :

```text
Dirichlet
  -> priors racine différents
  -> choix PUCT différents
  -> visit_counts de support 1 ou 2
  -> pi_target directement différente
```

À deux simulations avec Dirichlet, la JS entre répétitions vaut `0,3741` sur
les 16 positions et `0,4275` hors positions forcées. L'accord argmax n'est que
de `40,18 %` et `31,63 %` respectivement. La cible est donc majoritairement un
échantillon discret de l'exploration, pas encore une estimation riche de la
préférence de recherche.

## 13. Séparation des deux asymétries

### A. Asymétrie du moteur

- exacte et localisée au traitement historique du dépôt final après tour
  complet dans toutes les observations ;
- entre `0,327 %` et `0,511 %` des actions légales selon la source ;
- présente sur 4 transitions réellement jouées du pilote.

### B. Asymétrie statistique et d'apprentissage

- résultat position-pondéré `z≈+0,373` pour P1 et `z≈-0,368` pour P2 ;
- identité physique parfaitement corrélée à la parité du ply ;
- priors du réseau initial quasi uniformes ;
- 55,5 % de one-hot dans D_RL et 16 cibles initiales distinctes ;
- variabilité initiale reproduite entièrement par Dirichlet avec deux visites.

L'asymétrie moteur interdit juridiquement une symétrie globale, mais sa
fréquence par action est trop faible pour expliquer à elle seule l'écart Value
miroir moyen de `0,916` appris par G1-best. Les raccourcis statistiques et la
faible qualité des cibles sont les explications dominantes mesurées, sans que
l'audit prétende décomposer causalement chaque paramètre du réseau.

## 14. Décision sur la canonicalisation

**Catégorie B.** `T_keep` est exacte sauf une famille rare et identifiable,
mais la canonicalisation globale reste interdite pour le moteur actuel.

Le prédicat exact au niveau d'une transition est :

```text
eligible(S,a) <=>
  T(F(S,a)) == F(T(S),T_action(a))
```

La comparaison inclut cases, magasins, joueur au trait, terminal et gagnant.
Ce prédicat rejoue les deux transitions du moteur gelé ; il ne remplace pas
l'anomalie par une nouvelle règle supposée.

Le raccourci empirique « pas de dépôt final après tour complet » décrit toutes
les observations actuelles, mais n'est pas élevé au rang de preuve pour tous
les états futurs.

## 15. Décision sur l'augmentation miroir

L'augmentation générale :

```text
(S,M,pi,z) + (T(S),T(M),T(pi),z_T)
```

n'est pas mathématiquement valide sans condition.

- Pour une trajectoire observée, chaque transition jouée doit passer le
  prédicat et le terminal doit échanger le gagnant. Alors `z_T=z`.
- Pour une cible Policy, toutes les transitions visitées par l'arbre MCTS qui
  a produit `pi` doivent être équivariantes.

Le format D_RL actuel ne conserve pas les arêtes de l'arbre de recherche. Il
est donc impossible de certifier rétrospectivement toutes les cibles Policy du
pilote. Même les 16 trajectoires sans échec joué ne suffisent pas : MCTS a pu
explorer une action non équivariante qui n'a finalement pas été jouée.

Il ne faut donc pas augmenter le pilote existant.

## 16. Régularisation de symétrie

Une loss globale `L_sym_policy + L_sym_value` n'est pas justifiée tant que le
moteur n'est pas globalement symétrique. Elle pourrait être envisagée avec un
masque limité aux exemples dont tout le sous-arbre pertinent est certifié,
mais ce mécanisme serait coûteux et fragile. L'arbitrage du moteur est une
solution conceptuellement plus propre.

## 17. Recommandation pour le prochain D_RL

Avant G2 :

1. arbitrer puis tester le comportement du tour complet et du dépôt magasin ;
2. ne pas canonicaliser ni augmenter les labels actuels ;
3. enregistrer dans les prochains diagnostics les priors racine, la seed de
   recherche, les visit counts et un indicateur d'équivalence des arêtes
   explorées ;
4. exclure le budget 2 comme source principale de labels ;
5. comparer au minimum les budgets 8 et 32, avec 64 comme contrôle de coût ;
6. équilibrer les gagnants physiques **et la longueur des parties**, pas
   seulement le nombre de positions P1/P2 ;
7. conserver D_LAB hors entraînement ;
8. exécuter les audits de symétrie et de stabilité avant toute promotion.

## 18. Tests et reproductibilité

Tests ajoutés :

- involution des deux transformations ;
- mapping action, masque, Policy et gagnant ;
- dérivation de `z_T=z` ;
- équivariance des transitions initiales pour `T_keep` ;
- rejet de `T_reverse` ;
- contre-exemple historique à 14 graines ;
- déterminisme des agrégats moteur et Policy ;
- absence de mutation du moteur.

Commande :

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages \
  .venv/bin/python apps/trainer/scripts/audit_srn_lot9b.py
```

Artefacts :

- `packages/songo_ai/evaluation/engine_symmetry.py` ;
- `packages/songo_ai/evaluation/policy_target_diagnostics.py` ;
- `apps/trainer/scripts/audit_srn_lot9b.py` ;
- `packages/songo_ai/tests/test_engine_symmetry_lot9b.py` ;
- `data/experiments/lot9b_symmetry_engine_seed_20260924/report.json`.
