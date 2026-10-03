# Lot 9A — Symétrie P1/P2 et qualité informationnelle de D_RL

## 1. Décision

Le passage immédiat à G2 est suspendu. Le Lot 9A met en évidence deux défauts
qui doivent être traités avant de produire une nouvelle génération :

1. G1 n'est pas invariant au simple renommage des joueurs physiques P1/P2 ;
2. les cibles Policy du pilote D_RL sont trop pauvres pour représenter une
   préférence de recherche stable.

Ce lot est exclusivement diagnostique. Il n'a effectué aucun entraînement,
n'a modifié aucun poids et n'a promu aucun checkpoint.

## 2. Contrat de symétrie testé

Le réseau travaille avec des actions **locales** `0..6` et sa Value est définie
du point de vue du joueur au trait. Un échange des identités physiques doit
donc conserver la décision stratégique exprimée dans ce repère local.

Pour un état brut :

```text
S = (cases P1, cases P2, magasin P1, magasin P2, joueur)
```

la transformation testée est :

```text
mirror(S) = (cases P2, cases P1, magasin P2, magasin P1, adversaire(joueur))
```

Les indices d'actions locales ne changent pas. Le contrat attendu est alors :

```text
P(a | S) = P(a | mirror(S))
V(S)     = V(mirror(S))
```

Cette transformation n'est pas une hypothèse sur le réseau. Elle a été
validée sur le moteur : elle conserve le statut terminal, échange le gagnant
physique, conserve le masque d'actions locales et commute avec la transition
du jeu pour une même action locale.

## 3. Protocole expérimental

L'audit utilise les 350 positions de la batterie D_LAB élargie du Lot 8 :

- 175 états matérialisés comme P1 et 175 comme P2 ;
- sélection indépendante des sorties des réseaux et des labels teacher ;
- vérification de la légalité avant chaque inférence ;
- comparaison de chaque position avec son miroir physique exact ;
- aucune recherche MCTS dans ce test : seules les sorties brutes sont auditées.

Trois réseaux sont comparés : G0 non entraîné, G1-best (époque 8) et G1-last
(époque 12). La Policy est comparée par divergence de Jensen-Shannon et par
accord de l'argmax. La Value est comparée par différence absolue.

## 4. Résultats de symétrie

| Modèle | Écart Value moyen | Médiane | Maximum | JS Policy moyenne | Accord argmax |
|---|---:|---:|---:|---:|---:|
| G0 | 0,01834 | 0,01844 | 0,02381 | 0,0000479 | 71,71 % |
| G1-best | **0,91558** | **0,94895** | **1,86073** | 0,0001562 | **16,00 %** |
| G1-last | **0,66461** | **0,54957** | **1,81230** | 0,0003135 | **36,86 %** |

L'entraînement n'a pas seulement amplifié une petite asymétrie initiale. Il a
appris une séparation presque binaire entre les identités physiques dans de
nombreux états.

Pour G1-best :

| Identité de l'état original | Value originale moyenne | Value du miroir moyenne | Écart absolu moyen |
|---|---:|---:|---:|
| P1 | +0,40567 | -0,46847 | 0,87414 |
| P2 | -0,48208 | +0,47494 | 0,95701 |

Sur la pire paire, la même position locale reçoit `-0,94913` comme P2 et
`+0,91159` après renommage en P1, soit un écart de `1,86073` sur une sortie
bornée dans `[-1, +1]`.

La divergence Policy moyenne reste faible parce que les distributions sont
très plates. Cela ne rend pas le défaut négligeable : de petits écarts changent
l'action classée première dans 84 % des paires pour G1-best. Ce constat est
cohérent avec le diagnostic Policy du Lot 8.

## 5. Audit du pilote D_RL

Le pilote contient 20 parties et 2 593 exemples. Toutes les positions ont été
produites avec exactement **deux visites MCTS**.

### 5.1 Information contenue dans les cibles Policy

| Mesure | Résultat |
|---|---:|
| Cibles à support 1 | 1 439 / 2 593 |
| Cibles à support 2 | 1 154 / 2 593 |
| Fraction one-hot | **55,50 %** |
| Entropie moyenne | 0,30848 nat |
| Entropie médiane | **0** |

Avec deux simulations, une cible ne peut attribuer des visites qu'à une ou
deux actions. Plus de la moitié des exemples redeviennent donc, en pratique,
des labels one-hot. La cible conserve formellement le contrat `pi_MCTS`, mais
elle ne contient pas encore la richesse recherchée d'une distribution de
recherche.

### 5.2 Instabilité sur les états répétés

Le corpus contient 2 559 états uniques. Huit états apparaissent plusieurs
fois et les huit possèdent des cibles Policy contradictoires. Le cas le plus
clair est la position initiale :

- 20 occurrences, une par partie ;
- 16 cibles Policy distinctes ;
- même état brut et même masque légal.

Un réseau déterministe ne peut pas reproduire simultanément 16 distributions
différentes pour la même entrée. Cette dispersion n'est pas anormale pour une
recherche exploratoire, mais, avec seulement deux visites, le bruit domine le
signal et rend la cible Policy peu informative.

### 5.3 Corrélation entre résultat et identité physique

Le nombre de positions est presque équilibré : 1 302 avec P1 au trait et
1 291 avec P2. En revanche, les résultats ne le sont pas :

| Joueur physique au trait | Cible Value moyenne | `z=+1` | `z=-1` |
|---|---:|---:|---:|
| P1 | +0,37327 | 68,66 % | 31,34 % |
| P2 | -0,36793 | 31,60 % | 68,40 % |

Le pilote permet donc de réduire fortement la loss en exploitant l'identité
physique. Le comportement de G1 est fortement compatible avec ce raccourci.
Cet audit établit la corrélation et l'effet appris ; il ne prétend pas isoler à
lui seul toute la chaîne causale.

## 6. Origine architecturale possible du raccourci

Le graphe v0.1 n'est pas canonicalisé. Il expose explicitement :

- deux indicateurs d'appartenance absolue `owner == P1` et `owner == P2` ;
- l'indice physique du nœud `0..13` ;
- deux indicateurs globaux `player == P1` et `player == P2` ;
- les magasins dans l'ordre physique P1 puis P2 ;
- des nœuds d'action `0..6` pour P1 et `7..13` pour P2.

Ces informations rendent le raccourci accessible. Elles ne suffisent pas à
elles seules à prouver qu'il sera toujours appris : c'est leur combinaison
avec le déséquilibre des résultats du petit pilote qui pose problème.

## 7. Conséquence pour le contrat `(S, M, pi, z)`

Le contrat conceptuel reste valide :

```text
Input       = S
Constraint  = M
PolicyLabel = pi_MCTS
ValueLabel  = z_terminal, du point de vue du joueur au trait
```

Le défaut ne vient donc pas d'une confusion entre les champs. Il vient de la
représentation physique de `S`, de la distribution expérimentale des `z` et du
budget insuffisant utilisé pour estimer `pi`.

Une augmentation miroir correcte doit préserver :

```text
(S, M, pi, z) -> (mirror(S), M, pi, z)
```

`z` ne change pas : le gagnant physique change, mais la perspective locale du
joueur au trait reste la même. De même, `pi` ne change pas puisque les actions
sont locales.

## 8. Conditions avant G2

Le prochain lot d'implémentation devra traiter séparément les variables
suivantes, avec ablations :

1. fixer une canonicalisation ou une équivariance explicite de l'entrée ;
2. empêcher les features absolues P1/P2 de servir de raccourci ;
3. ajouter une augmentation miroir validée par le moteur ou régénérer un D_RL
   équilibré ;
4. comparer plusieurs budgets MCTS supérieurs à deux, sans déclarer à
   l'avance une valeur définitive ;
5. suivre l'écart de symétrie de Value, la JS Policy et l'accord d'argmax à
   chaque checkpoint ;
6. sélectionner un checkpoint sur la calibration et le jeu, pas uniquement
   sur la loss de validation.

L'augmentation des données ne doit pas masquer une architecture incohérente,
et la canonicalisation ne doit pas être déclarée suffisante sans vérifier la
qualité de `pi`. Les deux problèmes doivent être mesurés indépendamment.

## 9. Critères de passage proposés

Les seuils numériques d'acceptation ne sont pas encore figés : ils doivent
être déterminés à partir de contrôles et d'ablations. En revanche, les
invariants suivants sont non négociables avant G2 :

- la transformation miroir conserve exactement le moteur et les labels ;
- un exemple et son miroir appartiennent au même split de données ;
- aucune identité physique ne doit prédire systématiquement le signe de Value ;
- l'audit de symétrie doit être exécuté avant toute promotion ;
- le budget de recherche et la stochasticité de MCTS doivent être enregistrés ;
- D_LAB demeure un laboratoire indépendant et n'entre pas dans l'entraînement.

## 10. Reproductibilité et artefacts

Commande :

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages \
  .venv/bin/python apps/trainer/scripts/audit_srn_lot9a.py
```

Artefacts :

- `packages/songo_ai/evaluation/symmetry_audit.py` ;
- `apps/trainer/scripts/audit_srn_lot9a.py` ;
- `packages/songo_ai/tests/test_srn_symmetry_audit.py` ;
- `data/experiments/lot9a_symmetry_audit_seed_20260924/report.json`.

Le rapport contient les hachages SHA-256 de D_LAB, D_RL et des deux
checkpoints, les dix pires paires par modèle et les empreintes des paramètres
avant/après. Les trois empreintes sont inchangées.
