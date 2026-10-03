# Lot 1A - Caracterisation du moteur avant correction

Date de l'audit : 24 septembre 2026.

Statut : termine sans modification de `rules.py`, `fast_rules.py`, de la
canonicalisation, des datasets, des caches ou des checkpoints.

Ce document decrit le comportement reel du code. Il ne tranche pas encore la
regle metier correcte. Les assertions qui figent un comportement suspect sont
des tests de caracterisation, destines a etre remplaces apres arbitrage du
Lot 1B.

## 1. Livrables et commandes de reproduction

Tests ajoutes :

- `packages/songo_ai/tests/test_lot1a_characterization.py` ;
- deux tests du mode strict dans `packages/songo_ai/tests/test_generation.py`.

Script ajoute :

- `apps/trainer/scripts/diagnose_lot1a_engine.py`.

Commandes :

```bash
.venv/bin/python apps/trainer/scripts/diagnose_lot1a_engine.py
.venv/bin/python apps/trainer/scripts/diagnose_lot1a_engine.py --json
.venv/bin/pytest -q packages/songo_ai/tests/test_lot1a_characterization.py packages/songo_ai/tests/test_generation.py
.venv/bin/pytest -q
```

Resultats obtenus :

```text
Tests Lot 1A + generation : 38 passed in 1.34s
Suite complete du depot   : 240 passed in 59.64s
```

## 2. Tour complet et magasin

### 2.1 Cas minimal construit

Le plus petit nombre de graines qui active le traitement "derniere graine
apres un tour complet" est 14. Les deux positions suivantes conservent 70
graines, sont legales et sont l'image l'une de l'autre par echange des camps.

```text
J1 = [14, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0]
J2 = [0, 0, 0, 0, 0, 4, 1, 14, 0, 0, 0, 0, 51, 0, 0, 0]
```

Dans les deux cas, l'action locale est `0` : case physique 0 pour J1 et case
physique 7 pour J2.

### 2.2 Detail J1

```text
Etat initial             [14,0,0,0,0,51,0,0,0,0,0,0,4,1,0,0]
Joueur                   1
Action locale            0
Case physique            0
Graines dans la case     14
Trajectoire              [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
Arrivee physique reelle  14 = magasin J1
Valeur retournee _sow    13
Valeur recue par capture 13
Cases capturees          [13]
Graines capturees        2
Magasin J1 avant/apres   0 -> 3
Etat final               [0,1,1,1,1,52,1,1,1,1,1,1,5,0,3,0]
```

Le magasin gagne une graine semee et deux graines capturees. La capture est
declenchee depuis la case 13 alors que la derniere graine est physiquement
arrivee dans le magasin 14.

### 2.3 Detail J2

```text
Etat initial             [0,0,0,0,0,4,1,14,0,0,0,0,51,0,0,0]
Joueur                   2
Action locale            0
Case physique            7
Graines dans la case     14
Trajectoire              [8,9,10,11,12,13,0,1,2,3,4,5,6,15]
Arrivee physique reelle  15 = magasin J2
Valeur retournee _sow    14
Valeur recue par capture 14
Cases capturees          []
Graines capturees        0
Magasin J2 avant/apres   0 -> 1
Etat final               [1,1,1,1,1,5,2,0,1,1,1,1,52,1,0,1]
```

La valeur 14 est egalement differente de l'arrivee physique, mais elle designe
un magasin. `_capture()` l'ignore donc et aucune capture parasite ne se produit.

### 2.4 Origine exacte de la difference

Pour les deux joueurs, le code place correctement la derniere graine dans le
magasin, affecte `next_index` a l'indice du magasin, puis quitte la boucle.
Il retourne ensuite inconditionnellement `next_index - 1`.

```text
J1 : arrivee physique 14 -> retour 13 -> case adverse capturable
J2 : arrivee physique 15 -> retour 14 -> magasin ignore par _capture
```

Il s'agit d'une asymetrie logicielle certaine. La decision de savoir si une
capture devait ou non avoir lieu reste une decision metier du Lot 1B.

### 2.5 Comparaison des implementations

`rules.py`, `fast_rules.py` et `docs/songo_legacy_single.py` donnent exactement
les memes etats apres semis et apres capture sur ces deux cas.

Cette concordance prouve leur equivalence historique, pas la validite de la
regle metier. Aucun fichier C#/Unity n'est present dans ce depot. Seules des
notes historiques mentionnent une ancienne divergence C# sur le cas distinct
de la graine unique jouee depuis une case de bord.

## 3. Canonicalisation

Propriete mesuree :

```text
C(T(S,a)) == T(C(S),a')
```

La comparaison porte sur les 14 cases, les deux magasins, le joueur relatif au
trait, les masques avant et apres, la capture, la transmission, la terminalite,
le gagnant et la raison terminale.

La batterie deterministe comprend :

- des positions manuelles ;
- 40 trajectoires legales generees avec la graine `20260924` ;
- des captures ;
- des transmissions ;
- des positions proches de la famine ;
- des etats terminaux ;
- des passages devant l'indice des magasins ;
- des tours complets avec depot dans le magasin.

Resultats identiques pour les deux moteurs :

| Moteur | Cas testes | Succes | Echecs |
|---|---:|---:|---:|
| `SongoLegacyGame` | 2 005 | 1 993 | 12 |
| `FastSongoGame` | 2 005 | 1 993 | 12 |

Repartition notable :

| Categorie | Testes | Succes | Echecs |
|---|---:|---:|---:|
| Manuel | 11 | 10 | 1 |
| Genere | 1 994 | 1 983 | 11 |
| Tour complet/depot magasin | 68 | 56 | 12 |
| Simple | 7 | 7 | 0 |
| Passage devant l'indice magasin sans depot final | 858 | 858 | 0 |
| Capture | 240 | 239 | 1 |
| Transmission | 2 | 2 | 0 |
| Proche famine | 7 | 7 | 0 |
| Terminal | 12 | 12 | 0 |

Les 12 echecs sont tous des positions ou la case jouee contient 14 graines et
ou la derniere graine est deposee dans le magasin apres le tour complet. Les
champs divergents sont systematiquement les cases, les magasins, le nombre de
graines capturees et le masque legal suivant. Trois cas divergent aussi sur la
capacite de transmission apres le coup.

Exemple minimal : le cas J2 de la section 2, action locale 0. L'original J2 ne
capture rien ; son image canonique jouee comme J1 capture deux graines.

Conclusion limitee : la canonicalisation actuelle fonctionne sur les 1 993
transitions testees qui n'exposent pas cette asymetrie, mais elle ne peut pas
etre declaree valide tant que la regle du tour complet n'est pas arbitree puis
que la batterie complete n'est pas rejouee.

## 4. Audit de `legal_moves(player)`

### 4.1 Appelants

Le seul appelant de production qui fournit explicitement `player` est
`search/negamax.py::default_evaluate()` :

```text
legal_moves(perspective) - legal_moves(opponent(perspective))
```

Il attend manifestement la mobilite de chacun des deux joueurs sur la meme
position. Le premier appel fonctionne lorsque `perspective == game.turn`. Le
second renvoie toujours une liste vide, car `is_legal_move()` continue de
verifier l'appartenance a `game.turn`.

Tous les autres appelants trouves utilisent `legal_moves()` sans argument et
attendent les coups physiques du joueur courant. Cet usage est coherent.

### 4.2 Comparaison observee

Position initiale, J1 au trait :

```text
legal_moves()           = [0,1,2,3,4,5,6]
legal_moves(PLAYER_ONE) = [0,1,2,3,4,5,6]
legal_moves(PLAYER_TWO) = []
legal_mask()            = [1,1,1,1,1,1,1]
```

Position initiale, J2 au trait :

```text
legal_moves()           = [7,8,9,10,11,12,13]
legal_moves(PLAYER_ONE) = []
legal_moves(PLAYER_TWO) = [7,8,9,10,11,12,13]
legal_mask()            = [1,1,1,1,1,1,1]
```

Le parametre `player` est donc ambigu : sa signature suggere une interrogation
pour n'importe quel joueur, mais son comportement reel ne fonctionne que pour
le joueur au trait.

Contrat futur propose :

```text
legal_moves() -> coups physiques du joueur au trait uniquement
legal_local_actions() -> actions locales 0..6 du joueur au trait
legal_mask() -> masque local du joueur au trait
```

Si la mobilite hypothetique de l'autre joueur reste necessaire, elle doit etre
calculee par une API explicite sur un etat dont le tour est controle, par
exemple `legal_moves_for(player)`, et non par un parametre ambigu de l'API
principale.

## 5. Terminalite

Le moteur combine :

- une memoire mutable : `finished` et `winner` ;
- un calcul avant `play()` pour camp vide et famine sans transmission ;
- un calcul apres le semis pour les magasins, `35/35` et la famine ;
- une normalisation explicite partielle via `normalize_terminal()`.

`legal_mask()` et `legal_moves()` ne normalisent pas automatiquement l'etat.

| Cas importe avec `finished=False` | Avant normalisation | Apres normalisation | Score final territorial |
|---|---|---|---|
| Magasin J1 = 36 | `finished=False`, coups `[0,5]` | inchange | `70-0` |
| Magasins `35/35` | aucun coup, pas de gagnant memorise | `finished=True`, draw | `35-35` |
| Famine sans transmission | aucun coup, pas de gagnant memorise | `finished=True`, J1 | `46-24` |
| Camp courant vide | aucun coup, pas de gagnant memorise | `finished=True`, J2 | `25-45` |

Les resultats sont identiques dans `rules.py` et `fast_rules.py`. Un second
appel a `normalize_terminal()` est idempotent sur tous ces cas.

Constat : la normalisation calcule camp vide et famine, mais ne reconnait pas
un etat importe dont un magasin depasse deja 35 tant que cet etat conserve des
coups legaux.

## 6. Actions illegales des agents

Le comportement historique reste le comportement par defaut :

```text
action agent illegale -> premiere action legale
```

Un parametre optionnel `strict_actions=True` a ete ajoute a
`generate_trajectory()` et `generate_trajectories()` :

```text
action agent illegale -> IllegalMove explicite
```

Le futur self-play RL devra toujours utiliser le mode strict. Aucun appelant
historique n'est modifie, car la valeur par defaut reste `False`.

## 7. Parties tronquees et cycles

Etat actuel :

- `RepetitionTracker` compte les repetitions mais ne modifie ni le jeu, ni la
  terminalite, ni le gagnant ;
- `generate_trajectory()` s'arrete a `max_moves` sans classifier le resultat ;
- `evaluation.play_match()` comptabilise une partie encore non terminee a
  `max_moves` comme une nulle de tournoi ;
- aucune regle de repetition n'est integree au moteur.

Contrat conceptuel reserve au futur pipeline RL :

```text
WIN
LOSS
DRAW
TRUNCATED
```

`TRUNCATED` ne doit produire aucun `z` tant qu'une politique explicite de
cycle, repetition ou adjudication n'a pas ete validee.

## 8. Conservation des graines

Les tests renforces couvrent :

- semis simple ;
- tour complet J1/J2 ;
- capture en cascade ;
- transmission vers un camp vide ;
- etats de famine ;
- 10 longues trajectoires aleatoires par moteur, jusqu'a 500 coups.

L'invariant suivant reste satisfait dans tous les cas executes :

```text
sum(board[0:16]) == 70
```

## 9. Tableau de synthese

| Comportement | Attendu selon code actuel | Observe | Symetrique ? | Confirme fast_rules ? | Confirme legacy historique ? | Decision metier necessaire ? |
|---|---|---|---|---|---|---|
| Tour complet J1, derniere graine magasin | Retour `next_index - 1` | arrivee 14, retour 13, capture case 13 | Non face a J2 | Oui | Oui | Oui |
| Tour complet J2, derniere graine magasin | Retour `next_index - 1` | arrivee 15, retour 14, aucune capture | Non face a J1 | Oui | Oui | Oui |
| Conservation des 70 graines | Toujours conservee | Confirmee | Oui | Oui | Oui sur les cas compares | Non |
| Canonicalisation simple | Echange des camps/magasins | 1 993/1 993 hors echecs observes | Oui sur l'echantillon | Oui | Non mesuree globalement | Apres correction tour complet |
| Canonicalisation tour complet | Meme action locale | 12 echecs sur 68 tours complets testes | Non | Oui | Cas minimal confirme | Oui |
| `legal_moves(player courant)` | Coups du tour | Conforme | Oui | Oui | Oui | Non |
| `legal_moves(autre joueur)` | Signature suggere une requete valable | Liste vide | Non pertinent | Oui | Oui | Contrat API a valider |
| Etat importe magasin >35 | `normalize_terminal` ne teste pas le magasin | reste non terminal | Oui entre moteurs | Oui | Comportement historique equivalent | Oui |
| Etat importe `35/35`, camps vides | Detecte via camp vide | draw apres normalisation | Oui | Oui | Non rejoue dans le script | Non, sauf contrat import |
| Famine/camp vide importe | Detecte a la normalisation | conforme au calcul territorial | Oui | Oui | Non rejoue dans le script | Validation expert recommandee |
| Action agent illegale, mode historique | Remplacement par premier coup legal | Confirme | Non pertinent | Utilise moteur rapide | Non pertinent | Non |
| Action agent illegale, mode strict | Exception | Confirme par test | Non pertinent | Utilise moteur rapide | Non pertinent | Non |
| `max_moves` en tournoi | Compte comme draw statistique | Confirme | Non pertinent | Oui | Non pertinent | Oui avant D_RL |
| Repetition | Compteur externe uniquement | Aucun effet terminal | Non pertinent | Hash commun | Non pertinent | Oui |

## 10. Anomalies certaines

1. `_sow()` ne retourne pas l'indice physique de la derniere graine lors du
   depot final dans un magasin apres un tour complet.
2. Cette valeur incorrecte produit une capture parasite possible pour J1,
   mais pas pour J2.
3. Cette asymetrie suffit a invalider la canonicalisation actuelle sur les
   transitions concernees.
4. `legal_moves(player)` ne respecte pas l'interpretation naturelle de son
   parametre lorsque `player != game.turn`.
5. `normalize_terminal()` ne traite pas les seuils de magasin sur un etat
   importe avec `finished=False`.
6. La validation actuelle d'un dataset importe peut consulter `finished`
   avant toute normalisation du moteur.

## 11. Comportements suspects mais non arbitres

1. Faut-il interdire toute capture lorsque la derniere graine est placee dans
   un magasin apres un tour complet ? Le comportement physique du semis le
   suggere, mais l'expert doit trancher.
2. Un snapshot importe avec magasin superieur a 35 doit-il etre normalise
   immediatement, ou doit-il etre considere invalide car impossible comme
   etat non terminal produit par `play()` ?
3. L'API doit-elle permettre de calculer les coups hypothetique d'un joueur
   qui n'est pas au trait ?
4. Quelle repetition constitue une nulle officielle ?
5. Une partie depassant un budget doit-elle etre reprise, ignoree, adjugee ou
   classee `TRUNCATED` ?
6. La divergence historique C#/Python sur la graine unique de bord reste a
   arbitrer si le moteur Unity redevient une source normative.

## 12. Consequences possibles d'une correction

| Composant | Consequence possible |
|---|---|
| `RULES_VERSION` | Incrementation obligatoire si un etat successeur, une capture ou une terminalite change. |
| Datasets teacher | Certaines positions, labels, PV et consequences peuvent avoir ete produits sous l'ancienne transition. Audit/regeneration a planifier. |
| Caches | Les cles dependent du hash et de la configuration teacher, pas directement de la version des regles. Une invalidation ou un nouveau namespace sera necessaire. |
| Checkpoints | Ils ont appris des labels issus de l'ancien moteur. Ils restent des artefacts historiques mais ne doivent pas etre presentes comme compatibles avec les nouvelles regles sans evaluation. |
| MLP | Features, Policy/WDL/Q et performance peuvent changer ; reentrainement probablement necessaire. |
| Negamax | L'arbre, les captures, la mobilite et l'heuristique peuvent changer. Le correctif `legal_moves(player)` change aussi le terme de mobilite. |
| Futur SRN | Le graphe dynamique doit etre construit uniquement apres stabilisation des transitions. |
| Futur MCTS | Les enfants, valeurs terminales et backups dependent directement du moteur corrige. |
| Futur `D_RL` | Aucun corpus RL ne doit etre genere avant gel de `RULES_VERSION`, de la perspective et du statut `TRUNCATED`. |

## 13. Proposition de Lot 1B

1. Faire arbitrer par l'expert le depot final en magasin apres tour complet,
   la capture associee et la regle des snapshots de magasin terminal.
2. Transformer les decisions en tests metier normatifs distincts des tests de
   caracterisation.
3. Corriger d'abord `rules.py`, puis porter strictement le resultat dans
   `fast_rules.py`.
4. Rejouer la batterie d'equivalence entre implementations.
5. Rejouer la batterie de canonicalisation et exiger zero echec avant de
   figer la representation de perspective.
6. Clarifier `legal_moves()` et corriger le calcul de mobilite Negamax sans
   conserver une API ambigue.
7. Definir la normalisation des snapshots importes et renforcer
   `validate_observation()`.
8. Definir `DRAW` versus `TRUNCATED` et la politique de repetition avant le
   futur self-play.
9. Incrementer `RULES_VERSION`, isoler les anciens caches et produire une
   matrice de compatibilite des datasets/checkpoints.
10. Executer les tests complets, les tests differentiels et un audit cible des
    anciennes releases avant toute implementation GraphBuilder, SRN ou MCTS.

