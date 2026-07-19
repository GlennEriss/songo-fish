# Intégrer SongoFish (réseau + recherche) dans un jeu C#

Ce document s'adresse au développeur qui portera le moteur de décision
SongoFish dans le jeu Songo en C# — pas à quelqu'un qui connaît déjà le
code Python de ce dépôt. Il couvre : où sont les fichiers, comment
récupérer le réseau (ONNX), l'algorithme de recherche à réimplémenter, et
surtout **le point qui a causé une confusion réelle pendant le
développement : profondeur de recherche ≠ temps de réflexion.**

## Checklist de démarrage

- [ ] Accès au dépôt privé `github.com/GlennEriss/songo-fish` confirmé
      (déjà collaborateur invité) — sans ça, rien ci-dessous n'est
      accessible.
- [ ] Télécharger `model_v0.2.0.onnx` depuis la release
      [`model-v0.2.0`](https://github.com/GlennEriss/songo-fish/releases/tag/model-v0.2.0)
      (§1).
- [ ] Ajouter `Microsoft.ML.OnnxRuntime` au projet C# (NuGet).
- [ ] Lire dans l'ordre : §0 (pourquoi réseau seul ≠ suffisant) → §2
      (features d'entrée, à reproduire exactement) → §3 (algorithme de
      recherche) → §4 (profondeur vs budget de temps — **lire avant de
      choisir une configuration**, source d'une vraie confusion pendant
      le développement).
- [ ] En cas de doute sur une formule ou un comportement de règle, le
      code Python de référence est cité à chaque section correspondante
      (fichier + fonction) — c'est la source de vérité en cas d'écart.
- [ ] Toute question bloquante : contacter Glenn (propriétaire du dépôt),
      pas de canal de support séparé pour l'instant.

## 0. Résumé en une phrase

**Le réseau seul est faible. Le réseau + une recherche alpha-beta autour
est fort.** Il ne suffit pas de charger le modèle et de prendre le coup
qu'il préfère (`argmax` de la tête policy) — il faut le brancher comme
guide (ordonnancement des coups + évaluation des feuilles) à l'intérieur
d'une vraie recherche multi-coups. C'est ce couple qui bat un adversaire
minimax classique, pas le réseau isolément (mesuré : le réseau seul perd
92% de ses parties contre un minimax profondeur 4 ; réseau+recherche bat
un minimax profondeur 14/5s de réflexion).

## 1. Où récupérer le modèle

L'entraînement a tourné en local (pas sur GCP — GCP n'a servi qu'à
**générer les données d'entraînement**, jamais à entraîner ni héberger le
modèle), donc les fichiers ne sont pas dans le dépôt git (`data/` est
volontairement exclu, ce sont des artefacts régénérables). Le modèle est
distribué via une **release GitHub** :

**https://github.com/GlennEriss/songo-fish/releases/tag/model-v0.2.0**

| Fichier (pièce jointe de la release) | Contenu |
|---|---|
| `model_v0.2.0.onnx` | **Le fichier à utiliser côté C#.** Réseau exporté au format ONNX, chargeable directement via `Microsoft.ML.OnnxRuntime` (NuGet), sans dépendance Python côté jeu. |
| `model_v0.2.0.pt` | Poids PyTorch bruts (référence Python uniquement, pas utilisable tel quel en C#). |

L'architecture exacte (largeur, nombre de blocs, dropout) de la version
chargée est donnée au §2 ci-dessous — pas besoin d'un autre fichier pour
ça.

Une future version (0.3.0, etc.) sera publiée de la même façon, sous un
nouveau tag `model-vX.Y.Z` — vérifier la description du dépôt ou demander
la dernière release si ce document n'a pas été mis à jour entre-temps.

Pour régénérer l'export ONNX après un futur ré-entraînement (usage
interne, côté Python) :

```bash
.venv/bin/python apps/trainer/scripts/export_onnx.py --version champion
```

Vérifié : les sorties ONNX sont identiques (à ~1e-6 près, précision
float32 normale) aux sorties PyTorch sur des entrées aléatoires.

## 2. Le réseau : architecture et entrée/sortie

Réseau minuscule (~105 000 paramètres) — pas besoin de GPU côté C#, un
forward CPU est de l'ordre de la dizaine de microsecondes.

```
Entrée [batch, 33] (float32)
  -> Linear(33, 128) + ReLU
  -> 3x ResidualBlock(128) :  x -> ReLU(Linear(x)) -> Linear -> ReLU(+x)
  -> trois têtes independantes depuis la sortie du tronc (128) :
       policy_logits [batch, 7]   -- un score par coup local 0..6
       wdl_logits    [batch, 3]   -- victoire / nul / défaite (avant softmax)
       q_values      [batch, 7]   -- valeur estimée par coup (peu utilisée en pratique)
```

`width=128`, `num_blocks=3`, `dropout=0` en inférence (le dropout ne
s'applique qu'à l'entraînement). Ces valeurs sont dans
`registry.json` → `architecture` pour la version chargée ; ne pas les
supposer fixes si une future version change la taille.

### 2.1. Construire le vecteur d'entrée (33 valeurs) — À REPRODUIRE EXACTEMENT

C'est le point le plus facile à mal porter silencieusement (le réseau ne
plantera pas, il donnera juste des résultats médiocres). Le plateau
interne est un tableau de **16 entiers** :

```
index  0-5  : les 6 cases de semis du joueur 1
index  6    : case de semis "de bord" du joueur 1 (grain unique -> son propre magasin, pas l'adversaire)
index  7-12 : les 6 cases de semis du joueur 2
index  13   : case de semis "de bord" du joueur 2
index  14   : MAGASIN du joueur 1 (graines déjà capturées, hors jeu)
index  15   : MAGASIN du joueur 2
```

⚠️ Piège déjà rencontré côté Python : les magasins sont aux index 14/15,
**pas** aux cases 6/13 (qui sont juste les dernières cases de semis de
chaque joueur, avec la règle du grain unique).

**Étape 1 — canonicaliser** (le réseau n'a jamais vu autre chose que "mon
camp d'abord") :

```
si c'est au tour du joueur 1 :
    canonique = plateau tel quel
sinon (tour du joueur 2) :
    canonique[0..6]  = plateau[7..13]   # mes cases de semis
    canonique[7..13] = plateau[0..6]    # cases adverses
    canonique[14]    = plateau[15]      # mon magasin
    canonique[15]    = plateau[14]      # magasin adverse
```

**Étape 2 — calculer le masque des 7 coups légaux** (`legal_mask`, 7
booléens, un par case de semis du joueur au trait canonique 0..6) — c'est
la même logique de règles que le reste du moteur (Interdit/Solidarité/RSI),
à réutiliser depuis l'implémentation C# des règles du jeu, pas à
redériver.

**Étape 3 — construire les 33 features** (`TOTAL_SEEDS = 70`) :

```
features[0..15]  = canonique[i] / 70.0                          # 16 compteurs normalisés
features[16..22] = 1.0 si legal_mask[i] sinon 0.0                # 7 valeurs
features[23]     = (canonique[14] - canonique[15]) / 70.0        # store_diff
features[24]     = somme(canonique[0..6])  / 70.0                # my_territory
features[25]     = somme(canonique[7..13]) / 70.0                # opp_territory
features[26]     = somme(legal_mask) / 7.0                       # my_mobility
features[27]     = nb_coups_legaux_adverse(canonique) / 7.0      # opp_mobility (0 si adversaire fini)
features[28]     = 1.0 si le joueur peut transmettre a l'adversaire sinon 0.0   # can_transmit_mine
features[29]     = 1.0 si l'adversaire peut transmettre sinon 0.0               # can_transmit_opp
features[30]     = 1.0 si somme(canonique[7..13]) == 0 sinon 0.0  # opp_side_empty (camp adverse assèché)
features[31]     = 1.0 si somme(canonique[0..6])  == 0 sinon 0.0  # my_side_empty
features[32]     = somme(canonique[0..13]) / 70.0                 # seeds_in_play (proxy de phase de partie)
```

`opp_mobility`, `can_transmit_mine/opp` demandent de rejouer les règles
"comme si" c'était le tour de chaque joueur sur le plateau canonique — pas
de raccourci ici, c'est un calcul de règles à part entière (transmission/
solidarité), voir `packages/songo_ai/model/features.py::observation_features`
pour la référence Python exacte si un doute survient à l'implémentation.

### 2.2. Lire les sorties

- **Coup à jouer (mode "réseau seul", faible, sert de repli)** : `argmax`
  de `policy_logits` **parmi les coups légaux uniquement** (mettre les
  logits des coups illégaux à `-infini` avant l'argmax).
- **Évaluation d'une position (utilisée par la recherche, cf. §3)** :
  `softmax(wdl_logits)` → `(p_victoire, p_nul, p_défaite)`, puis
  `lean = p_victoire - p_défaite` (dans `[-1, 1]`), et enfin
  `score = lean * 300.0` — l'échelle 300 est choisie pour rester dans le
  même ordre de grandeur que l'heuristique de secours (différence de
  graines × 10), voir §3.

## 3. La recherche (ce qui rend le réseau fort)

SongoFish = **negamax/alpha-bêta avec approfondissement itératif**, table
de transposition, et le réseau branché à deux endroits précis — **aucune
recherche par Monte-Carlo (pas de MCTS/PUCT)**, c'est de l'alpha-bêta
classique.

```
fonction songofish_decide(position, budget_temps, profondeur_max):
    meilleur_coup_jusquici = premier coup légal  # filet de securité
    pour profondeur = 1, 2, 3, ... jusqu'a profondeur_max:
        si temps_ecoule >= budget_temps: arreter la boucle
        resultat = negamax_alphabeta(position, profondeur, -inf, +inf)
        si la recherche a ete interrompue en cours de route (budget depasse):
            arreter la boucle SANS utiliser ce resultat partiel
        meilleur_coup_jusquici = resultat.meilleur_coup
    retourner meilleur_coup_jusquici

fonction negamax_alphabeta(position, profondeur, alpha, beta):
    si position terminee: retourner evaluation_terminale(position)  # cf. ci-dessous
    si profondeur == 0: retourner evaluation_reseau(position)       # tete WDL, cf. §2.2
    coups = coups_legaux(position)
    ordonner `coups` par ordre decroissant de policy_logits du reseau (§2.2)
      -- exception : si un coup vient de la table de transposition pour
         cette position, il passe toujours en premier, avant meme le tri policy
    meilleur = -infini
    pour chaque coup dans coups (dans cet ordre) :
        enfant = jouer(position, coup)
        score = -negamax_alphabeta(enfant, profondeur-1, -beta, -alpha)
        si score > meilleur: meilleur = score
        alpha = max(alpha, score)
        si alpha >= beta: arreter la boucle (coupure)   # elagage alpha-beta standard
    enregistrer (position -> meilleur coup, score, profondeur) dans la table de transposition
    retourner meilleur

evaluation_terminale(position) :
    # une position terminee doit TOUJOURS dominer l'evaluation du reseau
    diff = (mon_score_final - score_final_adverse)   # magasin + territoire restant
    retourner diff * 1000.0
```

Deux optimisations supplémentaires existent côté Python (voir
`packages/songo_ai/search/negamax.py`) et valent la peine d'être portées
si le temps le permet, mais ne sont **pas indispensables** pour un premier
portage fonctionnel :

- **PVS (Principal Variation Search)** : après le premier coup, tester les
  suivants avec une fenêtre `[alpha, alpha+epsilon]` très étroite ("est-ce
  que ce coup fait mieux qu'alpha ?") avant de refaire une recherche
  complète seulement si la réponse est "peut-être" — gain mesuré ~10-30%
  de nœuds en moins, à ordre de coups égal.
- **Quiescence** : à `profondeur == 0`, si le dernier coup était une
  capture, continuer à chercher quelques coups de plus (uniquement les
  captures) avant d'évaluer, pour ne jamais couper l'évaluation en pleine
  séquence de récolte. Évite les erreurs tactiques grossières à
  l'horizon de recherche.

**Ce qui n'est PAS nécessaire ni recommandé** : killer moves / history
heuristic — testés côté Python, ils dégradaient l'élagage de ~26% au
Songo (7 coups possibles seulement, le tri par policy est déjà très
informatif ; ce qui marche aux échecs ne marche pas forcément ici). Ne
pas les porter.

## 4. LE POINT CRITIQUE : profondeur vs budget de temps

**"Mettre la recherche à 14" ne veut RIEN dire tout seul.** La profondeur
est un plafond, pas un objectif — l'approfondissement itératif s'arrête
dès que le budget de temps est épuisé et rejoue le meilleur coup trouvé
jusque-là, même si la profondeur demandée n'a pas été atteinte.

Mesure réelle sur la position initiale (réseau v0.2.0 + recherche
optimisée, position par position, cumulatif) :

| Profondeur atteinte | Temps cumulé |
|---|---|
| 7  | ~1,0 s |
| 8  | ~2,5 s |
| 9  | ~5,5 s |
| 10 | ~10,4 s |
| 11 | ~17,7 s |
| 12 | ~39,1 s |
| 13 | ~92,9 s |
| **14** | **~197 s (3 min 17)** |

La croissance est exponentielle à partir de la profondeur 9 — chaque
palier double quasiment le précédent. **Il n'existe donc pas de "faire
profondeur 14 en quelques secondes"** : soit on donne un gros budget de
temps (minutes) et on atteint réellement 14, soit on donne un budget de
quelques secondes et la recherche s'arrêtera d'elle-même bien avant 14
(vers 7-8), en jouant quand même le meilleur coup trouvé à cette
profondeur atteinte.

**Ce qui s'est passé concrètement pendant les tests** : la commande
`songofish:champion:14` avec un budget de 2 secondes (comme configuré
alors sur la table de test) n'a en réalité atteint que la **profondeur
7**, pas 14 — et a quand même battu un minimax profondeur 14 avec 5
secondes de réflexion. Ce n'est pas une anomalie : ça illustre exactement
le rôle du réseau (cf. §0) — un réseau qui guide bien la recherche compense
largement un plafond de profondeur non atteint, face à une recherche
"aveugle" (heuristique brute) qui doit creuser beaucoup plus loin pour
la même force.

### Recommandation pour le jeu en C#

**Toujours configurer un budget de temps par coup, jamais une profondeur
seule.** Exemple de configuration raisonnable pour du temps réel :

- **Coup rapide (< 2s)** : profondeur plafond 10-12 (jamais atteinte en
  pratique à ce budget), budget 1-2s → force déjà solide grâce au réseau
- **Coup réfléchi (adversaire "difficile", quelques secondes tolérées)** :
  budget 5-10s, profondeur plafond 14+ (peu importe, ne sera pas atteint)
- **Analyse hors-ligne / IA très forte sans contrainte de temps** : budget
  de plusieurs dizaines de secondes à quelques minutes si le jeu le permet
  (tour par tour asynchrone)

Le plafond de profondeur ne sert qu'à éviter une recherche infinie sur un
budget de temps très généreux — c'est un garde-fou, pas un réglage de
force à ajuster en priorité.
