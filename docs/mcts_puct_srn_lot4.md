# Lot 4 — MCTS/PUCT pour le SRN

Le MCTS utilise exclusivement le moteur historique gele pour la legalite, les
transitions et les resultats. Le SRN fournit les priors `P(S,a)` et la Value
`V(S)` ; il ne predit jamais un etat successeur.

## Convention de perspective

La Value du SRN est exprimee du point de vue de `player_to_move` dans l'etat
evalue. Chaque statistique `W(S,a)` et `Q(S,a)` du MCTS est exprimee du point
de vue du joueur au trait dans le noeud parent `S`.

Lors du backup, la conversion compare les identites physiques du joueur de la
feuille et du joueur du parent : la valeur est conservee si elles sont egales,
et inversee sinon. Aucune alternance de signe par profondeur n'est supposee.
Le moteur actuel change bien de joueur apres toute transition non terminale
observee ; une transition terminale conserve toutefois le dernier joueur dans
son champ `turn`, puisqu'il n'existe pas de joueur suivant.

## Selection

La formule implementee est :

```text
PUCT(S,a) = Q(S,a)
          + c_puct * P(S,a) * sqrt(1 + N_parent) / (1 + N(S,a))
```

`Q=0` quand une action n'a encore jamais ete visitee. Les actions illegales
ont un score `-inf` et ne peuvent pas etre selectionnees. Une seed controle la
resolution aleatoire des egalites exactes.

## Resultat et cible Policy

Les visites brutes sont retournees avant temperature. La cible `pi` est
calculee separement par `visit_counts_to_policy`. A temperature zero, elle est
un argmax deterministe avec departage par le plus petit indice d'action. La
temperature de cette cible n'est pas une politique de tirage de coup de
self-play.

`root_value` est la moyenne des valeurs sauvegardees sur les aretes de la
racine, dans la perspective du joueur au trait a la racine. Ce n'est pas la
Value brute emise lors de la premiere evaluation du SRN.

Le bruit de Dirichlet est desactive par defaut. Quand il est active, il ne
modifie que les priors des actions legales de la racine.

## Hors lot

Ce module ne contient ni self-play, ni Replay Buffer, ni boucle RL, ni cache
d'inference complexe, ni MCTS parallele, ni virtual loss, ni table de
transposition.
