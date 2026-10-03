# SRN v0.1 — contrat d'implementation

Cette implementation est une extension du code existant. Elle ne remplace ni
le MLP historique `SongoNet`, ni les pipelines teacher/Negamax, ni `D_LAB`.
Elle ne modifie pas le moteur de regles.

## Frontieres du lot

- `D_LAB` designe le corpus historique/laboratoire.
- `D_RL` designe exclusivement les futures experiences de self-play.
- Le contrat conceptuel de `D_RL` est `(S, M, pi, z, metadata)` ; `pi`, `z` et
  les comptes de visites peuvent rester absents tant que la recherche ou la
  partie ne les a pas produits.
- Aucun label teacher n'est converti automatiquement en cible RL.
- L'etat `S` conserve les 16 compteurs physiques et `player_to_move`. Aucune
  symetrie ou canonicalisation n'est supposee.

## Graphe et tenseurs

Les cases 0 a 13 sont les 14 noeuds. Les magasins 14 et 15 restent dans le
contexte global. Les relations de la baseline sont uniquement `NEXT` et
`PREV`, deux cycles diriges sur les 14 noeuds.

Les huit features d'un noeud sont : graines normalisees, ownership P1,
ownership P2, ownership par le joueur au trait, position absolue, position
locale, indicateur de frontiere et distance a la frontiere. Le contexte global
contient les deux magasins, le joueur au trait en one-hot et les graines encore
en jeu. Aucun concept strategique annote (Bidoua, Yinda, Minimax, etc.) n'entre
dans ces features.

Pour un batch de taille `B` :

- entrees noeuds : `[B, 14, 8]` ;
- entrees globales : `[B, 5]` ;
- noeuds d'action : `[B, 7]` (`0..6` pour P1, `7..13` pour P2) ;
- logits Policy : `[B, 7]` ;
- Value scalaire : `[B]`.

## Reseau

Le reseau encode les noeuds et le contexte global, applique `K` blocs de
message passing relationnel avec parametres propres a `NEXT` et `PREV`, puis
utilise une connexion residuelle et une `LayerNorm`. La baseline effectue un
mean pooling et fusionne le resultat avec le contexte global.

La Policy utilise un MLP partage : le logit de l'action `a` depend de
l'embedding du noeud physique associe a `a` et de l'embedding global de l'etat.
La Value scalaire est bornee par `tanh` et exprime le point de vue du joueur au
trait.

Le moteur fournit le masque legal. Les logits illegaux recoivent une grande
valeur negative finie ; apres softmax, leurs probabilites sont remises a zero
exactement puis renormalisees. Cette convention evite les produits
`0 * -inf` pendant l'entropie croisee distributionnelle.

## Explicitement hors lot

MCTS/PUCT, self-play complet, Replay Buffer, boucle RL, `DYNAMIC_SOWING`,
`MIRROR`, attention pooling et tete WDL ne sont pas implementes dans SRN v0.1.
