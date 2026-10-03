# Lot 5 — Self-play et production locale de D_RL

Le pipeline de self-play est ajoute dans le package `generation`, parallelement
aux trajectoires historiques utilisees par le teacher. Il ne modifie ni
`D_LAB`, ni Minimax/Negamax, ni le moteur.

## Trois politiques distinctes

Pour chaque position, le pipeline conserve trois objets differents :

- les priors reseau `P_theta`, utilises a l'interieur de MCTS ;
- `pi_target`, derivee des visites brutes avec `target_temperature` et stockee
  comme cible d'apprentissage ;
- `pi_play`, derivee des memes visites avec la temperature d'action du ply et
  utilisee uniquement pour tirer le coup effectivement joue.

Les `visit_counts[7]` restent la source non transformee. Changer une
temperature ne les modifie pas.

## Schedule d'exploration

`SelfPlayConfig` utilise `action_temperature` avant `temperature_drop_ply`,
puis `late_action_temperature`. Cette schedule depend uniquement du numero de
ply et ne suppose aucune classification ouverture/milieu/finale du Songo.

Chaque recherche recoit une seed derivee deterministement de la seed globale,
de l'index de partie et du ply. Le tirage des actions dispose d'un RNG separe.
Le bruit Dirichlet MCTS est active par defaut pour le self-play, mais peut etre
desactive pour les tests, le debug et l'evaluation.

## Resultat et troncatures

Une partie porte exactement l'un des statuts suivants :

- `TERMINAL_WIN` ;
- `TERMINAL_DRAW` ;
- `TRUNCATED_MAX_PLIES` ;
- `TRUNCATED_REPETITION`.

Une repetition est identifiee par `(board[16], player_to_move)`. Le seuil est
une protection technique du generateur et non une regle officielle du Songo.
De meme, `max_game_plies` ne transforme jamais une partie longue en nulle.

Pour un terminal reel, `z=+1` si le joueur au trait dans l'exemple est le
vainqueur, `-1` s'il est le perdant et `0` pour un nul moteur. Pour une partie
tronquee, `value_target=None`. Par defaut, ces exemples restent disponibles
pour analyse ou apprentissage Policy-only ; `include_truncated_examples=False`
permet de les exclure completement de la sortie d'entrainement.

## Format local D_RL

Le premier format physique est un JSONL versionne :

```text
format         = songo_d_rl_jsonl
version        = 1
dataset_family = D_RL
```

La premiere ligne est un manifeste. Chaque ligne suivante conserve l'etat
brut, le joueur, le masque legal, les visites, `pi_target`, `z` eventuel et les
metadonnees. Aucun champ teacher ou Minimax n'est lu ou ecrit.

## Hors lot

Ce lot n'entraine pas le SRN et n'ajoute ni Replay Buffer, ni generation
distribuee, ni promotion de champion, ni MCTS parallele, ni batching de
feuilles.
