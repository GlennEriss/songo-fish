# table

Table de jeu Songo interactive, en pygame. Premiere version : mode
spectateur (deux agents s'affrontent, on regarde), plateau simple et
fonctionnel (formes geometriques, pas de texture bois pour l'instant). Le
mode "jouer soi-meme contre un agent" viendra dans une prochaine iteration.

Consomme `packages/songo_ai` comme `apps/trainer` : meme moteur de regles,
meme recherche, meme chargement de modele, aucune duplication.

## Lancer

```bash
.venv/bin/pip install -e ".[table]"   # une fois, ajoute pygame

.venv/bin/python apps/table/src/main.py --player1 minimax:2 --player2 random
.venv/bin/python apps/table/src/main.py --player1 songofish:champion:8 --player2 minimax:4 --delay 0.4
```

- `--player1` / `--player2` :
  - `random`
  - `minimax:<profondeur>`
  - `songofish:<version|champion>[:profondeur]` — reseau + recherche alpha-beta bornee (etape 8, `packages/songo_ai/hybrid/`) : le reseau ordonne les coups et evalue les feuilles, la recherche regarde plusieurs coups a l'avance. Profondeur par defaut 10, mais chaque noeud fait un passage reseau (~1500 noeuds/s au lieu de ~50 000 pour l'heuristique brute) donc la profondeur reellement atteinte depend du budget de 2s/coup
  - Pas de mode "modele seul sans recherche" : un reseau sans lookahead n'a aucun interet a regarder jouer, cf. `packages/songo_ai/hybrid/`
- `--delay` : pause (s) entre la fin d'une animation et le coup suivant
- `R` pendant la partie : recommencer · fermer la fenetre : quitter

## Structure

- `src/config.py` — dimensions, couleurs
- `src/board_view.py` — rendu (cases, piles de pions, magasins, surbrillances)
- `src/animation.py` — anime un coup case par case (semis puis capture), a partir de la trace exposee par `songo_ai.songo.rules.SongoLegacyGame` (`last_sow_trace`/`last_capture_trace`) — pas de logique de regles dupliquee
- `src/controllers.py` — traduit un choix ("random"/"minimax:N"/"songofish:VERSION") en agent jouable
- `src/main.py` — boucle de jeu

Le moteur utilise ici est `SongoLegacyGame` (reference), pas `FastSongoGame`
(Numba) : la vitesse brute n'a aucune importance a l'echelle d'une partie
regardee a l'oeil nu, et la reference expose la trace coup-par-coup dont
l'animation a besoin.

## Tests

```bash
.venv/bin/pytest apps/table/src/test_animation.py
```
