# table

Table de jeu Songo interactive (à venir). Objectif : brancher le modèle
entraîné (`packages/songo_ai/model`) et différents niveaux de minimax
(`packages/songo_ai/search`, `packages/songo_ai/generation/agents.py`) pour
jouer en direct et observer comment le modèle entraîné se comporte —
partie contre l'humain, partie contre un niveau donné, ou match entre deux
agents à regarder.

Consommera `packages/songo_ai` comme `apps/trainer`, sans dupliquer les
règles, le moteur de recherche ou le chargement du modèle.

Pas encore implémenté — squelette de dossier seulement pour l'instant.
