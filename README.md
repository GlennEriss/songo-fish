# SongoFish

Refonte de l'IA du jeu Songo (Awélé) en Python : moteur de règles, professeur
Alpha-Beta profond, dataset annoté, réseau multi-têtes. Plan directeur complet
dans [`docs/Plan_directeur_final_IA_Songo_SongoFish_v3_0.docx`](docs/Plan_directeur_final_IA_Songo_SongoFish_v3_0.docx).

## Structure

```
songo/
├── packages/
│   └── songo_ai/       # coeur partage : regles, recherche, professeur,
│                        # generation, dataset, modele, evaluation
├── apps/
│   ├── trainer/         # generation de dataset, entrainement, tournois (runtime local/gcp)
│   └── table/            # table de jeu interactive (a venir) : jouer contre
│                          # le modele entraine et differents niveaux de minimax
├── docs/
│   ├── trainer/
│   └── table/
└── data/                 # datasets, checkpoints, cache d'annotations (non versionne)
```

`songo_ai` est un package partage : toute app qui a besoin de jouer une
partie, d'evaluer une position ou de charger le reseau importe le meme code,
jamais une copie.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,perf,train]"
```

## Tests

```bash
.venv/bin/pytest
```
