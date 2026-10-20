# Device selection policy

- `AUTO` tente CUDA, journalise toute indisponibilité/erreur d’initialisation, puis revient sur CPU.
- `CUDA` échoue explicitement si CUDA n’est pas initialisable.
- `CPU` ne consulte et n’initialise jamais CUDA.

Une OOM avant partie permet de reconstruire les réseaux complets sur CPU. Une panne pendant une partie invalide la tentative; un redémarrage repart de l’ouverture gelée avec une nouvelle identité de tentative. Un résultat déjà committé n’est jamais rejoué. Le budget reste 512 sur CPU comme sur GPU.
