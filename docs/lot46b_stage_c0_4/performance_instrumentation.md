# Performance instrumentation

Le schéma prépare les temps de chargement, initialisation, inférence, recherche, partie, commit SQLite et snapshot Drive, ainsi que simulations, débit, temps moyen par coup, mémoire maximale et device. CUDA devra être synchronisé autour des mesures; CPU utilise l’horloge monotone.

`REAL_GPU_THROUGHPUT=NOT_MEASURED` et `REAL_CPU_THROUGHPUT=NOT_MEASURED`. Aucun chiffre n’est extrapolé entre devices.
