# Stage C0.4 preexecution report

Le registre P6 et les deux fichiers G4R locaux ont les SHA256 attendus. L’implémentation MCTS existante conserve moteur autoritaire, PUCT, masque légal, perspective Value, seed et compte exact de simulations; elle n’a pas été modifiée ni appelée.

Les tests synthétiques couvrent devices, fallback, réseau complet CPU, sorties finies, registre/checkpoints, planning, inversion, autorisation, tentatives, exactly-once et snapshot/reprise. Le checkpoint final unconstrained n’étant pas présent localement, son chargement réel et son SHA restent `NOT_VERIFIED`; la publication Drive est `NOT_RUN` depuis cette machine.

Conclusion : infrastructure préparée, mais préexécution scientifique non complète tant que le checkpoint final n’est pas revérifié dans Colab. Aucun jeu réel ni MCTS exécuté.
