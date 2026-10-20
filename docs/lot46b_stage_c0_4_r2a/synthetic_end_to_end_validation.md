# Synthetic end-to-end validation

Le test appelle `production_control_flow`, identique au chemin de production, avec sélecteur de device et executor injectés. Deux résultats synthétiques sont commitées, les côtés sont vérifiés, les snapshots sont restaurés et une reprise retourne zéro pending game.

Les fixtures portent un marqueur interdit au validateur de production. Aucun moteur ou MCTS réel n'est exécuté.
