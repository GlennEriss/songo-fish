# LOT46B — Stage C0.4 preexecution

Cette étape prépare le smoke test technique de deux parties sans l’exécuter. Le registre P6, l’ouverture `LOT46B-C0-OPENING-0001`, MCTS512, l’inversion des côtés, C1=4800 et la marge C2=-0,05 restent gelés.

Le code ajouté ne dépend ni du moteur ni de `SongoMCTS`. L’autorisation réelle est une constante fail-closed distincte du choix de device et vaut `NO`. Aucun jeu, aucune recherche MCTS, aucun entraînement et aucune arène n’ont été lancés.

Le clone local contient le registre et les deux checkpoints G4R vérifiés, mais pas le checkpoint final `DEEP_POLICY_UNCONSTRAINED`. Sa validation réelle reste donc à effectuer dans Colab/Drive avant toute demande d’autorisation.
