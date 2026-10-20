# Authorization contract

L'autorisation est un JSON égal octet-logiquement au binding produit par `authorization_binding`. Elle lie : experiment ID, fingerprint du manifeste, registre, ouverture, modèles, hashes des checkpoints, MCTS512, deux parties, affectation des côtés, scope et version du protocole.

Une chaîne générique, un champ supplémentaire, un champ absent ou une différence de valeur provoque un arrêt. R1 utilise uniquement des fixtures temporaires dans les tests et ne crée aucune autorisation active.
