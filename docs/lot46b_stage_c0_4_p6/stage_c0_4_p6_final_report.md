# Lot46B Stage C0.4-P6 — rapport final

L'amendement approuvé a été vérifié sur ses octets exacts : SHA-256 `2e4dae3b58beeee4f15b17c72cf2c000729f3ae49e44147cc0a2601b29788662`. La nouvelle fiche d'approbation, append-only et limitée à la construction du registre, conserve explicitement le risque résiduel d'authenticité externe.

Le membre gzip source vérifié est `49eb0f83265a763d06f7b45032d994227208f4149ad0c9fb67b6209fd8289667`; la source humaine originale est `17d9ad19921cdbf1591eb54f6c40aa5b710376f6e903491dd39affc605a5ed18`. Les 491 états ont été relus. Le préfixe Lot45 gelé exclut 49 états et D_RL en exclut 2, soit 51 états distincts et 440 candidats.

`SHA256_RANK_V1` avec le seed `3538971373371044805` a sélectionné exactement 400 états. Les IDs vont de `LOT46B-C0-OPENING-0001` à `LOT46B-C0-OPENING-0400`. Deux constructions dans des répertoires temporaires séparés ont produit les mêmes octets et le même SHA-256 : `73cf111f5b8837075916c592c7f5c8cb973740becee674f94c84c240ebebf68e`.

Le registre final contient zéro doublon, zéro recouvrement exact d'entraînement et zéro recouvrement exact B3. La compatibilité statique avec six comparaisons, 400 ouvertures partagées et inversion des côtés produit 4800 enregistrements planifiés synthétiques. Aucun jeu n'a été exécuté.

Tous les artefacts gelés sont écrits atomiquement; une tentative d'écrasement divergent échoue. C0 historique, moteur, checkpoints et protocole statistique restent inchangés.

Tests : `54 passed in 3.53s` pour P6 et les régressions d'arène; `820 passed in 194.78s` pour la suite complète.
