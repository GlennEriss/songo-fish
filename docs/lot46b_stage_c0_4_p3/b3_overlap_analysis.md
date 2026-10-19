# Analyse de fuite B3

L’intersection exacte entre les 994 empreintes physiques B3 récupérées et les 440 états provisoires vaut zéro. Aucune exclusion B3 supplémentaire n’est requise et 440 candidats provisoires restent.

Ce résultat était cohérent avec la construction P2 : les 440 excluaient déjà toute intersection avec les 10 000 états du jeu final Lot45, dont les 994 lignes B3 sont un sous-ensemble de validation. P3 rend cette relation démontrable par la lignée historique et les états, plutôt que supposée.

`VALIDATION_LEAKAGE_STATUS = CLEAR` concerne strictement la fuite physique B3. Il ne transforme pas la provenance partielle de la source humaine en provenance complète.
