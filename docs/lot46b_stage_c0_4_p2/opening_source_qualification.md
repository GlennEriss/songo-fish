# Qualification de la source

`LOT45_D_REAL_HUMAN_SUBSET` contient 491 états complets, statiquement valides, non terminaux et uniques, issus de 474 parties identifiées. Sa chaîne documentée remonte à `data/real_matches/match_moves_v1.jsonl` (SHA256 `17d9ad19921cdbf1591eb54f6c40aa5b710376f6e903491dd39affc605a5ed18`), mais les octets originaux ne sont pas dans les bundles disponibles.

Après exclusion exacte de 51 états présents dans l'union d'entraînement connue et de tout recouvrement QDIAG (zéro), il reste 440 états provisoires issus de 429 parties. Aucun état final n'a été sélectionné et aucun registre n'a été construit.

La qualification définitive échoue fermée : les 994 lignes B3 ne transportent ni `board`, ni `player_to_move`, ni table vérifiable `position_id -> physical_state_fingerprint`.
