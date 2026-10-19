# Rapport final C0.4-P3

## Conclusion

L’audit forensique est achevé. Le `position_id` B3 est l’empreinte historique de la ligne du jeu final Lot45. Les 994 identifiants s’y raccordent univoquement, permettent de reconstruire 994 `RawSongoState` uniques, et donnent zéro chevauchement avec les 440 candidats P2. La fuite B3 est donc résolue.

Le protocole à 400 ouvertures reste numériquement faisable (440 disponibles), mais aucune ouverture n’est classée « définitivement éligible » tant que les octets historiques originaux de `data/real_matches/match_moves_v1.jsonl` ne sont pas récupérés ou que l’exigence de provenance n’est pas explicitement arbitrée. L’amendement reste proposé et non prêt à l’approbation.

## Preuves essentielles

- B3 archive : `4c03c46…c9809`, vérifiée, inspection sûre, 10 membres.
- B2 archive : `11e77ced…ea39`, vérifiée, 18 membres.
- Paired B3 : `62f1fd17…af74`, 994 lignes et 994 identifiants uniques.
- Final Lot45 : `6d2f6380…33cc`, 10 000 lignes et identifiants uniques.
- Source proposée : membre gzip `49eb0f83…9667`, 491 états humains.
- Source originale : hash historique `17d9ad19…ed18`, octets indisponibles.

## Contrôles

Les tests ciblent jointures, espaces de noms, ambiguïtés, absences, doublons, cardinalités, validation `RawSongoState`, empreintes, couverture partielle/complète, chevauchement, SHA-256 et provenance. `21 passed in 1.30s`; la sélection de régressions C0/C0.3/arène donne `41 passed in 1.73s`. La suite complète donne `785 passed, 1 failed in 174.40s`; l’échec préexistant et non lié est `test_end_to_end_three_workers_twelve_shards_with_a_crash`, qui attend un code 137 alors que les trois workers simulés terminent à 0. Aucun jeu, MCTS, modèle ou checkpoint n’entre dans le nouvel outil.

## Arrêt contrôlé

Pas de registre final, approbation, sélection des 400, C0.4, C1, C2, entraînement, self-play, MCTS ou promotion G5. Prochaine action : `RESOLVE_SOURCE_PROVENANCE_GAPS`.
