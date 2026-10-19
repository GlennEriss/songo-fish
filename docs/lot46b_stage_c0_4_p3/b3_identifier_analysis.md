# Analyse des identifiants B3

Au commit historique `e804a41b80c6b81380b5fe26138c907308362227`, `_paired_rows` dans `packages/songo_ai/training/lot46b_b3.py` écrit `position_id = str(row["fingerprint"])`. Les lignes viennent de `prepare_context(...)["splits"]["validation"]`; les configurations Standard et Unconstrained sont explicitement vérifiées comme identiques et alignées.

`position_id` est donc un SHA-256 dans l’espace de noms du `fingerprint` du jeu final Lot45, et non un identifiant opaque B3. La jointure vérifiée est `B3.position_id = Lot45-final.fingerprint`. Les 994 identifiants sont uniques et chacun trouve une ligne unique parmi 10 000.

La procédure B3 a évalué hors ligne tout le split de validation aligné. Elle n’a pas conservé l’état dans son export, mais le jeu final Lot45 vérifié le conserve. Chaque état est relu via `RawSongoState`, puis ré-empreint avec le schéma canonique `songo-opening-registry-v1`.

Champs B3 observés : `position_id` (string, unique, espace Lot45 fingerprint), `group` (string), `occurrences` (integer). Aucun `game_id`, `occurrence_id`, `shard_id`, `source_row` ou état physique n’est présent dans les lignes B3 exportées.
