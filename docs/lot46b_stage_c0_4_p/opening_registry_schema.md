# Schéma proposé

Le moteur figé est entièrement reconstructible par `RawSongoState(board[16], player_to_move)` : cases physiques 0–13, magasins 14–15, compteurs entiers non négatifs, somme 70, joueur physique 1 ou 2. Le registre proposé encode explicitement `schema`, `board` et `player_to_move` en JSON canonique ASCII trié.

- `PHYSICAL_STATE_FINGERPRINT = SHA256(canonical_state_bytes)`
- `OPENING_RECORD_FINGERPRINT = SHA256(canonical_record_bytes)`
- `OPENING_REGISTRY_FINGERPRINT = SHA256(canonical_registry_bytes)`

Chaque record exige aussi `opening_id`, `source_id`, `source_sha256`, `split` et `source_record_id`. Les états terminaux, sans coup légal, dupliqués ou sans provenance sont rejetés. Aucune canonicalisation par miroir/joueur n'est appliquée.
