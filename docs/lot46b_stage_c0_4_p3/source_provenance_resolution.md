# Résolution de provenance de la source

- `SOURCE_CONTENT_INTEGRITY`: `VERIFIED` pour le membre compressé Lot45 réellement audité.
- `SOURCE_HISTORICAL_IDENTITY`: `VERIFIED` par manifeste, commit et SHA-256 du membre.
- `SOURCE_DERIVATION_CHAIN`: `VERIFIED_PARTIAL`; le sous-ensemble humain est attribué et ses états sont présents dans la sélection dérivée.
- `SOURCE_ORIGINAL_BYTES_AVAILABILITY`: `UNAVAILABLE` après inspection des archives Drive, exports, Git et caches disponibles.
- `SOURCE_INDEPENDENCE`: `PARTIALLY_INDEPENDENT`; 51 chevauchements d’entraînement connus sont exclus.
- `SOURCE_LEAKAGE_RISK`: `B3_CLEAR`, avec zéro chevauchement parmi 994/994 états récupérés.

Le SHA-256 `49eb…9667` couvre les octets du membre gzip `data/colab_bridge/lot45_selection.jsonl.gz`, pas l’archive externe ni le JSONL décompressé. Le SHA-256 historique `17d9…ed18` et la taille 586283 de `data/real_matches/match_moves_v1.jsonl` sont consignés, mais ses octets originaux n’ont pas été retrouvés. Une reconstruction équivalente ne serait pas une récupération de l’original.

Classification conservée : `VERIFIED_DERIVATION_CHAIN_ORIGINAL_HASH_RECORDED_ORIGINAL_BYTES_UNAVAILABLE`.
