# Intégrité des sources

La source originale suivie par Git compte 586283 octets et son SHA-256 frais est `17d9ad19921cdbf1591eb54f6c40aa5b710376f6e903491dd39affc605a5ed18`, identique au relevé historique. Les octets originaux sont donc récupérés et vérifiés. Cela établit l'identité du fichier, pas l'authenticité externe des parties humaines.

Pour le dérivé, les représentations sont distinctes : archive `data/colab_bridge/lot45_inputs.tar.gz` = `75ea7844cc942932918c52d7b8ff891fe8c8810e35caa22a18cbd61630254af9`; membre gzip interne `data/colab_bridge/lot45_selection.jsonl.gz` = `49eb0f83265a763d06f7b45032d994227208f4149ad0c9fb67b6209fd8289667`; JSONL décompressé = `596ec818c5d1ef9e4f862e82f5793dea3ca489323a968208f9e4ccfaaf375a54`. Le hash publié désigne exactement les 9880048 octets compressés du membre interne.

Verdict : `SOURCE_DERIVED_INTEGRITY = VERIFIED`; provenance : `ORIGINAL_AND_DERIVED_BYTES_SHA256_VERIFIED_WITH_EXTERNAL_ORIGIN_LIMITATION`.
