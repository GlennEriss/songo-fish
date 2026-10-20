# Matrice de décision

## Option A — CONDITIONAL_ACCEPTANCE

Accepter la chaîne vérifiée, conserver les SHA-256 original et dérivé, les 440 empreintes candidates, les exclusions et les risques résiduels; approuver explicitement seed, design partagé et amendement avant toute sélection. Avantage : identité originale et reproductibilité fortes. Limite : l’origine humaine externe n’est pas certifiée par le hash.

## Option B — STRICT_REJECTION

Refuser si l’autorité exige une attestation externe supplémentaire de l’origine humaine. Il n’existe alors aucun registre; une autre source qualifiée est nécessaire avant C0.4. Avantage : standard d’authenticité maximal. Coût : délai et nouveau travail scientifique.

## Option C — DEFERRED_DECISION

Demander spécifiquement une attestation externe ou des métadonnées supplémentaires sur la collecte humaine. Les octets et leur hash étant désormais retrouvés, une nouvelle recherche de fichiers identiques n’apporterait rien.

Recommandation : Option A, `CONDITIONAL_ACCEPTANCE`, car le dérivé et les exclusions sont déterministes, B3 est résolu et la sélection est prospective et indépendante. Cette recommandation n’accepte ni la source ni le risque.
