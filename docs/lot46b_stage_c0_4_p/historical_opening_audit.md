# Audit historique des ouvertures

Les commits B2 `34ba99a`, B3 `e804a41`, C0 `e8e72ec`, C0.1 `016e69d`, C0.2 `9d018c1` et C0.3 `8f38010` ont été inspectés. Seul C0 introduit `stage_c0_arena_preregistration.json`; il fixe 400 clusters mais ne référence aucun registre, état ou SHA256 de registre. C0.1–C0.3 conservent ce document sans ajouter de registre.

L'archive B2 vérifiée contient entraînement, checkpoints et évaluations hors ligne, pas d'ouvertures d'arène. L'archive B3 vérifiée contient 994 lignes de positions appariées hors ligne, mais les lignes ne contiennent pas l'état physique : seulement identifiant, groupe et métriques dépendantes des candidats. Elles ne permettent ni reconstruction moteur ni sélection indépendante des sorties candidates.

Conclusion : aucune preuve d'un `HISTORICAL_ARTIFACT` préenregistré. Les références au nombre d'ouvertures sont des `HISTORICAL_REFERENCE`, pas un registre. Scenario B.
