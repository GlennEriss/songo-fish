# Architecture du real runner

Le CLI séparé est sûr par défaut et retourne `SAFE_PREFLIGHT_ONLY`. Une future exécution R2 exigera simultanément `--execute`, une autorisation strictement liée au manifeste et `--single-runner-confirmed`.

La couche d'orchestration valide le calendrier immuable de deux parties, journalise chaque tentative séparément des résultats scientifiques, réutilise `ArenaStore` WAL/FULL et publie des snapshots par `SQLiteSnapshotManager`. Les imports du moteur et de `SongoMCTS` sont différés dans `RealAdapters`, après les barrières d'autorisation.

États valides : `CREATED → PREFLIGHT_VALIDATED → AUTHORIZED → RUNNING → COMPLETED`, avec chemins contrôlés `RUNNING → INTERRUPTED → RESUMABLE → RUNNING`.
