# Lot46B Stage C0.2 — intégration et readiness GPU

`ArenaAdapter` traduit un `ScheduledGame` vers la boucle existante par injection de `game_executor`; il ne duplique ni moteur ni MCTS. En dry-run, il valide manifeste, empreinte moteur, checkpoints, ouvertures, réglages MCTS, planning et SQLite sans charger de modèle ni appeler l'exécuteur.

L'exécution est fail-closed : autorisation structurée, identifiant approuvé, empreinte exacte, provenance checkpoint et readiness doivent tous correspondre. Aucun identifiant n'est approuvé en C0.2. Les checkpoints sont des réseaux complets; `BLOCKED_SHARED_ENCODER` demeure.

Les ouvertures portent source, split, ordre, référence d'état et SHA256 canonique. Aucune ouverture aléatoire de remplacement n'est permise. Les réglages imposent bruit racine désactivé, température 0, masque légal et terminalité moteur; les budgets restent attachés aux candidats.

`cuda_readiness` rapporte disponibilité, GPU, versions, mémoire libre, device et précision sans lancer MCTS. `PerformanceMetrics` prépare les compteurs, sans mesure réelle en C0.2.

`SQLiteSnapshotManager` utilise `sqlite3.Connection.backup`, contrôle `PRAGMA integrity_check`, SHA256 et empreinte d'expérience, puis publie des générations immuables. Une restauration étrangère/corrompue ou un écrasement est refusé. Google Drive n'offrant pas les garanties POSIX locales, on conserve toujours la dernière génération validée et on ne copie jamais directement un DB+WAL actif.

Colab sûr : monter Drive, cloner le commit figé, vérifier archives/checkpoints, appeler `prepare_lot46b_stage_c0_2.py --expected-commit <sha>`, créer/restaurer un snapshot validé, puis s'arrêter. Il n'existe aucune cellule autorun d'arène. Déconnexions/quota/kernel : restaurer uniquement la dernière génération dont SHA256 et fingerprint passent.

Tests C0.2 : mocks/fakes uniquement; aucun moteur réel ou `SongoMCTS`. Limites : mapping réel non validé par partie, checkpoints Drive non revérifiés ici, débit T4 inconnu, C1/C2 non autorisés.
