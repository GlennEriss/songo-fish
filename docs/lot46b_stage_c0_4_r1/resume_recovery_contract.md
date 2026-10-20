# Resume and recovery

La reprise recharge le manifeste immuable dans `ArenaStore`, vérifie son fingerprint et reconstruit la liste des seules parties non commitées. Une tentative interrompue reste dans `c04_attempts` et ne compte pas comme résultat. Une nouvelle tentative reçoit un nouvel ID.

Chaque résultat complet est transactionnel et unique par scheduled-game ID. Les snapshots sont hashés, restaurés dans une base temporaire et soumis à `PRAGMA integrity_check` avant toute affirmation de récupération.
