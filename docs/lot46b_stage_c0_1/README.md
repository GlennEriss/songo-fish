# Lot46B Stage C0.1 — durcissement du runner

`arena_runner.py` sépare le manifeste scientifique immuable des métadonnées d'exécution. La sérialisation JSON canonique produit les empreintes du manifeste, du planning et des jeux. Chaque ouverture crée exactement deux jeux avec côtés inversés; les budgets restent attachés aux candidats, y compris pour C2 asymétrique.

SQLite (`WAL`, `synchronous=FULL`) assure insertion transactionnelle, unicité des `game_id`, reprise par différence schedule/results et journalisation des conflits. Une répétition identique est idempotente; un contenu divergent est rejeté. Seules les paires complètes sont exposées à l'analyse appariée.

L'export est écrit dans un fichier temporaire, rouvert pour contrôle, puis publié par renommage atomique. Un chemin existant est refusé. Le runner est `DRY_RUN` par défaut; l'exécution exige simultanément flag explicite, identifiant autorisé et checkpoints valides.

Les checkpoints représentent des réseaux complets. Aucun échange naïf Policy/Value n'est réalisé car l'encodeur et `state_fusion` sont partagés.

Tests synthétiques uniquement : aucune importation du moteur ou de MCTS, aucune partie réelle. Exemple sûr :

```bash
PYTHONPATH=packages pytest -q packages/songo_ai/tests/test_arena_runner.py
```

Limites : aucun débit T4 mesuré; le futur adaptateur d'exécution réel et la collecte GPU restent à implémenter sous autorisation séparée. C1/C2 ne sont pas lancés.
