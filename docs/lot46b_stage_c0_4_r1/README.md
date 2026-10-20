# LOT46B Stage C0.4-R1

R1 fournit l'infrastructure du futur smoke test réel C0.4 sans exécuter de partie ni de recherche MCTS. Le dry-run historique reste séparé et inchangé.

- Runner futur : `apps/trainer/scripts/run_lot46b_stage_c0_4_real.py`
- Contrats : `packages/songo_ai/evaluation/stage_c0_4_real.py`
- Tests synthétiques : `packages/songo_ai/tests/test_stage_c0_4_r1.py`
- Limite de concurrence : Google Drive n'offre pas de compare-and-swap atomique ; le runner impose donc une confirmation opérateur single-runner et refuse l'exécution automatique concurrente.

R1 n'installe aucun fichier d'autorisation de production.
