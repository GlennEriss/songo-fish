# Lot46B Stage C0.3

Validation en deux niveaux strictement séparés. Local : archives synthétiques, contrats de sorties, masque légal, immutabilité et planning C1 de 4 800 jeux. Colab : SHA256 B2/B3 réels, CUDA réel, chargement strict des checkpoints finaux complets, transfert GPU, `eval`, paramètres gelés et deux forwards synthétiques déterministes sous `torch.inference_mode()`.

Le script `validate_lot46b_stage_c0_3.py` n'importe ni boucle d'arène ni MCTS et n'accepte aucun flag d'exécution de partie. Il écrit uniquement dans `experiments/lot46b_stage_c0_3`. Les latences sont étiquetées `synthetic_inference_latency_s` et ne représentent jamais un débit MCTS.

Workflow Colab : monter Drive; cloner et vérifier le commit; restaurer les liens `data` vérifiés; lancer le script avec `--repo`, `--drive-root /content/drive/MyDrive/songo-ai` et `--output /content/drive/MyDrive/songo-ai/experiments/lot46b_stage_c0_3`; tester ensuite le snapshot SQLite synthétique C0.2. Aucun lancement C1/C2.
